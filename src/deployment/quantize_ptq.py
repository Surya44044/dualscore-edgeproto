import torch
import torch.ao.quantization as quant
import argparse
import os
import yaml
from src.models.edgeproto import get_model
from src.data.feature_mapping import get_dataloaders

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def quantize_model(config_path, data_prefix, model_path, output_dir):
    config = load_config(config_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data for calibration
    train_csv = f"{data_prefix}_train.csv"
    val_csv = f"{data_prefix}_val.csv"
    train_loader, _, _, _, label_map = get_dataloaders(
        train_csv, val_csv, val_csv, val_csv, batch_size=256
    )
    
    num_classes = len(label_map)
    model = get_model(config, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Set backend to qnnpack for ARM/mobile or fbgemm for x86
    torch.backends.quantized.engine = 'qnnpack'
    
    # Prepare model for static quantization
    # We must insert QuantStub and DeQuantStub if we were modifying the model code,
    # but for simple CNNs we can use torch.ao.quantization.quantize_dynamic or static.
    # Because EdgeProto has Conv1d, static quantization is better.
    # Note: the model needs QuantStub and DeQuantStub. We'll wrap the encoder.
    
    class QuantizedEdgeProto(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.model = model
            self.dequant = torch.ao.quantization.DeQuantStub()
            
        def forward(self, x):
            x = self.quant(x)
            embeds, dists = self.model(x)
            embeds = self.dequant(embeds)
            dists = self.dequant(dists)
            return embeds, dists
            
    q_model = QuantizedEdgeProto(model)
    q_model.eval()
    
    # Attach qconfig
    q_model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    
    # Prepare
    torch.ao.quantization.prepare(q_model, inplace=True)
    
    # Calibrate
    print("Calibrating model...")
    with torch.no_grad():
        for i, (x, y) in enumerate(train_loader):
            q_model(x)
            if i > 10: # Calibrate on ~2500 samples
                break
                
    # Convert
    print("Converting to INT8...")
    torch.ao.quantization.convert(q_model, inplace=True)
    
    out_file = os.path.join(output_dir, "edgeproto_int8_ptq.pt")
    try:
        scripted = torch.jit.script(q_model)
        torch.jit.save(scripted, out_file)
        print(f"Saved TorchScript INT8 model to {out_file}")
    except Exception as e:
        print("WARNING: TorchScript save failed; saving eager quantized checkpoint fallback instead.")
        print(f"TorchScript error: {type(e).__name__}: {e}")
        torch.save({
            "model_state_dict": q_model.state_dict(),
            "format": "eager_quantized_state_dict_fallback",
            "note": "Smoke-test artifact only. Final deployment export should use ONNX/TFLite/stable TorchScript.",
        }, out_file)
        print(f"Saved fallback quantized checkpoint to {out_file}")
    
    # Compare sizes
    fp32_size = os.path.getsize(model_path) / 1024
    int8_size = os.path.getsize(out_file) / 1024
    print(f"FP32 Size: {fp32_size:.2f} KB")
    print(f"INT8 Size: {int8_size:.2f} KB")
    print(f"Reduction: {fp32_size/int8_size:.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_prefix", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/quantized")
    args = parser.parse_args()
    
    quantize_model(args.config, args.data_prefix, args.model_path, args.output_dir)
