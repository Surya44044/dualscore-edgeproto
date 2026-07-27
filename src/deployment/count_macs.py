import torch
from ptflops import get_model_complexity_info
from src.models.edgeproto import get_model
from src.models.baselines import MLPClassifier
import yaml
import argparse

def count_macs(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    print("--- EdgeProto ---")
    model_edge = get_model(config, num_classes=8) # Dummy 8 classes
    macs_e, params_e = get_model_complexity_info(model_edge, tuple((4, 12)), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs_e))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params_e))
    
    print("\n--- MLP Classifier ---")
    model_mlp = MLPClassifier(num_classes=8)
    macs_m, params_m = get_model_complexity_info(model_mlp, tuple((4, 12)), as_strings=True,
                                           print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs_m))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params_m))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default_cpu_config.yaml")
    args = parser.parse_args()
    
    count_macs(args.config)
