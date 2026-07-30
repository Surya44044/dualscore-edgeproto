import os
import torch
import numpy as np
import yaml
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import SmallAutoencoder

class DualScoreModelManager:
    def __init__(self, model_dir: str = "models", config_path: str = "configs/rt_h1_arp_spoofing.yaml"):
        self.device = torch.device("cpu")
        self.demo_mode = False
        self.edgeproto = None
        self.autoencoder = None
        self.prototypes = None
        self.label_map = ["Benign", "DDoS", "DoS", "Recon"]
        self.mean_proto = 0.0
        self.std_proto = 1.0
        self.mean_ae = 0.0
        self.std_ae = 1.0
        self.zero_day_threshold = 0.0 # Normalized score > 0.0 classified as zero-day

        # Attempt to load config & models
        try:
            ep_path = os.path.join(model_dir, "rt_h1_arp_spoofing", "best_edgeproto.pt")
            ae_path = os.path.join(model_dir, "hybrid_autoencoders", "rt_h1_arp_spoofing_ae_known.pt")

            if not (os.path.exists(ep_path) and os.path.exists(ae_path) and os.path.exists(config_path)):
                print(f"[WARN] Model checkpoints missing ({ep_path} or {ae_path}). Starting in demo_mode=True.")
                self.demo_mode = True
                return

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            num_classes = len(self.label_map)
            self.edgeproto = get_model(config, num_classes=num_classes).to(self.device)
            self.edgeproto.load_state_dict(torch.load(ep_path, map_location=self.device))
            self.edgeproto.eval()

            self.autoencoder = SmallAutoencoder(input_dim=48).to(self.device)
            self.autoencoder.load_state_dict(torch.load(ae_path, map_location=self.device))
            self.autoencoder.eval()

            # Dummy initialization of prototypes if needed
            if hasattr(self.edgeproto, "prototypes") and self.edgeproto.prototypes is not None:
                self.prototypes = self.edgeproto.prototypes
            else:
                self.prototypes = torch.zeros((num_classes, 32), device=self.device)

            print("[INFO] DualScore-EdgeProto models successfully loaded into memory.")
        except Exception as e:
            print(f"[EXCEPT] Error loading models ({e}). Falling back to demo_mode=True.")
            self.demo_mode = True

    def predict(self, x_tensor: torch.Tensor):
        """
        x_tensor shape: [B, 4, 12]
        """
        if self.demo_mode or self.edgeproto is None or self.autoencoder is None:
            return {
                "demo_mode": True,
                "prediction": "Unknown (Demo Mode)",
                "nearest_known_class": "N/A",
                "unknown_score": 0.0,
                "zero_day": False,
                "message": "API operating in demo_mode=True (model checkpoints not found on server). Install weights under models/ to enable real inference."
            }

        with torch.no_grad():
            x_tensor = x_tensor.to(self.device)
            embeddings = self.edgeproto.encoder(x_tensor)
            
            # Compute distance to prototypes
            if self.prototypes is not None:
                z_dot_c = torch.matmul(embeddings, self.prototypes.t())
                dists = torch.clamp(2.0 - 2.0 * z_dot_c, min=0.0)
            else:
                dists = torch.zeros((x_tensor.size(0), len(self.label_map)))

            min_dists, nearest_idx = torch.min(dists, dim=1)
            s_proto = min_dists.cpu().numpy()

            # AE reconstruction MSE
            x_flat = x_tensor.view(x_tensor.size(0), -1)
            recon = self.autoencoder(x_flat)
            s_ae = torch.mean((recon - x_flat)**2, dim=1).cpu().numpy()

            # Hybrid score (normalized)
            z_proto = (s_proto - self.mean_proto) / self.std_proto
            z_ae = (s_ae - self.mean_ae) / self.std_ae
            s_hybrid = (z_proto + z_ae) / 2.0

            results = []
            for i in range(len(x_tensor)):
                n_idx = int(nearest_idx[i].item())
                nearest_cls = self.label_map[n_idx] if n_idx < len(self.label_map) else f"Class_{n_idx}"
                unk_score = float(s_hybrid[i])
                is_zero_day = bool(unk_score > self.zero_day_threshold)
                pred_label = "ZERO_DAY_ATTACK" if is_zero_day else nearest_cls

                results.append({
                    "demo_mode": False,
                    "prediction": pred_label,
                    "nearest_known_class": nearest_cls,
                    "unknown_score": round(unk_score, 4),
                    "zero_day": is_zero_day,
                    "message": "Zero-day intrusion anomaly detected by DualScore fusion!" if is_zero_day else "Traffic classified as normal known class."
                })

            return results[0] if len(results) == 1 else results

# Global Manager Instance
model_manager = DualScoreModelManager()
