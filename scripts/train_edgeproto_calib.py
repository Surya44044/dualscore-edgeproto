#!/usr/bin/env python3
"""Train EdgeProto base (no OE) with per-seed output dirs for the 3-seed calibration re-run."""
import os, sys, gc, yaml, torch, random, numpy as np, pandas as pd
import torch.optim as optim
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from src.models.edgeproto import get_model
from src.training.losses import PrototypeLoss, SeparationLoss

def load_config(path):
    def _resolve(p):
        with open(p) as f:
            d = yaml.safe_load(f) or {}
        inc = d.pop('include', None)
        if inc:
            base = _resolve(str(Path(p).parent / inc))
            for k, v in d.items():
                base[k] = {**base[k], **v} if isinstance(base.get(k), dict) and isinstance(v, dict) else v
            return base
        return d
    return _resolve(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data_prefix', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    dev = torch.device('cpu')
    bs = config['training'].get('batch_size', 256)
    epochs = config['training'].get('epochs', 30)
    lr = config['training'].get('lr', 1e-3)

    tr_df = pd.read_csv(f'{args.data_prefix}_train.csv')
    val_df = pd.read_csv(f'{args.data_prefix}_val.csv')
    tr_labels = tr_df['label'].values; val_labels = val_df['label'].values
    cols = [c for c in tr_df.columns if c != 'label']
    pad = [f'pad_{i:02d}' for i in range(48 - len(cols))]
    Xtr = tr_df[cols + pad].values.astype(np.float32)
    Xval = val_df[cols + pad].values.astype(np.float32)
    classes = sorted(set(tr_labels))
    cl2i = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cl2i[c] for c in tr_labels]); yval = np.array([cl2i[c] for c in val_labels])
    num_cls = len(classes)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr).reshape(-1, 4, 12); Xval = sc.transform(Xval).reshape(-1, 4, 12)

    tr_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    val_ds = TensorDataset(torch.tensor(Xval), torch.tensor(yval))
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    model = get_model(config, num_classes=num_cls).to(dev)
    proto_c = PrototypeLoss(temperature=config['training'].get('temperature', 0.5))
    sep_c = SeparationLoss(margin=config['training'].get('margin_sep', 0.5))
    lam_sep = config['training'].get('lambda_sep', 0.01)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=config['training'].get('weight_decay', 1e-4))

    best = float('inf')
    for ep in range(epochs):
        model.train(); tl = 0.0; n = 0
        for x, y in tr_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            emb = model.encoder(x)
            protos = model.compute_prototypes(emb, y, num_cls)
            zc = torch.matmul(emb, protos.t())
            dists = torch.clamp(2.0 - 2.0 * zc, min=0.0)
            loss = proto_c(dists, y) + lam_sep * sep_c(protos)
            loss.backward(); opt.step()
            tl += loss.item(); n += 1
        tl /= max(n, 1)
        model.eval(); vl = 0.0; m = 0
        with torch.no_grad():
            embs, labs = [], []
            for x, y in tr_loader:
                embs.append(model.encoder(x.to(dev))); labs.append(y)
            gprotos = model.compute_prototypes(torch.cat(embs), torch.cat(labs), num_cls)
            model.prototypes.data.copy_(gprotos)
            for x, y in val_loader:
                _, vd = model(x.to(dev), prototypes=gprotos)
                vl += proto_c(vd, y.to(dev)).item(); m += 1
        vl /= max(m, 1)
        print(f'Epoch {ep+1}/{epochs} - Train Loss: {tl:.4f} - Val Loss: {vl:.4f}', flush=True)
        if vl < best:
            best = vl
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_edgeproto.pt'))
        gc.collect()
    print('Training complete.', flush=True)

if __name__ == '__main__':
    main()
