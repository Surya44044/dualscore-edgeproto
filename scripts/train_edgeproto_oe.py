#!/usr/bin/env python3
"""EdgeProto + Outlier Exposure trainer for CICIoT2023 zero-day holdouts.
Pseudo-unknowns: RT-IoT2022 attack flows (cross-dataset OE), scaled with the
holdout train scaler, zero-padded 46->48. Loss = Proto + sep + lambda_oe*OE."""
import os, sys, gc, yaml, torch, random, numpy as np, pandas as pd
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from src.data.feature_mapping import IoTDataset
from src.models.edgeproto import get_model
from src.training.losses import PrototypeLoss, SeparationLoss

FEAT48 = None  # filled from a train CSV's columns

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

def pad48(df, cols):
    for i in range(48 - len(cols)):
        df[f'pad_{i:02d}'] = 0.0
    return df[cols + [f'pad_{i:02d}' for i in range(48 - len(cols))]]

class OEDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], torch.tensor(-1)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data_prefix', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rtiot', default='/home/azureuser/RT_IOT2022')
    ap.add_argument('--n_oe', type=int, default=20000)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lambda_oe', type=float, default=0.5)
    ap.add_argument('--margin_oe', type=float, default=0.3)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    dev = torch.device('cpu')
    bs = config['training'].get('batch_size', 256)
    epochs = args.epochs or config['training'].get('epochs', 30)
    lr = config['training'].get('lr', 1e-3)
    num_cls = config['training'].get('num_classes', None)

    tr_df = pd.read_csv(f'{args.data_prefix}_train.csv')
    val_df = pd.read_csv(f'{args.data_prefix}_val.csv')
    tr_labels = tr_df['label'].values; val_labels = val_df['label'].values
    cols = [c for c in tr_df.columns if c != 'label']
    global FEAT48
    FEAT48 = cols
    tr_df = pad48(tr_df, cols).copy(); val_df = pad48(val_df, cols).copy()
    Xtr = tr_df[cols + [f'pad_{i:02d}' for i in range(48 - len(cols))]].values.astype(np.float32)
    Xval = val_df[cols + [f'pad_{i:02d}' for i in range(48 - len(cols))]].values.astype(np.float32)

    classes = sorted(set(tr_labels))
    cl2i = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cl2i[c] for c in tr_labels]); yval = np.array([cl2i[c] for c in val_labels])
    num_cls = len(classes)

    # scaler from train
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr); Xval = sc.transform(Xval)

    # OE pseudo-unknowns from RT-IoT
    rt = pd.read_csv(args.rtiot, on_bad_lines='skip')
    benign = ['MQTT_Publish', 'Thing_Speak', 'Wipro_bulb']
    rt = rt[~rt['Attack_type'].isin(benign)] if 'Attack_type' in rt.columns else rt
    common = [c for c in cols if c in rt.columns]
    Xoe = rt[common].values.astype(np.float32)
    # align to cols order
    Xoe_full = np.zeros((len(Xoe), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        if c in rt.columns:
            Xoe_full[:, j] = rt[c].values.astype(np.float32)
    Xoe_pad = np.zeros((len(Xoe), 48), dtype=np.float32)
    Xoe_pad[:, :len(cols)] = Xoe_full
    Xoe_pad = sc.transform(Xoe_pad)
    idx = np.random.default_rng(args.seed).choice(len(Xoe_pad), size=min(args.n_oe, len(Xoe_pad)), replace=False)
    Xoe_pad = Xoe_pad[idx]
    print(f'OE pseudo-unknowns: {len(Xoe_pad)} (from {len(rt)} RT-IoT attack rows)', flush=True)

    Xtr = Xtr.reshape(-1, 4, 12); Xval = Xval.reshape(-1, 4, 12); Xoe_pad = Xoe_pad.reshape(-1, 4, 12)
    tr_ds = torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    val_ds = torch.utils.data.TensorDataset(torch.tensor(Xval), torch.tensor(yval))
    oe_ds = OEDataset(Xoe_pad)
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    oe_loader = DataLoader(oe_ds, batch_size=bs, shuffle=True)
    oe_batches = list(oe_loader)

    model = get_model(config, num_classes=num_cls).to(dev)
    proto_c = PrototypeLoss(temperature=config['training'].get('temperature', 0.5))
    sep_c = SeparationLoss(margin=config['training'].get('margin_sep', 0.5))
    lam_sep = config['training'].get('lambda_sep', 0.01)
    lam_oe = args.lambda_oe; margin_oe = args.margin_oe
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
            loss_p = proto_c(dists, y); loss_s = sep_c(protos)
            loss = loss_p + lam_sep * loss_s
            # OE: one random mini-batch of pseudo-unknowns per training batch (standard OE)
            xoe = oe_batches[np.random.randint(len(oe_batches))][0].to(dev)
            eoe = model.encoder(xoe)
            zc_oe = torch.matmul(eoe, protos.t())
            d_oe = torch.clamp(2.0 - 2.0 * zc_oe, min=0.0).min(1).values
            loss = loss + lam_oe * torch.clamp(margin_oe - d_oe, min=0.0).mean()
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
    print('OE training complete.', flush=True)

if __name__ == '__main__':
    main()
