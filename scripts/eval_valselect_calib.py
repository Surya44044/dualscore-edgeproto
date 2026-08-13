#!/usr/bin/env python3
import sys, numpy as np, torch
from src.data.feature_mapping import get_dataloaders
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config
from sklearn.metrics import roc_auc_score

def main():
    seed = '2025'
    device = torch.device('cpu')
    exps = sys.argv[1:] or ['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
    ws = [0.0, 0.25, 0.5, 0.75, 1.0]
    for exp in exps:
        config = load_config(f'configs/{exp}.yaml')
        tr, val, tk, tu, label_map = get_dataloaders(
            f'data/splits/{exp}_train.csv', f'data/splits/{exp}_val.csv',
            f'data/splits/{exp}_test_known.csv', f'data/splits/{exp}_test_unknown.csv', batch_size=256)
        model = get_model(config, num_classes=len(label_map)).to(device)
        model.load_state_dict(torch.load(f'models/ciciot50k_{exp}_base/best_edgeproto.pt', map_location=device))
        model.eval()
        embs, labs = [], []
        with torch.no_grad():
            for x, y in tr:
                embs.append(model.encoder(x)); labs.append(y)
        protos = model.compute_prototypes(torch.cat(embs), torch.cat(labs), len(label_map))
        def dists(loader):
            out = []
            with torch.no_grad():
                for x, _ in loader:
                    _, dd = model(x.to(device), prototypes=protos)
                    out.append(dd.cpu().numpy())
            return np.concatenate(out)
        Dv = dists(val); Lv = np.array(val.dataset.labels)
        Dk = dists(tk); Du = dists(tu)
        mk, mu = Dk.min(1), Du.min(1)
        y = np.r_[np.zeros(len(mk)), np.ones(len(mu))]
        m0, s0 = Dv.min(1).mean(), Dv.min(1).std() + 1e-8
        zg = np.r_[(mk-m0)/s0, (mu-m0)/s0]
        predv = Dv.argmin(1); means={}; stds={}
        for c in range(len(label_map)):
            vals = Dv[predv==c].min(1)
            means[c] = vals.mean() if len(vals) else 0.0
            stds[c] = vals.std() + 1e-8 if len(vals) else 1e-8
        def pz(D):
            pr = D.argmin(1)
            return (D.min(1) - np.array([means[c] for c in pr])) / np.array([stds[c] for c in pr])
        zp = np.r_[pz(Dk), pz(Du)]
        pv = pz(Dv)
        best_w, best_loo = None, -1.0
        for w in ws:
            aucs = []
            for c in np.unique(Lv):
                mask = (Lv == c)
                zk = w*(Dv[~mask].min(1)-m0)/s0 + (1-w)*pv[~mask]
                zu = w*(Dv[mask].min(1)-m0)/s0 + (1-w)*pv[mask]
                yy = np.r_[np.zeros(len(zk)), np.ones(len(zu))]
                aucs.append(roc_auc_score(yy, np.r_[zk, zu]))
            loo = float(np.mean(aucs))
            if loo > best_loo:
                best_loo, best_w = loo, w
        s_test = best_w*zg + (1-best_w)*zp
        au = roc_auc_score(y, s_test)
        print(f'{exp:<20} best_w={best_w:.2f} (valLOO={best_loo:.3f}) TEST_AUROC={au:.4f}')

if __name__ == '__main__':
    main()
