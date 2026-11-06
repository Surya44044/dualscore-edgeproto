"""Mode selection across seeds 42/123/2025 using per-seed calib models + AEs. Eval only."""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'.')
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import load_config, SmallAutoencoder
from src.evaluation.oscr import compute_oscr
ROOT=Path('.'); OUT=ROOT/'results/tables/modesel_3seed'; OUT.mkdir(parents=True,exist_ok=True)
EXPS=['h1_mirai','h2_recon','h3_spoofing','h4_web_bruteforce','h5_dos','h5b_dos_ddos']
SEEDS=[42,123,2025]
def load_split(exp,split):
    df=pd.read_csv(f'data/splits/{exp}_{split}.csv'); lab=df.pop('label')
    return df.values.astype(np.float32), lab
def one(exp,seed):
    Xtr,ytr=load_split(exp,'train'); Xva,yva=load_split(exp,'val')
    Xtk,ytk=load_split(exp,'test_known'); Xtu,_=load_split(exp,'test_unknown')
    classes=sorted(set(ytr)); cl2i={c:i for i,c in enumerate(classes)}
    ytr_n=np.array([cl2i[c] for c in ytr]); yva_n=np.array([cl2i[c] for c in yva])
    cfg=load_config(f'configs/{exp}.yaml')
    m=get_model(cfg,num_classes=len(classes))
    m.load_state_dict(torch.load(ROOT/f'models/calib/{exp}_s{seed}/best_edgeproto.pt',map_location='cpu')); m.eval()
    ae=SmallAutoencoder(48); ae.load_state_dict(torch.load(ROOT/f'models/calib/{exp}_s{seed}_ae.pt',map_location='cpu')); ae.eval()
    with torch.no_grad():
        etr=m.encoder(torch.tensor(Xtr.reshape(-1,4,12)))
        protos=m.compute_prototypes(etr,torch.tensor(ytr_n),len(classes))
    def dists(X):
        with torch.no_grad():
            _,d=m(torch.tensor(X.reshape(-1,4,12)),prototypes=protos); return d.numpy()
    def aes(X):
        with torch.no_grad():
            xf=torch.tensor(X.reshape(len(X),-1)); return ((ae(xf)-xf)**2).mean(1).numpy()
    Dv,Dk,Du=dists(Xva),dists(Xtk),dists(Xtu)
    Ev,Ek,Eu=aes(Xva),aes(Xtk),aes(Xtu)
    m0,s0=Dv.min(1).mean(),Dv.min(1).std()+1e-8; me,se=Ev.mean(),Ev.std()+1e-8
    predv=Dv.argmin(1); means,stds={},{}
    for c in range(len(classes)):
        vals=Dv[predv==c].min(1)
        means[c]=vals.mean() if len(vals) else 0.0; stds[c]=vals.std()+1e-8 if len(vals) else 1e-8
    def pz(D):
        pr=D.argmin(1); return (D.min(1)-np.array([means[c] for c in pr]))/np.array([stds[c] for c in pr])
    za_v=(Ev-me)/se; std_v=((Dv.min(1)-m0)/s0+za_v)/2; cal_v=0.25*pz(Dv)+0.75*za_v
    mode_of={}
    for c in range(len(classes)):
        mask=yva_n==c; y=np.r_[np.zeros((~mask).sum()),np.ones(mask.sum())]
        a1=roc_auc_score(y,np.r_[std_v[~mask],std_v[mask]]); a2=roc_auc_score(y,np.r_[cal_v[~mask],cal_v[mask]])
        mode_of[c]='cal' if a2>=a1 else 'std'
    zg_t=(np.r_[Dk,Du].min(1)-m0)/s0; za_t=(np.r_[Ek,Eu]-me)/se
    std_t=(zg_t+za_t)/2; cal_t=0.25*np.r_[pz(Dk),pz(Du)]+0.75*za_t
    pr_t=np.r_[Dk,Du].argmin(1)
    sel=np.array([cal_t[i] if mode_of[pr_t[i]]=='cal' else std_t[i] for i in range(len(pr_t))])
    y=np.r_[np.zeros(len(Dk)),np.ones(len(Du))]
    au=float(roc_auc_score(y,sel))
    preds=Dk.argmin(1); labels=np.array([cl2i[c] for c in ytk])
    oscr=float(compute_oscr(-sel[:len(Dk)],-sel[len(Dk):],preds==labels))
    return {'exp':exp,'seed':seed,'AUROC':au,'OSCR':oscr,'modes':{classes[c]:mode_of[c] for c in range(len(classes))}}
recs=[]
for exp in EXPS:
    for s in SEEDS:
        try:
            r=one(exp,s); recs.append(r)
            json.dump(r,open(OUT/f'{exp}_s{s}_modesel.json','w'),indent=2)
            print("MODE {0} seed{1} AUROC={2:.4f} OSCR={3:.4f}".format(exp, s, r["AUROC"], r["OSCR"]), flush=True)
        except Exception as e:
            print('FAIL',exp,s,repr(e)[:150],flush=True)
# aggregate vs saved s2025 results
import csv, collections
agg=collections.defaultdict(list)
for r in recs: agg[r['exp']].append(r)
rows=[]
for exp,rs in agg.items():
    au=[x['AUROC'] for x in rs]; os_=[x['OSCR'] for x in rs]
    rows.append({'exp':exp,'n_seeds':len(rs),'AUROC_mean':round(float(np.mean(au)),4),
                 'AUROC_std':round(float(np.std(au,ddof=1)),4) if len(au)>1 else 0,
                 'OSCR_mean':round(float(np.mean(os_)),4),
                 'per_seed_AUROC':'/'.join(f"{x['AUROC']:.4f}" for x in sorted(rs,key=lambda z:z['seed']))})
with open(OUT/'modesel_3seed_summary.csv','w',newline='') as f:
    if rows: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("MODESEL DONE", len(recs),"runs")
