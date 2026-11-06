import time, numpy as np, torch, json
from src.models.edgeproto import get_model
from src.evaluation.evaluate_edgeproto_hybrid import SmallAutoencoder, load_config
cfg=load_config('configs/h1_mirai.yaml'); m=get_model(cfg,num_classes=7)
try: m.load_state_dict(torch.load('models/calib/h1_mirai_s42/best_edgeproto.pt',map_location='cpu'))
except Exception: m.load_state_dict(torch.load('models/ciciot50k_h1_mirai_base/best_edgeproto.pt',map_location='cpu'))
m.eval(); ae=SmallAutoencoder(48)
try: ae.load_state_dict(torch.load('models/calib/h1_mirai_s42_ae.pt',map_location='cpu'))
except Exception: pass
ae.eval()
X=torch.randn(2000,4,12); Xf=X.view(2000,48)
with torch.no_grad():
    for _ in range(50): _=m(X[:1]); _=ae(Xf[:1])   # warmup
    t0=time.perf_counter()
    for i in range(2000): _=m(X[i:i+1]); _=ae(Xf[i:i+1])
    dt=(time.perf_counter()-t0)/2000*1000
json.dump({'ms_per_sample_batch1_pinned':round(dt,4),'samples':2000,'note':'core0-only via taskset; MemoryMax 512M via systemd-run wrapper'},
          open('results/edge_bench.json','w'),indent=2)
print(f'EDGE BENCH: {dt:.4f} ms/sample batch=1 pinned-core')
