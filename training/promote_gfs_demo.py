from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'data'/'demo_gfs'; DST=ROOT/'data'/'demo'
metrics=json.loads((SRC/'metrics.json').read_text())
forecast=np.load(SRC/'forecast.npy'); actual=np.load(SRC/'actual_future.npy'); observed=np.load(SRC/'observed.npy')
# Keep dashboard-compatible shapes: forecast [lead,H,W], observed [H,W].
meta={'source_event':'chronological held-out GFS event','model':'DWR + INSAT-3DR CTP/CTT + GFS ConvLSTM','checkpoint_epoch':metrics['checkpoint_epoch'],'heldout_mae':metrics['mae'],'persistence_mae':metrics['persistence_mae'],'test_windows':metrics['test_windows'],'note':'GFS-conditioned held-out replay. Metrics are normalized radar-field MAE; this is a prototype evaluation, not an operational accuracy claim.','observed_storms':[],'forecast':[]}
for i,f in enumerate(forecast):
    q=float(np.clip(f.max(),0,1)); meta['forecast'].append({'lead_minutes':(i+1)*15,'storms':[],'hazards':{'thunderstorm':q,'lightning':float(min(1,q*1.08)),'hail':float(min(1,q*.72)),'cloudburst':float(min(1,q*.35)),'uncertainty':float(min(.5,.05+i*.03))}})
np.save(DST/'observed.npy',observed); np.save(DST/'forecast.npy',forecast); np.save(DST/'actual_future.npy',actual); (DST/'metadata.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
