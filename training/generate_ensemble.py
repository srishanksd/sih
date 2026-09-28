"""Generate a stochastic multimodal forecast ensemble on the held-out GFS window."""
from pathlib import Path
import json, sys, numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from training.datasets.multimodal_sequence import MultimodalSequenceDataset
from training.models.convlstm import ConvLSTMNowcaster

DATA=ROOT/'data'/'processed'/'multimodal'/'gfs_features'; CK=ROOT/'training'/'checkpoints_gfs_multimodal'/'best.pt'; OUT=ROOT/'data'/'demo'/'ensemble'; OUT.mkdir(parents=True,exist_ok=True)

def split(files):
    files=sorted(files); n=len(files); a=max(1,int(n*.70)); b=max(1,int(n*.15));
    if a+b>=n: a,b=n-2,1
    return files[:a],files[a:a+b],files[a+b:]

def main(members=24):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _,_,test=split(sorted(DATA.glob('segment_*.npy')))
    ds=MultimodalSequenceDataset(test,8,4,1); x,y=next(iter(DataLoader(ds,batch_size=1,shuffle=False)))
    ck=torch.load(CK,map_location=device,weights_only=False)
    model=ConvLSTMNowcaster(23,(32,64),1,3).to(device); model.load_state_dict(ck['model']); model.eval()
    x=x.to(device); rng=np.random.default_rng(8400); preds=[]
    with torch.no_grad():
        for i in range(members):
            xp=x.clone()
            eps=torch.as_tensor(rng.normal(0,.015,xp.shape),dtype=xp.dtype,device=device)
            # Perturb physical predictors; keep availability/mask channels unchanged.
            xp[:,:,:22]=torch.clamp(xp[:,:,:22]+eps[:,:,:22],0,1)
            preds.append(model(xp,4).clamp(0,1).cpu().numpy()[0])
    samples=np.stack(preds)
    np.save(OUT/'members.npy',samples); np.save(OUT/'mean.npy',samples.mean(0));
    np.save(OUT/'p10.npy',np.quantile(samples,.10,axis=0)); np.save(OUT/'p50.npy',np.quantile(samples,.50,axis=0));
    np.save(OUT/'p90.npy',np.quantile(samples,.90,axis=0)); np.save(OUT/'spread.npy',samples.std(0))
    meta={'members':members,'checkpoint_epoch':ck.get('epoch'),'test_windows':len(ds),'method':'stochastic predictor perturbation','calibrated':False,'note':'Prototype probabilistic ensemble; calibration requires more independent events.'}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2))

if __name__=='__main__': main()
