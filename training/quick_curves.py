from pathlib import Path
import sys,json,random
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.nn import functional as F
from training.models.convlstm import ConvLSTMNowcaster
from training.losses.forecast import WeightedForecastLoss
OUT=ROOT/'assets'/'training_curves';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(84);np.random.seed(84)
def windows(paths,ch,limit=12):
    out=[]
    for p in paths:
        a=np.load(p,mmap_mode='r')
        for i in range(0,max(0,len(a)-12+1),max(1,(len(a)-11)//limit)):
            b=np.array(a[i:i+12],dtype=np.float32)
            if b.ndim==3:b=b[:,None]
            b=np.nan_to_num(b,nan=0,posinf=1,neginf=0).clip(0,1)
            x=F.interpolate(torch.from_numpy(b[:8]),size=(32,32),mode='area')
            y=F.interpolate(torch.from_numpy(b[8:,0:1]),size=(32,32),mode='area')
            out.append((x,y))
            if len(out)>=limit: return out
    return out
def run(name,subdir,ch):
    paths=sorted((ROOT/subdir).glob('segment_*.npy'))
    n=len(paths);a=max(1,int(n*.7));b=max(1,int(n*.15))
    if a+b>=n:a,b=n-2,1
    tr=windows(paths[:a],ch);va=windows(paths[a:a+b],ch)
    print(name,'files',n,'samples',len(tr),len(va),flush=True)
    if not tr or not va:raise RuntimeError(name+' insufficient windows')
    model=ConvLSTMNowcaster(ch,(8,8),1,3)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    lossfn=WeightedForecastLoss();history={'model':name,'run_type':'diagnostic retraining, reduced spatial resolution and hidden size; NOT original production run','train_loss':[],'val_loss':[]}
    for ep in range(1,4):
        model.train();ts=[]
        random.Random(ep).shuffle(tr)
        for x,y in tr:
            opt.zero_grad();pred=model(x.unsqueeze(0),4);loss=lossfn(pred,y.unsqueeze(0));loss.backward();opt.step();ts.append(loss.item())
        model.eval();vs=[]
        with torch.no_grad():
            for x,y in va:vs.append(lossfn(model(x.unsqueeze(0),4),y.unsqueeze(0)).item())
        history['train_loss'].append(float(np.mean(ts)));history['val_loss'].append(float(np.mean(vs)))
        print(name,ep,history['train_loss'][-1],history['val_loss'][-1],flush=True)
    (OUT/(name+'.json')).write_text(json.dumps(history,indent=2))
    plt.figure(figsize=(8,5));plt.plot(range(1,4),history['train_loss'],marker='o',label='Train');plt.plot(range(1,4),history['val_loss'],marker='o',label='Validation')
    plt.xlabel('Epoch');plt.ylabel('Weighted forecast loss');plt.title(name.replace('_',' ').upper()+' — diagnostic');plt.grid(alpha=.3);plt.legend();plt.tight_layout();plt.savefig(OUT/(name+'.png'),dpi=200);plt.close()
if __name__=='__main__':
    run('radar_convlstm','data/processed/events',1)
    run('dwr_insat_convlstm','data/processed/multimodal',4)
    run('dwr_insat_gfs_convlstm','data/processed/multimodal/gfs_features',23)
    print('ALL NEURAL DIAGNOSTIC CURVES FINISHED',flush=True)
