"""Build GFS atmospheric channels aligned to existing DWR+INSAT events."""
from pathlib import Path
import re, json
import numpy as np
import cfgrib
from scipy.ndimage import zoom
from datetime import datetime, timedelta
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'data'/'raw'/'gfs'; MM=ROOT/'data'/'processed'/'multimodal'; OUT=MM/'gfs_features'; OUT.mkdir(parents=True,exist_ok=True)
META=json.loads((MM/'metadata.json').read_text()); TARGET=(128,128)
def norm_temp(x): return np.clip((x-240.0)/80.0,0,1)
def norm_wind(x): return np.clip((x+50.0)/100.0,0,1)
def norm_cape(x): return np.clip(x/4000.0,0,1)
def norm_cin(x): return np.clip((-x)/500.0,0,1)
def resize(a): return zoom(a,(TARGET[0]/a.shape[0],TARGET[1]/a.shape[1]),order=1).astype(np.float32)
def parse_file_time(path):
    m=re.search(r'gfs\.(\d{8})\.t00z\.f(\d{3})\.grib2$',path.name)
    return datetime.strptime(m.group(1),'%Y%m%d')+timedelta(hours=int(m.group(2))) if m else None

def load_feature(path):
    datasets=cfgrib.open_datasets(str(path),indexpath='')
    surface={}; iso=None
    for ds in datasets:
        names=set(ds.data_vars)
        if {'u10','v10'}<=names: surface.update(u10=ds.u10.values,v10=ds.v10.values)
        if {'t2m','d2m'}<=names: surface.update(t2m=ds.t2m.values,d2m=ds.d2m.values)
        if {'cape','cin'}<=names: surface.update(cape=ds.cape.values,cin=ds.cin.values)
        if {'t','u','v'}<=names and 'isobaricInhPa' in ds.coords: iso=ds
    required=['u10','v10','t2m','d2m']; missing=[k for k in required if k not in surface]
    if iso is None or missing: raise RuntimeError(f'Missing required GFS fields in {path.name}: {missing}')
    cape=surface.get('cape',np.zeros_like(surface['t2m'],dtype=np.float32)); cin=surface.get('cin',np.zeros_like(surface['t2m'],dtype=np.float32))
    levels=[850,700,500,300]
    chans=[norm_wind(surface['u10']),norm_wind(surface['v10']),norm_temp(surface['t2m']),norm_temp(surface['d2m']),norm_cape(cape),norm_cin(cin)]
    latv=iso.latitude.values; lonv=iso.longitude.values
    for level in levels:
        t=iso.t.sel(isobaricInhPa=level).values; u=iso.u.sel(isobaricInhPa=level).values; v=iso.v.sel(isobaricInhPa=level).values
        chans.extend([norm_temp(t),norm_wind(u),norm_wind(v)])
    arr=np.stack(chans,axis=0)
    lat_idx=np.where((latv>=20)&(latv<=30))[0]; lon_idx=np.where((lonv>=85)&(lonv<=100))[0]
    if len(lat_idx)==0 or len(lon_idx)==0: raise RuntimeError('GFS crop bounds not found')
    arr=arr[:,lat_idx.min():lat_idx.max()+1,lon_idx.min():lon_idx.max()+1]
    return np.stack([resize(x) for x in arr],axis=0)

def main():
    files=[p for p in sorted(RAW.glob('gfs.202609*.grib2')) if p.name.startswith('gfs.')]; cache={}
    for i,path in enumerate(files,1):
        vt=parse_file_time(path)
        if vt is not None:
            print(f'GFS {i}/{len(files)} {path.name}',flush=True); cache[vt]=load_feature(path)
    times=sorted(cache)
    if not times: raise RuntimeError('No valid GFS files found')
    for event in META['events']:
        p=MM/event['file']; x=np.load(p); start=datetime.fromisoformat(event['start']); end=datetime.fromisoformat(event['end'])
        pair_times=[datetime.fromisoformat(pair['dwr_time']) for pair in META['pairs'] if start<=datetime.fromisoformat(pair['dwr_time'])<=end]
        if len(pair_times)!=x.shape[0]: raise RuntimeError(f'Timestamp/frame mismatch: {event["file"]}')
        gfs=np.zeros((x.shape[0],19,128,128),dtype=np.float32)
        for j,t in enumerate(pair_times):
            nearest=min(times,key=lambda q:abs(q-t)); delta=abs(nearest-t).total_seconds()/60.0
            if delta<=100.0: gfs[j,:18]=cache[nearest]; gfs[j,18]=1.0
        combined=np.concatenate([x,gfs],axis=1); out=OUT/event['file']; np.save(out,combined.astype(np.float32))
        print(f'SAVED {out.name} shape={combined.shape}',flush=True)
    print('DONE',flush=True)

if __name__=='__main__': main()
