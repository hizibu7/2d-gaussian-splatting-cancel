import os, time, torch
import torch.nn.functional as F
import numpy as np
_S={"init":False,"data":[],"out":None,"elapsed":0.0,"calls":0}
def step(it, g, photo_loss, image, cam, mp, K=1000, R=3):
    if not _S["init"]:
        os.makedirs(mp, exist_ok=True)
        _S["out"]=os.path.join(mp,"cancellation.npz"); _S["init"]=True
    t0=time.time()
    if not image.requires_grad: return
    try: dI=torch.autograd.grad(photo_loss, image, retain_graph=True)[0]
    except Exception: return
    H,W=dI.shape[-2:]
    N=g._xyz.shape[0]; K=min(K,N)
    idx=torch.randperm(N, device=g._xyz.device)[:K]
    try: gx=torch.autograd.grad(photo_loss, g._xyz, retain_graph=True, allow_unused=True)[0]
    except Exception: gx=None
    if gx is None: return
    gn=gx[idx].norm(dim=-1)
    scales=g.get_scaling
    max_sc=scales[idx].max(dim=-1).values
    xyz=g.get_xyz[idx]
    ones=torch.ones(K,1,device=xyz.device)
    xyzh=torch.cat([xyz,ones],dim=1)
    proj=xyzh @ cam.full_proj_transform
    pw=proj[:,3:4].clamp(min=1e-8)
    proj=proj/pw
    behind=(pw.squeeze() < 1e-6) | (proj[:,2] < 0) | (proj[:,2] > 1)
    u=((proj[:,0]+1)*W*0.5).long().clamp(0,W-1)
    v=((proj[:,1]+1)*H*0.5).long().clamp(0,H-1)
def finalize():
    if _S["out"] is None or not _S["data"]: return
    out={}
    keys=_S["data"][0].keys()
    for k in keys:
        if k in ("iter","n_gauss"):
            out[k]=np.array([r[k] for r in _S["data"]])
        else:
            out[k]=np.stack([r[k] for r in _S["data"]])
    np.savez_compressed(_S["out"],**out)
    print(f"[cancel] saved {_S['out']} calls={_S['calls']} total={_S['elapsed']:.1f}s",flush=True)
