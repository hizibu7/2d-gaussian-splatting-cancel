import os,sys,torch,numpy as np,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser
import diff_surfel_rasterization as dsr
SH0=0.28209479177387814
p=ArgumentParser();mp=ModelParams(p,sentinel=True);pp=PipelineParams(p)
p.add_argument("--iteration",type=int,default=30000)
p.add_argument("--n_views",type=int,default=20)
p.add_argument("--out",type=str,required=True)
a=p.parse_args(sys.argv[1:]);a.eval=False
g=GaussianModel(a.sh_degree if a.sh_degree else 3)
sc=Scene(a,g,load_iteration=a.iteration,shuffle=False)
bg=torch.tensor([0,0,0],dtype=torch.float32,device='cuda')
pipe=pp.extract(a)
for q in [g._xyz,g._rotation,g._scaling,g._opacity,g._features_dc,g._features_rest]:
    q.requires_grad_(True)
P0=g._xyz.shape[0];dev='cuda'
cams=sc.getTrainCameras()
print(f'P={P0} views={len(cams)}')
LOSSES=['photo','dist','normal']
GRADS=['m2d','op','tm','color','n3d']
BUF={'m2d':'mean2D','op':'opacity','tm':'transMat','color':'color','n3d':'normal3D'}
acc={(L,G):{'sg':torch.zeros(P0,device=dev),'sm':torch.zeros(P0,device=dev),'nr':torch.zeros(P0,device=dev)} for L in LOSSES for G in GRADS}
scl=g.get_scaling.detach().max(dim=1).values.cpu().numpy()
sthr=np.percentile(scl,75)
def gloss(pkg,cam,kind):
 if kind=='photo':
  gt=cam.original_image.cuda()
  return 0.8*l1_loss(pkg['render'],gt)+0.2*(1.0-ssim(pkg['render'],gt))
 if kind=='dist': return pkg['rend_dist'].mean()
 rn=pkg['rend_normal'];sn=pkg['surf_normal']
 return (1 - (rn*sn).sum(dim=0))[None].mean()
def sgrad(name,gm,b):
 z=torch.zeros(P0,device=dev)
 if name=='m2d': return torch.sqrt(b.get('mean2D_sx',z)**2+b.get('mean2D_sy',z)**2)
 if name=='op':
  og=gm._opacity.grad;return og.abs().squeeze() if og is not None else z
 if name=='tm':
  rg=gm._rotation.grad;sg=gm._scaling.grad
  rn=rg.norm(dim=1) if rg is not None else z
  sn=sg.norm(dim=1) if sg is not None else z
  return torch.sqrt(rn*rn+sn*sn)
 if name=='color':
  fdc=gm._features_dc.grad
  return fdc[:,0,:].norm(dim=1)/SH0 if fdc is not None else z
 return torch.sqrt(b.get('normal3D_sx',z)**2+b.get('normal3D_sy',z)**2+b.get('normal3D_sz',z)**2)
for vi in range(min(a.n_views,len(cams))):
    cam=cams[vi]
    for L in LOSSES:
        dsr.set_cancel_buffers(P0,dev)
        for q in [g._xyz,g._rotation,g._scaling,g._opacity,g._features_dc,g._features_rest]:
            if q.grad is not None: q.grad.zero_()
        pkg=render(cam,g,pipe,bg)
        loss=gloss(pkg,cam,L)
        loss.backward()
        b=dsr.get_cancel_buffers()
        nr=b.get('n_rays',torch.zeros(P0,device=dev)).clone()
        for G in GRADS:
            sg=sgrad(G,g,b)
            sm=b.get(BUF[G],torch.zeros(P0,device=dev)).clone()
            acc[(L,G)]['sg']+=sg;acc[(L,G)]['sm']+=sm;acc[(L,G)]['nr']+=nr
    if vi%5==0: print(f'  view {vi}',flush=True)
out={}
for L in LOSSES:
    for G in GRADS:
        d=acc[(L,G)]
        v=(d['nr']>=4)&(d['sm']>0)
        Nv=int(v.sum().item())
        if Nv<10:
            out[f'{L}_{G}']={'N':Nv,'sat':0,'mean':0,'big_sat':0,'big_mean':0}
            continue
        cm=(1.0-d['sg'][v]/(d['sm'][v]+1e-12)).clamp(0,1).cpu().numpy()
        big_v_idx=np.where(v.cpu().numpy())[0]
        big_mask=scl[big_v_idx]>=sthr
        out[f'{L}_{G}']={'N':Nv,'sat':float(100*(cm>=0.999).mean()),'mean':float(cm.mean()),'big_sat':float(100*(cm[big_mask]>=0.999).mean()) if big_mask.any() else 0,'big_mean':float(cm[big_mask].mean()) if big_mask.any() else 0}
        print(f'{L:6}|{G:5}: N={Nv:>6} mean={cm.mean():.3f} sat%={100*(cm>=0.999).mean():.1f}')
json.dump(out,open(a.out,'w'),indent=1)
print('saved',a.out)
out={}
for L in LOSSES:
    for G in GRADS:
        d=acc[(L,G)]
        v=(d['nr']>=4)&(d['sm']>0)
        Nv=int(v.sum().item())
        if Nv<10:
            out[f'{L}_{G}']={'N':Nv,'sat':0,'mean':0,'big_sat':0,'big_mean':0}; continue
        cm=(1.0-d['sg'][v]/(d['sm'][v]+1e-12)).clamp(0,1).cpu().numpy()
        bi=np.where(v.cpu().numpy())[0]
        bm=scl[bi]>=sthr
        out[f'{L}_{G}']={'N':Nv,'sat':float(100*(cm>=0.999).mean()),'mean':float(cm.mean()),'big_sat':float(100*(cm[bm]>=0.999).mean()) if bm.any() else 0,'big_mean':float(cm[bm].mean()) if bm.any() else 0}
        print(f'{L}|{G}: N={Nv} mean={cm.mean():.3f} sat={100*(cm>=0.999).mean():.1f}')
json.dump(out,open(a.out,'w'),indent=1)
print('saved',a.out)
