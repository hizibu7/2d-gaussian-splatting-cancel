import os,sys,torch,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser
import diff_surfel_rasterization as dsr
SH0=0.28209479177387814
p=ArgumentParser();mp=ModelParams(p,sentinel=True);pp=PipelineParams(p)
p.add_argument('--iteration',type=int,default=30000)
p.add_argument('--n_views',type=int,default=20)
p.add_argument('--view_idx',type=int,default=0)
p.add_argument('--top_n',type=int,default=1500)
p.add_argument('--opa_thr',type=float,default=0.5)
p.add_argument('--view_cnt_thr',type=int,default=8)
p.add_argument('--out',type=str,required=True)
a=p.parse_args(sys.argv[1:]);a.eval=False
def quat_to_R(q):
    q=q/(np.linalg.norm(q,axis=1,keepdims=True)+1e-12)
    qw,qx,qy,qz=q[:,0],q[:,1],q[:,2],q[:,3]
    R=np.zeros((q.shape[0],3,3))
    R[:,0,0]=1-2*(qy*qy+qz*qz);R[:,0,1]=2*(qx*qy-qw*qz);R[:,0,2]=2*(qx*qz+qw*qy)
    R[:,1,0]=2*(qx*qy+qw*qz);R[:,1,1]=1-2*(qx*qx+qz*qz);R[:,1,2]=2*(qy*qz-qw*qx)
    R[:,2,0]=2*(qx*qz-qw*qy);R[:,2,1]=2*(qy*qz+qw*qx);R[:,2,2]=1-2*(qx*qx+qy*qy)
    return R
g=GaussianModel(a.sh_degree if a.sh_degree else 3)
sc=Scene(a,g,load_iteration=a.iteration,shuffle=False)
bg=torch.tensor([0,0,0],dtype=torch.float32,device='cuda')
pipe=pp.extract(a)
for q in [g._xyz,g._rotation,g._scaling,g._opacity,g._features_dc,g._features_rest]:
    q.requires_grad_(True)
P0=g._xyz.shape[0];dev='cuda'
cams=sc.getTrainCameras()
print(f'P={P0} views={len(cams)}')
ch={'color':{'sg_acc':torch.zeros(P0,device=dev),'sm_acc':torch.zeros(P0,device=dev)},
    'tm':{'sg_acc':torch.zeros(P0,device=dev),'sm_acc':torch.zeros(P0,device=dev)},
    'op':{'sg_acc':torch.zeros(P0,device=dev),'sm_acc':torch.zeros(P0,device=dev)}}
nr_acc=torch.zeros(P0,device=dev)
def zero_grad():
    for q in [g._xyz,g._rotation,g._scaling,g._opacity,g._features_dc,g._features_rest]:
        if q.grad is not None: q.grad.zero_()
for vi in range(min(a.n_views,len(cams))):
    cam=cams[vi]
    z=torch.zeros(P0,device=dev)
    # photo loss for color+opacity
    dsr.set_cancel_buffers(P0,dev);zero_grad()
    pkg=render(cam,g,pipe,bg)
    gt=cam.original_image.cuda()
    Lp=0.8*l1_loss(pkg['render'],gt)+0.2*(1.0-ssim(pkg['render'],gt))
    Lp.backward()
    b=dsr.get_cancel_buffers()
    ch['color']['sm_acc']+=b.get('color',z).clone()
    fdc=g._features_dc.grad
    ch['color']['sg_acc']+=(fdc[:,0,:].norm(dim=1)/SH0) if fdc is not None else z
    ch['op']['sm_acc']+=b.get('opacity',z).clone()
    og=g._opacity.grad
    ch['op']['sg_acc']+=og.abs().squeeze() if og is not None else z
    nr_acc+=b.get('n_rays',z).clone()
    # normal loss for transMat
    dsr.set_cancel_buffers(P0,dev);zero_grad()
    pkg=render(cam,g,pipe,bg)
    rn=pkg['rend_normal'];sn=pkg['surf_normal']
    Ln=(1-(rn*sn).sum(dim=0))[None].mean()
    Ln.backward()
    b=dsr.get_cancel_buffers()
    ch['tm']['sm_acc']+=b.get('transMat',z).clone()
    rg=g._rotation.grad;sg=g._scaling.grad
    rnn=rg.norm(dim=1) if rg is not None else z
    snn=sg.norm(dim=1) if sg is not None else z
    ch['tm']['sg_acc']+=torch.sqrt(rnn*rnn+snn*snn)
    if vi%5==0: print(f'v{vi}',flush=True)
# compute cancels
cancels={}
for k,v in ch.items():
    c=torch.zeros(P0,device=dev)
    valid=(v['sm_acc']>1e-9)
    c[valid]=1.0-v['sg_acc'][valid]/(v['sm_acc'][valid]+1e-12)
    cancels[k]=c.cpu().numpy()
# floater filter
op_val=torch.sigmoid(g._opacity).detach().squeeze().cpu().numpy()
vc=nr_acc.cpu().numpy()
non_floater=(op_val>a.opa_thr)&(vc>=a.view_cnt_thr*100)
print(f'non_floater: {non_floater.sum()}/{P0} ({100*non_floater.mean():.1f}%)')
# render reference view
cam=cams[a.view_idx]
with torch.no_grad():
    pkg=render(cam,g,pipe,bg)
    img=pkg['render'].clamp(0,1).cpu().numpy().transpose(1,2,0)
H,W=img.shape[:2]
