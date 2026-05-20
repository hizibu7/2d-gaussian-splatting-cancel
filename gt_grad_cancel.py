import os,sys,torch,numpy as np
import torch.nn.functional as F
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
p.add_argument('--out',type=str,required=True)
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
sm_acc=torch.zeros(P0,device=dev);sg_acc=torch.zeros(P0,device=dev);nr_acc=torch.zeros(P0,device=dev)
grad_acc=torch.zeros(P0,device=dev);view_count=torch.zeros(P0,device=dev)
sx_k=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=dev).view(1,1,3,3)
sy_k=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=dev).view(1,1,3,3)
def sobel(img):
    lum=0.299*img[0]+0.587*img[1]+0.114*img[2]
    lum=lum.unsqueeze(0).unsqueeze(0)
    gx=F.conv2d(lum,sx_k,padding=1)[0,0]
    gy=F.conv2d(lum,sy_k,padding=1)[0,0]
    return torch.sqrt(gx*gx+gy*gy)
for vi in range(min(a.n_views,len(cams))):
    cam=cams[vi]
    dsr.set_cancel_buffers(P0,dev)
    for q in [g._xyz,g._rotation,g._scaling,g._opacity,g._features_dc,g._features_rest]:
        if q.grad is not None: q.grad.zero_()
    pkg=render(cam,g,pipe,bg)
    gt=cam.original_image.cuda()
    loss=0.8*l1_loss(pkg['render'],gt)+0.2*(1.0-ssim(pkg['render'],gt))
    loss.backward()
    b=dsr.get_cancel_buffers()
    sm=b.get('color',torch.zeros(P0,device=dev)).clone()
    fdc=g._features_dc.grad
    sg=fdc[:,0,:].norm(dim=1)/SH0 if fdc is not None else torch.zeros(P0,device=dev)
    nr=b.get('n_rays',torch.zeros(P0,device=dev)).clone()
    sm_acc+=sm;sg_acc+=sg;nr_acc+=nr
    H,W=gt.shape[1],gt.shape[2]
    grad_map=sobel(gt)
    proj=cam.full_proj_transform.detach()
    xyz_h=torch.cat([g._xyz,torch.ones(P0,1,device=dev)],dim=1)
    p4=xyz_h@proj
    w=p4[:,3:4].clamp(min=1e-6)
    pix=(p4[:,:3]/w)
    u=(pix[:,0]+1)*W/2
    v=(pix[:,1]+1)*H/2
    iv=(u>=0)&(u<W)&(v>=0)&(v<H)&(p4[:,3]>0)
    ui=u.long().clamp(0,W-1)
    vj=v.long().clamp(0,H-1)
    samp=grad_map[vj,ui]
    grad_acc[iv]+=samp[iv].detach()
    view_count[iv]+=1
cancel=torch.zeros(P0,device=dev)
v=(nr_acc>=4)&(sm_acc>0)
cancel[v]=1.0-sg_acc[v]/(sm_acc[v]+1e-12)
cancel[~v]=0.5
gt_grad=torch.zeros(P0,device=dev)
v2=view_count>0
gt_grad[v2]=grad_acc[v2]/view_count[v2]
scl=g.get_scaling.detach().max(dim=1).values
print(f'cancel mean={cancel[v].mean():.3f} gt_grad mean={gt_grad[v2].mean():.3e}')
np.savez(a.out,cancel=cancel.cpu().numpy(),gt_grad=gt_grad.cpu().numpy(),valid=v.cpu().numpy(),v2=v2.cpu().numpy(),scale=scl.cpu().numpy(),view_count=view_count.cpu().numpy())
print('saved',a.out)
