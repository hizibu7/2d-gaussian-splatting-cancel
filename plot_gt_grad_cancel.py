import numpy as np,json,os
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
G={'fur':[105,82,50],'det':[97,3,28],'sim':[40,8,35]}
COL={'fur':'r','det':'orange','sim':'b'}
fig,axs=plt.subplots(3,3,figsize=(15,12))
R={}
for ar,(g_,scl) in zip(axs,G.items()):
 for ax,s in zip(ar,scl):
  p=f'eval/gg_s{s}.npz'
  if not os.path.exists(p):continue
  d=np.load(p)
  v=d['valid']&d['v2']&(d['view_count']>=3)
  c=d['cancel'][v];gt=d['gt_grad'][v];sc=d['scale'][v]
  st=np.percentile(sc,75);bg=sc>=st
  ax.scatter(gt[~bg],c[~bg],s=2,alpha=0.2,c='gray')
  ax.scatter(gt[bg],c[bg],s=4,alpha=0.4,c=COL[g_])
  if bg.sum()>10:
   co=np.polyfit(gt[bg],c[bg],1)
   r=np.corrcoef(gt[bg],c[bg])[0,1]
   ax.text(0.05,0.95,f's={co[0]:.2e} r={r:.2f}',transform=ax.transAxes,va='top',fontsize=8)
   R[s]={'grp':g_,'slope':float(co[0]),'r':float(r)}
  ax.set_xlabel('GT|grad|');ax.set_ylabel('cancel');ax.set_title(f's{s}({g_})')
plt.tight_layout();plt.savefig('eval/gg_scatter.png',dpi=110);plt.close()
json.dump(R,open('eval/gg_R.json','w'),indent=1);print('saved')
