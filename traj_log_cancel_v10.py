import os, torch
import numpy as np
_S={"init":False,"data":[],"out":None,"calls":0}

def log_at_densify(iteration, gaussians, max_grad, min_opacity, extent, mp):
    if not _S["init"]:
        os.makedirs(mp, exist_ok=True)
        _S["out"]=os.path.join(mp,"densify_log_v10.npz"); _S["init"]=True
    grads = (gaussians.xyz_gradient_accum / gaussians.denom.clamp(min=1)).flatten()
    grads_safe = grads.detach().cpu().numpy().astype(np.float32)
    max_sc = gaussians.get_scaling.max(dim=-1).values.detach().cpu().numpy().astype(np.float32)
    opac = gaussians.get_opacity.flatten().detach().cpu().numpy().astype(np.float32)
    pt = (grads >= max_grad).cpu().numpy()
    is_big = (gaussians.get_scaling.max(dim=-1).values > gaussians.percent_dense * extent).cpu().numpy()
    ws = pt & is_big
    wc = pt & ~is_big
    wp = (opac < min_opacity)
    rec = {"iter":iteration,"n_gauss":len(grads_safe),"grad":grads_safe,"max_scale":max_sc,
           "opacity":opac,"pt":pt.astype(np.uint8),"ws":ws.astype(np.uint8),
           "wc":wc.astype(np.uint8),"wp":wp.astype(np.uint8),"max_grad":float(max_grad), "gauss_id":gaussians.gauss_id.detach().cpu().numpy().astype(np.int64) if hasattr(gaussians,"gauss_id") and gaussians.gauss_id.numel()>0 else np.full(len(grads_safe),-1,dtype=np.int64)}
    _S["data"].append(rec); _S["calls"] += 1
    if _S["calls"] % 10 == 1:
        n=len(grads_safe);ps=pt.sum();sp=ws.sum();cl=wc.sum()
        print(f"[v10@{iteration}] N={n} pass={ps}",flush=True)

def finalize():
    if _S["out"] is None or not _S["data"]: return
    out = {}
    for k in ["iter","n_gauss","max_grad"]:
        out[k] = np.array([r[k] for r in _S["data"]])
    for k in ["grad","max_scale","opacity","pt","ws","wc","wp","gauss_id"]:
        out[k] = np.array([r[k] for r in _S["data"]], dtype=object)
    np.savez_compressed(_S["out"], **out)
