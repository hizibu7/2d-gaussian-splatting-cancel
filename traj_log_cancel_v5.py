import os, time, torch
import numpy as np
import diff_surfel_rasterization as dsr
_S={"init":False,"data":[],"out":None,"elapsed":0.0,"calls":0}
def _measure_cancel(g, loss):
    if loss is None: return None, None
    try:
        if float(loss.item())==0.0: return None, None
    except Exception: return None, None
    P = g._xyz.shape[0]
    dsr.set_cancel_buffers(P, g._xyz.device)
    try:
        torch.autograd.grad(loss, g._xyz, retain_graph=True, allow_unused=True)
    except Exception:
        return None, None
    bufs = dsr.get_cancel_buffers()
    lg = dsr.get_last_grads()
    sum_t = bufs.get("transMat", torch.empty(0))
    if sum_t.numel()==0 or lg.get("transMat") is None: return None, None
    summed_t = lg["transMat"].norm(dim=-1)
    cancel = 1.0 - summed_t / (sum_t + 1e-8)
    n = bufs.get("n_rays", torch.empty(0))
    return cancel, n
def step(it, g, photo_loss, dist_loss, normal_loss, total_loss, viewpoint_cam, mp, K=1000):
    if not _S["init"]:
        os.makedirs(mp, exist_ok=True)
        _S["out"]=os.path.join(mp,"cancellation_v5.npz"); _S["init"]=True
    t0=time.time()
    P=g._xyz.shape[0]
    cancel_p, n_p = _measure_cancel(g, photo_loss)
    cancel_d, n_d = _measure_cancel(g, dist_loss)
    cancel_n, n_n = _measure_cancel(g, normal_loss)
    cancel_t, n_t = _measure_cancel(g, total_loss)
    if cancel_t is None: return
    scales = g.get_scaling
    max_sc = scales.max(dim=-1).values
    idx = torch.randperm(P, device=g._xyz.device)[:min(K,P)]
    rec = {"iter":it, "n_gauss":P}
    rec["max_scale"] = max_sc[idx].detach().cpu().numpy().astype(np.float32)
    rec["cancel_total"] = cancel_t[idx].detach().cpu().numpy().astype(np.float32)
    if n_t.numel()>0:
        rec["n_rays_total"] = n_t[idx].detach().cpu().numpy().astype(np.float32)
    if cancel_p is not None:
        rec["cancel_photo"] = cancel_p[idx].detach().cpu().numpy().astype(np.float32)
    if cancel_d is not None:
        rec["cancel_dist"] = cancel_d[idx].detach().cpu().numpy().astype(np.float32)
    if cancel_n is not None:
        rec["cancel_normal"] = cancel_n[idx].detach().cpu().numpy().astype(np.float32)
    _S["data"].append(rec)
    # Reset buffers to avoid stale-size OOB in main backward
    for k in dsr._ns: dsr._ns[k] = torch.empty(0, device=g._xyz.device)
    el=time.time()-t0; _S["elapsed"]+=el; _S["calls"]+=1
    if it%2000==0: print(f"[v5@{it}] {el*1000:.0f}ms", flush=True)
def finalize():
    if _S["out"] is None or not _S["data"]: return
    out={}
    keys = set()
    for r in _S["data"]: keys.update(r.keys())
    for k in keys:
        if k in ("iter","n_gauss"):
            out[k] = np.array([r[k] for r in _S["data"]])
        else:
            arrs = [r.get(k, np.zeros(_S["data"][0].get("max_scale", np.zeros(1000)).shape, dtype=np.float32)) for r in _S["data"]]
            try: out[k] = np.stack(arrs)
            except: continue
    np.savez_compressed(_S["out"],**out)
    print(f"[v5] saved {_S['out']} calls={_S['calls']} total={_S['elapsed']:.1f}s",flush=True)
