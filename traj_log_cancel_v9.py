import os, time, torch
import numpy as np
import diff_surfel_rasterization as dsr
_S={"init":False,"data":[],"out":None,"elapsed":0.0,"calls":0}
def step(it, g, photo_loss, viewpoint_cam, mp, K=1000):
    if not _S["init"]:
        os.makedirs(mp, exist_ok=True)
        _S["out"]=os.path.join(mp,"cancellation_v9.npz"); _S["init"]=True
    t0=time.time()
    P=g._xyz.shape[0]
    dsr.set_cancel_buffers(P, g._xyz.device)
    try:
        _ = torch.autograd.grad(photo_loss, g._xyz, retain_graph=True, allow_unused=True)
    except Exception as e:
        print(f"[cancel@{it}] grad fail: {e}", flush=True); return
    bufs = dsr.get_cancel_buffers()
    lg = dsr.get_last_grads()
    w_sum = bufs.get("w", torch.empty(0))
    wd = bufs.get("wd", torch.empty(0))
    wd2 = bufs.get("wd2", torch.empty(0))
    wd3 = bufs.get("wd3", torch.empty(0))
    wd4 = bufs.get("wd4", torch.empty(0))
    n_rays = bufs.get("n_rays", torch.empty(0))
    eps = 1e-8
    w_safe = w_sum.clamp(min=eps)
    mean_d = wd / w_safe
    var_d = (wd2 / w_safe - mean_d**2).clamp(min=0)
    std_d = var_d.sqrt().clamp(min=eps)
    m3 = wd3/w_safe - 3*mean_d*wd2/w_safe + 2*mean_d**3
    m4 = wd4/w_safe - 4*mean_d*wd3/w_safe + 6*(mean_d**2)*wd2/w_safe - 3*mean_d**4
    skew_d = m3 / (std_d**3 + eps)
    kurt_d = m4 / (var_d**2 + eps) - 3.0
    if lg.get("transMat") is None: print(f"[cancel@{it}] no transMat grad"); return
    summed_t = lg["transMat"].norm(dim=-1)
    sum_t = bufs["transMat"]
    cancel_t = 1.0 - summed_t / (sum_t + 1e-8)
    summed_m = lg["mean2D"].norm(dim=-1) if lg.get("mean2D") is not None else torch.zeros_like(sum_t)
    sum_m = bufs["mean2D"]
    cancel_m = 1.0 - summed_m / (sum_m + 1e-8)
    sx_m = bufs.get("mean2D_sx", torch.zeros_like(sum_m)); sy_m = bufs.get("mean2D_sy", torch.zeros_like(sum_m))
    signed_m = torch.sqrt(sx_m**2 + sy_m**2)
    cancel_m2d = 1.0 - signed_m / (sum_m + 1e-8)
    summed_op = lg["opacity"].abs().squeeze(-1) if lg.get("opacity") is not None else torch.zeros_like(sum_t)
    sum_op = bufs["opacity"]
    cancel_op = 1.0 - summed_op / (sum_op + 1e-8)
    scales = g.get_scaling
    max_sc = scales.max(dim=-1).values
    xyz=g.get_xyz
    cam_pos=viewpoint_cam.camera_center
    depth = (xyz - cam_pos.unsqueeze(0)).norm(dim=-1).clamp(min=1e-6)
    focal = float(viewpoint_cam.image_width) / (2.0 * np.tan(viewpoint_cam.FoVx * 0.5))
    footprint_pix = (max_sc * focal / depth)
    idx = torch.randperm(P, device=xyz.device)[:min(K,P)]
    rec = {"iter": it, "n_gauss": P}
    rec["max_scale"] = max_sc[idx].detach().cpu().numpy().astype(np.float32)
    rec["cancel_t"] = cancel_t[idx].detach().cpu().numpy().astype(np.float32)
    rec["cancel_m"] = cancel_m[idx].detach().cpu().numpy().astype(np.float32)
    rec["cancel_op"] = cancel_op[idx].detach().cpu().numpy().astype(np.float32)
    rec["footprint_pix"] = footprint_pix[idx].detach().cpu().numpy().astype(np.float32)
    rec["depth"] = depth[idx].detach().cpu().numpy().astype(np.float32)
    rec["sum_t"] = sum_t[idx].detach().cpu().numpy().astype(np.float32)
    rec["summed_t_norm"] = summed_t[idx].detach().cpu().numpy().astype(np.float32)
    rec["sum_m"] = sum_m[idx].detach().cpu().numpy().astype(np.float32)
    rec["summed_m_norm"] = summed_m[idx].detach().cpu().numpy().astype(np.float32)
    rec["sx_m"] = sx_m[idx].detach().cpu().numpy().astype(np.float32)
    rec["sy_m"] = sy_m[idx].detach().cpu().numpy().astype(np.float32)
    rec["signed_m"] = signed_m[idx].detach().cpu().numpy().astype(np.float32)
    rec["cancel_m2d"] = cancel_m2d[idx].detach().cpu().numpy().astype(np.float32)
    rec["sum_op"] = sum_op[idx].detach().cpu().numpy().astype(np.float32)
    rec["summed_op_norm"] = summed_op[idx].detach().cpu().numpy().astype(np.float32)
    rec["mean_d"] = mean_d[idx].detach().cpu().numpy().astype(np.float32)
    rec["var_d"] = var_d[idx].detach().cpu().numpy().astype(np.float32)
    rec["skew_d"] = skew_d[idx].detach().cpu().numpy().astype(np.float32)
    rec["kurt_d"] = kurt_d[idx].detach().cpu().numpy().astype(np.float32)
    rec["w_sum"] = w_sum[idx].detach().cpu().numpy().astype(np.float32)
    rec["n_rays"] = n_rays[idx].detach().cpu().numpy().astype(np.float32) if n_rays.numel()>0 else np.zeros(len(idx), dtype=np.float32)
    rec["idx"] = idx.detach().cpu().numpy().astype(np.int32)
    rec["gauss_id"] = g.gauss_id[idx].detach().cpu().numpy().astype(np.int64) if hasattr(g, "gauss_id") and g.gauss_id.numel() > 0 else np.full(len(idx), -1, dtype=np.int64)
    _S["data"].append(rec)
    # Reset buffers to size 0 so main backward skips norm_sum atomicAdd
    for k in dsr._ns: dsr._ns[k] = torch.empty(0, device=g._xyz.device)
    el=time.time()-t0
    _S["elapsed"]+=el; _S["calls"]+=1
    if it%2000==0:
        print(f"[v9@{it}] {el*1000:.0f}ms tot={_S['elapsed']:.0f}s avg_cancel_t={cancel_t.mean().item():.3f} corr~{((max_sc*cancel_t).sum()/(max_sc.sum()*cancel_t.mean()+1e-8)).item():.2f}",flush=True)

def finalize():
    if _S["out"] is None or not _S["data"]: return
    out={}
    for k in _S["data"][0]:
        if k in ("iter","n_gauss"): out[k]=np.array([r[k] for r in _S["data"]])
        else: out[k]=np.stack([r[k] for r in _S["data"]])
    np.savez_compressed(_S["out"],**out)
    print(f"[v9] saved {_S['out']} calls={_S['calls']} total={_S['elapsed']:.1f}s",flush=True)
