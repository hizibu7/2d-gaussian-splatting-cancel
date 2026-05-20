import os, time, torch
import numpy as np
import diff_surfel_rasterization as dsr

def compute_cancel(gaussians, photo_loss):
    P = gaussians._xyz.shape[0]; dev = gaussians._xyz.device
    dsr.set_cancel_buffers(P, dev)
    try:
        torch.autograd.grad(photo_loss, gaussians._xyz, retain_graph=True, allow_unused=True)
    except Exception as e:
        print(f"[cancel_prune] grad fail: {e}", flush=True)
        for k in dsr._ns: dsr._ns[k] = torch.empty(0, device=dev)
        return None
    b = dsr.get_cancel_buffers()
    sm = b.get("mean2D", torch.zeros(P, device=dev))
    sx = b.get("mean2D_sx", torch.zeros(P, device=dev))
    sy = b.get("mean2D_sy", torch.zeros(P, device=dev))
    nr = b.get("n_rays", torch.zeros(P, device=dev))
    sg = torch.sqrt(sx*sx + sy*sy)
    cm2d = 1.0 - sg / (sm + 1e-8)
    for k in dsr._ns: dsr._ns[k] = torch.empty(0, device=dev)
    return cm2d, nr, sm

def get_summed_norm(gaussians, photo_loss):
    """Get ‖∂L/∂μ‖ per Gaussian via autograd hook on mean2D viewspace."""
    # Reuse cancel_buffers approach but return summed_m_norm via PyTorch grad
    P = gaussians._xyz.shape[0]; dev = gaussians._xyz.device
    dsr.set_cancel_buffers(P, dev)
    try:
        torch.autograd.grad(photo_loss, gaussians._xyz, retain_graph=True, allow_unused=True)
    except Exception:
        for k in dsr._ns: dsr._ns[k] = torch.empty(0, device=dev); return None
    lg = dsr.get_last_grads()
    if lg.get("mean2D") is not None:
        sm_norm = lg["mean2D"].norm(dim=-1)
    else:
        sm_norm = torch.zeros(P, device=dev)
    for k in dsr._ns: dsr._ns[k] = torch.empty(0, device=dev)
    return sm_norm

def prune_at(iteration, gaussians, photo_loss, mp):
    P_before = gaussians._xyz.shape[0]
    res = compute_cancel(gaussians, photo_loss)
    if res is None: return 0
    cm2d, nr, sm = res
    sm_norm = get_summed_norm(gaussians, photo_loss)
    if sm_norm is None: return 0
    trapped = (cm2d >= 0.999) & (nr >= 4)
    valid = (nr >= 4)
    if valid.sum() > 10:
        thr = torch.quantile(sm_norm[valid], 0.75)
        q1 = trapped & (sm_norm >= thr)
    else:
        q1 = torch.zeros_like(trapped)
    pm = trapped | q1
    pm = pm & (~torch.isnan(cm2d))
    n = int(pm.sum().item())
    nt=int(trapped.sum().item()); nq=int(q1.sum().item())
    print(f"[cancel_prune@{iteration}] P={P_before} T={nt} Q1={nq} prune={n}",flush=True)
    if n==0: return 0
    gaussians.prune_points(pm)
    return n
