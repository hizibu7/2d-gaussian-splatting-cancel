"""Patch lgc6_grid_all.py to 9 scene × 3 view layout (27 cells per PNG)."""
TARGET = "lgc6_grid_all.py"
with open(TARGET, 'r') as f:
    src = f.read()

old_block = """    for G in GRADS:
        for mode in ['single', 'multi']:
            fig, axs = plt.subplots(3, 3, figsize=(18, 14))
            for ridx, scan in enumerate(SCAN_ORDER):
                cidx = ridx % 3; rrow = ridx // 3
                ax = axs[rrow, cidx]
                c = cache[scan]
                npz_v0 = c['npzs'][0]
                if npz_v0 is None:
                    ax.text(0.5, 0.5, f's{scan} MISSING', ha='center', va='center')
                    ax.axis('off'); continue
                # single: V=0 mask
                if mode == 'single':
                    sel = np.where(compute_mask_lgc6(npz_v0, G, args.g_abs_mean_min, args.nr_min, args.cm_corr_min))[0]
                else:
                    # multi: intersection across available views
                    masks = []
                    for V in VIEWS:
                        nz = c['npzs'][V]
                        if nz is None: continue
                        masks.append(compute_mask_lgc6(nz, G, args.g_abs_mean_min, args.nr_min, args.cm_corr_min))
                    if not masks:
                        sel = np.array([], dtype=int)
                    else:
                        N = min(m.shape[0] for m in masks)
                        m_all = masks[0][:N]
                        for m in masks[1:]:
                            m_all = m_all & m[:N]
                        sel = np.where(m_all)[0]
                # use V=0 image
                v0 = c['views'][0]
                overlay(ax, v0['img'], c['xyz'], c['R'], c['sca'], v0['proj'], sel)
                ax.set_title(f's{scan} [{grp(scan)}] n={len(sel)}', fontsize=10, color=GROUPS[grp(scan)]['color'] if grp(scan) else 'black')

            plt.suptitle(f'photo | {G} | {mode} | sigmoid-corrected cm', fontsize=14)
            plt.tight_layout()
            out_p = f'{OUT_DIR}/grid_photo_{G}_{mode}.png'
            plt.savefig(out_p, dpi=110, bbox_inches='tight')
            plt.close()
            print(f'wrote {out_p}')"""

new_block = """    for G in GRADS:
        for mode in ['single', 'multi']:
            fig, axs = plt.subplots(9, 3, figsize=(18, 42))
            for ridx, scan in enumerate(SCAN_ORDER):
                c = cache[scan]
                # Compute mask once per scene
                if mode == 'multi':
                    # intersection across all views, applied uniformly to every cell
                    masks = []
                    for V in VIEWS:
                        nz = c['npzs'][V]
                        if nz is None: continue
                        masks.append(compute_mask_lgc6(nz, G, args.g_abs_mean_min, args.nr_min, args.cm_corr_min))
                    if not masks:
                        sel_uniform = np.array([], dtype=int)
                    else:
                        N = min(m.shape[0] for m in masks)
                        m_all = masks[0][:N]
                        for m in masks[1:]:
                            m_all = m_all & m[:N]
                        sel_uniform = np.where(m_all)[0]
                for cidx, V in enumerate(VIEWS):
                    ax = axs[ridx, cidx]
                    nz = c['npzs'][V]
                    if nz is None:
                        ax.text(0.5, 0.5, f's{scan} v{V} MISSING', ha='center', va='center')
                        ax.axis('off'); continue
                    if mode == 'single':
                        # per-view independent mask
                        sel = np.where(compute_mask_lgc6(nz, G, args.g_abs_mean_min, args.nr_min, args.cm_corr_min))[0]
                    else:
                        sel = sel_uniform
                    vimg = c['views'][V]
                    overlay(ax, vimg['img'], c['xyz'], c['R'], c['sca'], vimg['proj'], sel)
                    ax.set_title(f's{scan} [{grp(scan)}] v{V} n={len(sel)}', fontsize=9,
                                 color=GROUPS[grp(scan)]['color'] if grp(scan) else 'black')

            plt.suptitle(f'photo | {G} | {mode} | sigmoid-corrected cm (cm>={args.cm_corr_min})', fontsize=14)
            plt.tight_layout()
            out_p = f'{OUT_DIR}/grid_photo_{G}_{mode}.png'
            plt.savefig(out_p, dpi=100, bbox_inches='tight')
            plt.close()
            print(f'wrote {out_p}')"""

assert old_block in src, "anchor not found"
src = src.replace(old_block, new_block)
with open(TARGET, 'w') as f:
    f.write(src)
print("patched lgc6_grid_all.py to 9 scene x 3 view layout")

