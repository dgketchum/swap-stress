import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from map.learning.mae.train_mae import load_mae


def _zscore(a):
    mu = np.nanmean(a, axis=0, keepdims=True)
    sd = np.nanstd(a, axis=0, keepdims=True) + 1e-6
    out = (a - mu) / sd
    return out


def _make_patch_mask(T, patch_len=30, n_patches=3, seed=None):
    rng = np.random.RandomState(seed)
    L = int(min(patch_len, T))
    m = np.zeros(T, dtype=bool)
    for _ in range(max(1, int(n_patches))):
        if T - L <= 0:
            m[:] = True
            break
        s = int(rng.randint(0, T - L + 1))
        m[s:s + L] = True
    return m


def _load_series(site, vwc_dir, gridmet_dir):
    vfp = os.path.join(vwc_dir, f"{site}.parquet")
    gfp = os.path.join(gridmet_dir, f"{site}.parquet")
    vdf = pd.read_parquet(vfp)[['shallow', 'middle']]
    gdf = pd.read_parquet(gfp)
    return vdf, gdf


def _select_window(vdf, gdf, window_len):
    idx = vdf.index.intersection(gdf.index)
    idx = pd.DatetimeIndex(idx).sort_values()
    T = len(idx)
    W = int(min(window_len, T))
    s0 = max(0, T - W)
    s1 = s0 + W
    sub_idx = idx[s0:s1]
    xv = vdf.loc[sub_idx].values.astype('float32')
    xf = gdf.loc[sub_idx].values.astype('float32')
    return sub_idx, xv, xf


def _plot_met(ax, gdf, label_prefix):
    cols = [c for c in ['pr', 'vpd', 'pet', 'tmmx'] if c in gdf.columns]
    X = _zscore(gdf[cols].values.astype('float32'))
    for i, c in enumerate(cols):
        ax.plot(gdf.index, X[:, i], lw=0.8, label=f"{label_prefix}{c}")
    ax.set_xlim(gdf.index.min(), gdf.index.max())
    ax.set_ylabel('met (z)')
    ax.legend(loc='upper right', ncols=3, fontsize=7)


def _plot_vwc_with_masks(ax, vdf, site, patch_len, n_patches):
    X = vdf[['shallow', 'middle']].values.astype('float32')
    Xz = _zscore(X)
    ax.plot(vdf.index, Xz[:, 0], color='tab:blue', lw=1.5, label='shallow')
    ax.plot(vdf.index, Xz[:, 1], color='tab:orange', lw=1.5, label='middle')
    T = len(vdf)
    seed = abs(hash(site)) % (2 ** 32)
    m = _make_patch_mask(T, patch_len=patch_len, n_patches=n_patches, seed=seed)
    if m.any():
        on = False
        s = 0
        for i, flag in enumerate(m):
            if flag and not on:
                s = i
                on = True
            if on and (not flag or i == T - 1):
                e = i + 1 if (flag and i == T - 1) else i
                ax.axvspan(vdf.index[s], vdf.index[e - 1], color='k', alpha=0.12)
                on = False
    ax.set_ylabel('VWC (z)')
    ax.legend(loc='upper right', fontsize=7)


def _embed_to_image(model, xv, xf):
    x = torch.tensor(_zscore(xv), dtype=torch.float32).unsqueeze(0)
    f = torch.tensor(_zscore(xf), dtype=torch.float32).unsqueeze(0)
    x = x.to(model.device)
    f = f.to(model.device)
    with torch.no_grad():
        e = model.embed(x, feats=f)
    vec = e.squeeze(0).detach().cpu().numpy()
    img = vec.reshape(8, 8)  # likely error if embedding dimension != 64
    return img


def plot_mae_examples(site_a='pretrain_2678', site_b='pretrain_1400',
                      vwc_dir='/data/ssd2/swapstress/vwc/hhp/rosetta',
                      gridmet_dir='/data/ssd2/swapstress/vwc/gridmet/rosetta',
                      checkpoint=None, window_len=180, patch_len=30, n_patches=3,
                      figsize=(14, 8)):
    vdf_a, gdf_a = _load_series(site_a, vwc_dir, gridmet_dir)
    vdf_b, gdf_b = _load_series(site_b, vwc_dir, gridmet_dir)

    idx_a, xv_a, xf_a = _select_window(vdf_a, gdf_a, window_len)
    idx_b, xv_b, xf_b = _select_window(vdf_b, gdf_b, window_len)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=4, ncols=2, width_ratios=[3.0, 1.2], height_ratios=[1, 1, 1, 1], wspace=0.25,
                          hspace=0.25)

    ax_a_met = fig.add_subplot(gs[0, 0])
    ax_a_vwc = fig.add_subplot(gs[1, 0], sharex=ax_a_met)
    ax_b_met = fig.add_subplot(gs[2, 0])
    ax_b_vwc = fig.add_subplot(gs[3, 0], sharex=ax_b_met)
    ax_a_emb = fig.add_subplot(gs[0:2, 1])
    ax_b_emb = fig.add_subplot(gs[2:4, 1])

    _plot_met(ax_a_met, gdf_a.loc[idx_a], label_prefix='A: ')
    _plot_vwc_with_masks(ax_a_vwc, vdf_a.loc[idx_a], site_a, patch_len, n_patches)
    ax_a_vwc.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_a_met.set_title(f"Example A: {site_a}")

    _plot_met(ax_b_met, gdf_b.loc[idx_b], label_prefix='B: ')
    _plot_vwc_with_masks(ax_b_vwc, vdf_b.loc[idx_b], site_b, patch_len, n_patches)
    ax_b_vwc.tick_params(axis='x', which='both', labelbottom=False, bottom=False)
    ax_b_met.set_title(f"Example B: {site_b}")

    if checkpoint is not None:
        model = load_mae(checkpoint)
        model.eval()
        img_a = _embed_to_image(model, xv_a, xf_a)
        vmin = float(np.min(img_a))
        vmax = float(np.max(img_a))
        im = ax_a_emb.imshow(img_a, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
        ax_a_emb.set_title('A embedding (8x8)')
        ax_a_emb.axis('off')
        fig.colorbar(im, ax=ax_a_emb, fraction=0.046, pad=0.04)

        img_b = _embed_to_image(model, xv_b, xf_b)
        vmin = float(np.min(img_b))
        vmax = float(np.max(img_b))
        im = ax_b_emb.imshow(img_b, cmap='viridis', vmin=vmin, vmax=vmax, origin='upper')
        ax_b_emb.set_title('B embedding (8x8)')
        ax_b_emb.axis('off')
        fig.colorbar(im, ax=ax_b_emb, fraction=0.046, pad=0.04)
    else:
        ax_a_emb.text(0.5, 0.5, 'Provide checkpoint', ha='center', va='center')
        ax_b_emb.text(0.5, 0.5, 'Provide checkpoint', ha='center', va='center')
        ax_a_emb.axis('off')
        ax_b_emb.axis('off')

    return fig


if __name__ == '__main__':

    home_ = os.path.expanduser('~')
    root_ = os.path.join(home_, 'data', 'IrrigationGIS')

    vwc_dir = '/data/ssd2/swapstress/vwc/hhp/rosetta'
    gridmet_dir = '/data/ssd2/swapstress/vwc/gridmet/rosetta'

    ckpt_root_ = os.path.join(vwc_dir, 'checkpoints')
    checkpoint = os.path.join(ckpt_root_, 'both_20250917_170218/mae-both-20250917-epoch=99-val_loss=0.1314.ckpt')

    fig_file = os.path.join(root_, 'soils/swapstress/figures/mae/example.png')
    fig_ = plot_mae_examples(checkpoint=checkpoint)
    plt.savefig(fig_file)
    # plt.show()

# ========================= EOF ====================================================================
