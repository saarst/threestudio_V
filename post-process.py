# eval_metrics_simple.py
import argparse, os, glob, csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import torch, lpips  # pip install lpips

# ---------- image & metrics ----------

def to_rgb_numpy(path):
    return np.array(Image.open(path).convert("RGB"))

def psnr(img, ref):
    img = img.astype(np.float32); ref = ref.astype(np.float32)
    mse = np.mean((img - ref) ** 2)
    if mse == 0:
        return 99.0
    return 20.0*np.log10(255.0) - 10.0*np.log10(mse)

def ssim_rgb(img, ref):
    return ssim(ref, img, channel_axis=2, data_range=255)

def lpips_dist(net, img, ref):
    ten  = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()/127.5 - 1.0
    tref = torch.from_numpy(ref).permute(2,0,1).unsqueeze(0).float()/127.5 - 1.0
    with torch.no_grad():
        return float(net(ten, tref).item())

# ---------- helpers ----------

def read_total_time(exp_dir):
    """
    Reads exp_dir/total_time.txt
    Expected like: '113.91 seconds'
    Returns float seconds or None.
    """
    path = os.path.join(exp_dir, "total_time.txt")
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                line = f.readline().strip()
                val = float(line.split()[0])
                return val
        except Exception:
            return None
    return None

def parse_meta_from_name(name: str):
    """
    Handles:
      SDI_SANA_NeRF_{CG|Adam}_..._n_opt_{N}[_lr_{...}]
      sample_SANA_NeRF_{ours_CG|ours_Adam|Adam}_..._n_opt_{N}[_lr_{...}]
    No regex; uses simple substring and split logic.
    """
    # source (script family)
    if name.startswith("SDI_SANA_NeRF"):
        source = "SDI"
    elif name.startswith("sample_SANA_NeRF"):
        source = "sample"
    else:
        source = "unknown"

    # optimizer label
    if "_ours_CG_" in name:
        optimizer = "ours-CG"
    elif "_ours_Adam_" in name:
        optimizer = "ours-Adam"
    elif "_CG_" in name:
        optimizer = "CG"
    elif "_Adam_" in name:
        # make sure we didn't already match ours_Adam
        optimizer = "Adam" if "_ours_Adam_" not in name else "ours-Adam"
    else:
        optimizer = "unknown"

    # n_opt
    n_opt = None
    if "_n_opt_" in name:
        try:
            n_opt = int(name.split("_n_opt_")[1].split("_")[0])
        except Exception:
            n_opt = None

    # lr (only if present in name)
    lr = ""
    if "_lr_" in name:
        lr = name.split("_lr_")[1].split("_")[0]

    # gentle nudge if Adam-like but no lr encoded
    if ("Adam" in optimizer) and lr == "":
        print(f"[NOTE] {name}: Adam variant without `_lr_...` in exp_name → label will omit LR.")

    return {
        "exp_name": name,
        "source": source,
        "optimizer": optimizer,
        "n_opt": n_opt,
        "lr": lr,
    }

def build_pairs(gen_dir, gt_dir, limit, ext):
    gen = sorted(glob.glob(os.path.join(gen_dir, f"*.{ext}")))
    gt  = sorted(glob.glob(os.path.join(gt_dir,  f"*.{ext}")))
    if not gen or not gt:
        return []
    # filename-based map first
    gt_map = {os.path.basename(p): p for p in gt}
    pairs = [(g, gt_map[os.path.basename(g)]) for g in gen[:limit] if os.path.basename(g) in gt_map]
    if not pairs:
        n = min(limit, min(len(gen), len(gt)))
        pairs = list(zip(gen[:n], gt[:n]))
    return pairs

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def scatter(rows, metric, ylabel, outpng):
    """
    Metric vs. time with visual encoding:
      • Color hue = learning rate (LR). Non-Adam -> 'no LR' (gray).
      • Shade/lightness = n_opt (lighter=fewer iters, darker=more iters).
      • Marker shape = method:
          - SDI ->  $\hat{x}_0$-Adam / $\hat{x}_0$-CG
          - sample (no ours) -> x_t-Adam
          - sample (ours) ->  v (Ours)-Adam / v (Ours)-CG
    """
    data = [r for r in rows if r.get("total_time_sec") is not None and r.get("n_opt") is not None]
    if not data:
        print(f"No data for {metric}")
        return

    # --- method label & markers ---
    def method_label(r):
        src, opt = r["source"], r["optimizer"]
        if src == "SDI":
            return r'$\hat{x}_0$-Adam' if "Adam" in opt else r'$\hat{x}_0$-CG'
        else:
            if opt == "ours-Adam": return "v (Ours)-Adam"
            if opt == "ours-CG"  : return "v (Ours)-CG"
            if opt == "Adam"     : return "x_t-Adam"
            if opt == "CG"       : return "x_t-CG"
            return "unknown"

    marker_for_method = {
        r'$\hat{x}_0$-Adam': 'o',
        r'$\hat{x}_0$-CG'  : 's',
        'x_t-Adam'         : '^',
        'x_t-CG'           : 'P',
        'v (Ours)-Adam'    : 'D',
        'v (Ours)-CG'      : 'X',
        'unknown'          : 'x',
    }

    # --- base colors per LR (hue only) ---
    lr_values = sorted({
        (r["lr"] if ("Adam" in r["optimizer"] and r["lr"]) else "N/A") for r in data
    }, key=lambda x: (x=="N/A", x))
    cmap = plt.colormaps.get_cmap("tab10")
    base_color = {}
    non_na = [v for v in lr_values if v != "N/A"]
    for i, lr in enumerate(non_na):
        base_color[lr] = cmap(i % cmap.N)
    base_color["N/A"] = (0.5, 0.5, 0.5, 1.0)  # gray for no-LR

    # --- shade by n_opt (lighter for smaller, darker for larger) ---
    levels = sorted({r["n_opt"] for r in data})
    # mix factor t in [0.45 .. 0.0]; t=0.45 -> much lighter, t=0.0 -> original
    if len(levels) > 1:
        ts = [0.45 * (1 - i/(len(levels)-1)) for i in range(len(levels))]
    else:
        ts = [0.0]
    shade_for_level = dict(zip(levels, ts))

    def lighten(c, t):
        # blend toward white by factor t \in [0, 0.45]
        r,g,b,a = c
        return (r + (1-r)*t, g + (1-g)*t, b + (1-b)*t, a)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(14, 9))
    for r in data:
        m = method_label(r)
        lr_tag = r["lr"] if ("Adam" in r["optimizer"] and r["lr"]) else "N/A"
        base = base_color[lr_tag]
        shade = lighten(base, shade_for_level[r["n_opt"]])
        ax.scatter(
            r["total_time_sec"], r[metric],
            s=120, color=shade,
            marker=marker_for_method.get(m, 'x'),
            edgecolor="black", linewidths=0.5
        )

    ax.set_xlabel("Total time (s)", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(f"{ylabel} vs. total time", fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=14)

    # --- legends under the plot ---
    # 1) LR legend (hues only)
    lr_handles = [Line2D([0],[0], marker='o', color='w',
                         markerfacecolor=base_color[lr], markeredgecolor='black',
                         markersize=10, linewidth=0, label=(f"lr={lr}" if lr!="N/A" else "no LR"))
                  for lr in lr_values]
    lr_legend = fig.legend(
        handles=lr_handles, title="Learning rate (hue)",
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=max(3, len(lr_handles)), frameon=True, fontsize=12
    )

    # 2) n_opt legend (shade only; generic swatches)
    shade_handles = [Line2D([0],[0], marker='s', color='w',
                            markerfacecolor=lighten((0.2,0.2,0.2,1.0), shade_for_level[lvl]),
                            markeredgecolor='black', markersize=10, linewidth=0, label=f"n_opt={lvl}")
                     for lvl in levels]
    fig.legend(
        handles=shade_handles, title="Iterations (shade)",
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=max(3, len(shade_handles)), frameon=True, fontsize=12
    )

    # 3) Method legend (markers only)
    methods_present = sorted({method_label(r) for r in data})
    method_handles = [Line2D([0],[0], marker=marker_for_method[m], color='w',
                             markerfacecolor='white', markeredgecolor='black',
                             markersize=10, linewidth=0, label=m)
                      for m in methods_present]
    fig.legend(
        handles=method_handles, title="Method (marker)",
        loc="upper center", bbox_to_anchor=(0.5, -0.28),
        ncol=max(3, len(method_handles)), frameon=True, fontsize=12
    )

    # Save with ample room for 3 legends
    fig.savefig(outpng, dpi=200, bbox_inches="tight")
    plt.close(fig)

    
import pandas as pd  # if not already imported

def load_mean_mse_traj(run_dir):
    """
    Expects run_dir/mse_results.csv with columns:
      seed, step_1, step_2, ..., step_T
    Returns (steps_idx:list[int], mean_mse:np.ndarray) or (None, None) if missing.
    """
    csv_path = os.path.join(run_dir, "mse_results.csv")
    if not os.path.isfile(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    step_cols = [c for c in df.columns if c.startswith("step_")]
    if not step_cols:
        return None, None

    step_cols = sorted(step_cols, key=lambda s: int(s.split("_")[1]))
    mean_mse = df[step_cols].mean(axis=0, skipna=True).to_numpy()
    steps_idx = [int(s.split("_")[1]) for s in step_cols]
    return steps_idx, mean_mse

def plot_mse_traj_all(run_trajs, outdir, fname_png="mse_traj_all.png",
                      title="MSE trajectory (mean over seeds) — all runs"):
    if not run_trajs:
        print("[INFO] No MSE trajectories found to plot.")
        return

    import math
    os.makedirs(outdir, exist_ok=True)

    # Bigger canvas to fit a bottom legend
    plt.figure(figsize=(14, 10))

    cmap = plt.colormaps.get_cmap("tab20")
    for i, rt in enumerate(run_trajs):
        plt.plot(rt["steps"], rt["mean_mse"],
                 marker="o", linewidth=2,
                 label=rt["name"], color=cmap(i % cmap.N))

    plt.yscale("log")
    plt.xlabel("Diffusion step")
    plt.ylabel("Mean MSE (log scale)")
    plt.title(title)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="SANA_samples", help="GT images folder")
    ap.add_argument("--runs-root", default="exp_2", help="Folder with all run dirs")
    ap.add_argument("--exp-glob", default="*SANA_NeRF*", help="Pattern for run dirs inside runs-root")
    ap.add_argument("--limit", type=int, default=10, help="Max images per run")
    ap.add_argument("--img-ext", default="png", help="Image file extension")
    ap.add_argument("--out-csv", default="eval_summary.csv", help="CSV output file")
    args = ap.parse_args()

    outdir = os.path.join("figs", os.path.basename(args.runs_root.rstrip("/")))
    os.makedirs(outdir, exist_ok=True)

    net = lpips.LPIPS(net="alex").eval()

    run_dirs = sorted([p for p in glob.glob(os.path.join(args.runs_root, args.exp_glob)) if os.path.isdir(p)])
    if not run_dirs:
        print("No runs found in", args.runs_root)
        return

    rows = []
    run_trajs = []  # before the loop

    for d in run_dirs:
        name = os.path.basename(d.rstrip("/"))
        meta = parse_meta_from_name(name)
        ttime = read_total_time(d)

        # collect MSE trajectory if exists
        steps_idx, mean_mse = load_mean_mse_traj(d)
        if steps_idx is not None:
            run_trajs.append({"name": name, "steps": steps_idx, "mean_mse": mean_mse})
        else:
            print(f"[INFO] No mse_results.csv in {d}")

        pairs = build_pairs(d, args.gt_dir, args.limit, args.img_ext)
        if not pairs:
            print(f"[WARN] no image pairs for {d}")
            continue

        ps, ss, ls = [], [], []
        for gp, rp in pairs:
            g = to_rgb_numpy(gp); r = to_rgb_numpy(rp)
            if g.shape != r.shape:
                H, W = r.shape[:2]
                g = np.array(Image.fromarray(g).resize((W, H), Image.BICUBIC))
            ps.append(psnr(g, r))
            ss.append(ssim_rgb(g, r))
            ls.append(lpips_dist(net, g, r))

        rows.append({
            "exp_name": meta["exp_name"],
            "source": meta["source"],
            "optimizer": meta["optimizer"],
            "n_opt": meta["n_opt"],
            "lr": meta["lr"],
            "PSNR": float(np.mean(ps)),
            "SSIM": float(np.mean(ss)),
            "LPIPS": float(np.mean(ls)),
            "total_time_sec": ttime,
            "count": len(ps),
        })

    plot_mse_traj_all(run_trajs, outdir, fname_png="mse_traj_all.png",
                    title="MSE trajectory (mean over seeds) — all runs")

    # save CSV
    out_csv_path = os.path.join(outdir, args.out_csv)
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out_csv_path} ({len(rows)} rows)")

    scatter(rows, "PSNR",  "PSNR (dB)",               os.path.join(outdir, "psnr_vs_time.png"))
    scatter(rows, "SSIM",  "SSIM",                    os.path.join(outdir, "ssim_vs_time.png"))
    scatter(rows, "LPIPS", "LPIPS (lower is better)", os.path.join(outdir, "lpips_vs_time.png"))


if __name__ == "__main__":
    main()
