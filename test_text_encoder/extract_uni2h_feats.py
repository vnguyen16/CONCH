import os
import glob
import argparse
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import torch
from PIL import Image


# ============================================================
# UNI2-h loader + preprocess (timm)
# ============================================================
def load_uni2h_model_and_preprocess(
    device: torch.device,
    ckpt_dir: str | None = None,
    use_hf_pretrained: bool = True,
):
    """
    Loads UNI2-h (MahmoodLab/UNI2-h) via timm and returns (model, preprocess).

    If ckpt_dir is provided and contains pytorch_model.bin, we load that state_dict.
    Otherwise, if use_hf_pretrained=True, timm will fetch pretrained weights from HF hub.
    """
    import timm
    from timm.layers import SwiGLUPacked
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    # Create model
    if use_hf_pretrained:
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    else:
        model = timm.create_model(**timm_kwargs)

    # Optional: override weights from a local checkpoint dir
    if ckpt_dir is not None:
        ckpt_path = os.path.join(ckpt_dir, "pytorch_model.bin")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"UNI2 checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=True)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, preprocess


@torch.inference_mode()
def encode_images_uni2(model, x: torch.Tensor) -> torch.Tensor:
    """
    UNI2-h forward returns [B,1536] when num_classes=0.
    """
    use_amp = x.device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        z = model(x)
    if isinstance(z, (tuple, list)):
        z = z[0]
    if isinstance(z, dict):
        # unlikely for UNI2-h, but keep robust
        for k in ["image_embeds", "embeds", "features", "proj"]:
            if k in z:
                z = z[k]
                break
    return z.detach().float()


# ============================================================
# I/O helpers
# ============================================================
def read_patch_map(patch_map_csv: str) -> pd.DataFrame:
    """
    patch_map.csv expected columns: patch_file, x, y
    """
    df = pd.read_csv(patch_map_csv)
    df.columns = [c.strip().lower() for c in df.columns]

    if "patch_file" not in df.columns:
        df = df.rename(columns={df.columns[0]: "patch_file"})
    if "x" not in df.columns or "y" not in df.columns:
        for candx in ["x_px", "xcoord", "x_coord"]:
            if candx in df.columns:
                df = df.rename(columns={candx: "x"})
        for candy in ["y_px", "ycoord", "y_coord"]:
            if candy in df.columns:
                df = df.rename(columns={candy: "y"})
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"patch_map.csv missing x/y columns: {patch_map_csv}")

    df["patch_file"] = df["patch_file"].astype(str)
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["x", "y"])
    return df


def npy_to_pil_rgb(npy_path: str) -> Image.Image:
    """
    Loads a patch stored as npy and returns an RGB PIL Image.
    Accepts:
      - HxWx3 (uint8/float)
      - 3xHxW (uint8/float)
    """
    arr = np.load(npy_path)

    if arr.ndim != 3:
        raise ValueError(f"{npy_path}: expected 3D array, got shape {arr.shape}")

    # CHW -> HWC
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = arr[:3, ...]
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
    else:
        raise ValueError(f"{npy_path}: cannot infer channels from shape {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


def save_h5(out_h5: str, feats: np.ndarray, coords: np.ndarray):
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("features", data=feats, compression="gzip")
        f.create_dataset("coords", data=coords, compression="gzip")


def save_pt(out_pt: str, feats: np.ndarray, coords: np.ndarray):
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    obj = {
        "features": torch.from_numpy(feats),
        "coords": torch.from_numpy(coords),
    }
    torch.save(obj, out_pt)


# ============================================================
# Extraction
# ============================================================
def extract_one_slide(
    slide_dir: str,
    patch_map_name: str,
    model,
    preprocess,
    device: torch.device,
    out_h5_dir: str,
    out_pt_dir: str,
    batch_size: int,
    max_patches: int,
):
    slide_id = os.path.basename(slide_dir.rstrip("\\/"))

    patch_map_csv = os.path.join(slide_dir, patch_map_name)
    if not os.path.exists(patch_map_csv):
        return {"slide_id": slide_id, "status": "missing_patch_map", "n": 0}

    dfm = read_patch_map(patch_map_csv)

    patch_paths = []
    coords = []
    for _, r in dfm.iterrows():
        p = os.path.join(slide_dir, r["patch_file"])
        if os.path.exists(p):
            patch_paths.append(p)
            coords.append([int(r["x"]), int(r["y"])])

    if len(patch_paths) == 0:
        return {"slide_id": slide_id, "status": "no_patches_found", "n": 0}

    if max_patches > 0 and len(patch_paths) > max_patches:
        patch_paths = patch_paths[:max_patches]
        coords = coords[:max_patches]

    coords = np.asarray(coords, dtype=np.int32)

    feats_chunks = []
    model.eval()

    for i in range(0, len(patch_paths), batch_size):
        batch_paths = patch_paths[i : i + batch_size]

        imgs = []
        for p in batch_paths:
            pil = npy_to_pil_rgb(p)
            t = preprocess(pil)  # 3xHxW, normalized by timm transform
            imgs.append(t)

        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)  # Bx3xHxW
        z = encode_images_uni2(model, x)  # Bx1536

        feats_chunks.append(z.cpu().numpy().astype(np.float32))

    feats = np.concatenate(feats_chunks, axis=0).astype(np.float32)

    out_h5 = os.path.join(out_h5_dir, f"{slide_id}.h5")
    out_pt = os.path.join(out_pt_dir, f"{slide_id}.pt")
    save_h5(out_h5, feats, coords)
    save_pt(out_pt, feats, coords)

    return {"slide_id": slide_id, "status": "ok", "n": int(feats.shape[0]), "dim": int(feats.shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--slides_root",
        type=str,
        required=True,
        help="Root dir with per-slide folders containing .npy patches + patch_map.csv",
    )
    ap.add_argument("--patch_map_name", type=str, default="patch_map.csv")
    ap.add_argument("--out_h5_dir", type=str, required=True)
    ap.add_argument("--out_pt_dir", type=str, required=True)

    # UNI2 options
    ap.add_argument(
        "--uni2_ckpt_dir",
        type=str,
        default="",
        help="Optional local dir containing pytorch_model.bin for UNI2-h (leave empty to use HF pretrained weights).",
    )
    ap.add_argument(
        "--no_hf_pretrained",
        action="store_true",
        help="If set, do not load HF pretrained weights (only meaningful if you also pass --uni2_ckpt_dir).",
    )

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_patches", type=int, default=0, help="0 = all patches, else cap per slide")
    args = ap.parse_args()

    # Device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("Loading UNI2-h model...")
    ckpt_dir = args.uni2_ckpt_dir.strip() or None
    model, preprocess = load_uni2h_model_and_preprocess(
        device=device,
        ckpt_dir=ckpt_dir,
        use_hf_pretrained=(not args.no_hf_pretrained),
    )

    slide_dirs = sorted(
        [
            p
            for p in glob.glob(os.path.join(args.slides_root, "**"), recursive=True)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, args.patch_map_name))
        ]
    )

    print(f"Found {len(slide_dirs)} slide dirs under {args.slides_root}")

    rows = []
    for sd in tqdm(slide_dirs, desc="Extract slides"):
        try:
            rows.append(
                extract_one_slide(
                    slide_dir=sd,
                    patch_map_name=args.patch_map_name,
                    model=model,
                    preprocess=preprocess,
                    device=device,
                    out_h5_dir=args.out_h5_dir,
                    out_pt_dir=args.out_pt_dir,
                    batch_size=args.batch_size,
                    max_patches=args.max_patches,
                )
            )
        except Exception as e:
            rows.append({"slide_id": os.path.basename(sd), "status": f"error: {e}", "n": 0})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_h5_dir), exist_ok=True)
    out_csv = os.path.join(os.path.dirname(args.out_h5_dir), "uni2h_image_feature_extraction_summary.csv")
    df.to_csv(out_csv, index=False)

    print(f"Saved summary: {out_csv}")
    print(df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()