import os
import glob
import argparse
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import torch
from PIL import Image
from pathlib import Path


def load_virchow2_model_and_preprocess(device: torch.device):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    # model card requires these args for proper init
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    ).to(device).eval()

    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, preprocess

@torch.no_grad()
def encode_images(model, x: torch.Tensor) -> torch.Tensor:
    """
    Returns BxD image embeddings.
    Tries model.encode_image first (CLIP-style), falls back to model.visual.
    """
    if hasattr(model, "encode_image"):
        z = model.encode_image(x)
    elif hasattr(model, "visual"):
        z = model.visual(x)
    else:
        z = model(x)

    # Some CLIP variants return (features, ...) or dict
    if isinstance(z, dict):
        for k in ["image_embeds", "embeds", "features", "proj"]:
            if k in z:
                z = z[k]
                break
    if isinstance(z, (tuple, list)):
        z = z[0]

    # Ensure float tensor
    return z.float()

@torch.no_grad()
def encode_images_virchow2(model, x: torch.Tensor, use_concat_2560: bool = True) -> torch.Tensor:
    """
    Virchow2 returns tokens: B x 261 x 1280
    token 0 = class token
    tokens 1-4 = register tokens
    tokens 5: = 256 patch tokens
    """
    out = model(x)  # B x 261 x 1280

    class_tok = out[:, 0]      # B x 1280
    patch_toks = out[:, 5:]    # B x 256 x 1280

    if use_concat_2560:
        emb = torch.cat([class_tok, patch_toks.mean(1)], dim=-1)  # B x 2560
    else:
        emb = class_tok  # B x 1280

    return emb.float()

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
        # allow some alternates
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

    # Convert to uint8 RGB for PIL
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        # If it looks like 0..1, scale; else assume 0..255
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
            t = preprocess(pil)  # 3xHxW, already normalized
            imgs.append(t)

        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)  # Bx3xHxW
        # z = encode_images(model, x)  # BxD

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            z = encode_images_virchow2(model, x, use_concat_2560=True)

        feats_chunks.append(z.detach().cpu().numpy().astype(np.float32))

    feats = np.concatenate(feats_chunks, axis=0).astype(np.float32)

    out_h5 = os.path.join(out_h5_dir, f"{slide_id}.h5")
    out_pt = os.path.join(out_pt_dir, f"{slide_id}.pt")
    save_h5(out_h5, feats, coords)
    save_pt(out_pt, feats, coords)

    return {"slide_id": slide_id, "status": "ok", "n": int(feats.shape[0]), "dim": int(feats.shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides_root", type=str, required=True,
                    help="Root dir with per-slide folders containing .npy patches + patch_map.csv")
    ap.add_argument("--patch_map_name", type=str, default="patch_map.csv")
    ap.add_argument("--out_h5_dir", type=str, required=True)
    ap.add_argument("--out_pt_dir", type=str, required=True)

    # ap.add_argument("--project_root", type=str, required=True,
    #                 help="Your CONCH repo root, e.g. C:\\Users\\Vivian\\Documents\\CONCH")
    # ap.add_argument("--checkpoint_path", type=str, required=True,
    #                 help="Path to CONCH checkpoint .bin")
    # ap.add_argument("--model_cfg", type=str, default="conch_ViT-B-16")

    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_patches", type=int, default=0,
                    help="0 = all patches, else cap per slide")
    args = ap.parse_args()

    # Device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # print("Loading CONCH model...")
    # model, preprocess = load_conch_model_and_preprocess(
    #     project_root=args.project_root,
    #     checkpoint_path=args.checkpoint_path,
    #     model_cfg=args.model_cfg,
    #     device=device,
    # )

    print("Loading Virchow2 model...")
    model, preprocess = load_virchow2_model_and_preprocess(device=device)

    # slide_dirs = sorted([p for p in glob.glob(os.path.join(args.slides_root, "*")) if os.path.isdir(p)])
    slide_dirs = sorted([
    p for p in glob.glob(os.path.join(args.slides_root, "**"), recursive=True)
    if os.path.isdir(p) and os.path.exists(os.path.join(p, args.patch_map_name))
])

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
    out_csv = os.path.join(os.path.dirname(args.out_h5_dir), "conch_image_feature_extraction_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")
    print(df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
