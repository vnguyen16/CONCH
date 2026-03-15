# import os
# import glob
# import argparse
# import numpy as np
# import pandas as pd
# import h5py
# from tqdm import tqdm

# import torch
# from PIL import Image
# from pathlib import Path

# from conch.open_clip_custom.factory import create_model_from_pretrained


# # ============================================================
# # CONCH v1.5 loader (local checkpoint OR HF pretrained)
# # ============================================================
# def load_conch_model_and_preprocess(project_root, checkpoint_path, model_cfg, device):
#     project_root = Path(project_root)
#     if str(project_root) not in os.sys.path:
#         os.sys.path.insert(0, str(project_root))

#     from conch.open_clip_custom import create_model_from_pretrained

#     if not os.path.isfile(checkpoint_path):
#         raise FileNotFoundError(checkpoint_path)

#     model, preprocess = create_model_from_pretrained(
#         model_cfg=model_cfg,
#         checkpoint_path=checkpoint_path,
#         device=device,
#     )

#     model.eval()
#     return model, preprocess

# @torch.no_grad()
# def encode_images(model, x: torch.Tensor) -> torch.Tensor:
#     """
#     Returns BxD image embeddings.
#     For CONCH: encode_image(x, proj_contrast=..., normalize=...)
#     """
#     if hasattr(model, "encode_image"):
#         z = model.encode_image(x, proj_contrast=False, normalize=False)  # unnorm feats
#     elif hasattr(model, "visual"):
#         z = model.visual(x)
#     else:
#         z = model(x)

#     if isinstance(z, dict):
#         for k in ["image_embeds", "embeds", "features", "proj"]:
#             if k in z:
#                 z = z[k]
#                 break
#     if isinstance(z, (tuple, list)):
#         z = z[0]
#     return z.float()


# # ============================================================
# # I/O helpers
# # ============================================================
# def read_patch_map(patch_map_csv: str) -> pd.DataFrame:
#     df = pd.read_csv(patch_map_csv)
#     df.columns = [c.strip().lower() for c in df.columns]

#     if "patch_file" not in df.columns:
#         df = df.rename(columns={df.columns[0]: "patch_file"})
#     if "x" not in df.columns or "y" not in df.columns:
#         for candx in ["x_px", "xcoord", "x_coord"]:
#             if candx in df.columns:
#                 df = df.rename(columns={candx: "x"})
#         for candy in ["y_px", "ycoord", "y_coord"]:
#             if candy in df.columns:
#                 df = df.rename(columns={candy: "y"})
#     if "x" not in df.columns or "y" not in df.columns:
#         raise ValueError(f"patch_map.csv missing x/y columns: {patch_map_csv}")

#     df["patch_file"] = df["patch_file"].astype(str)
#     df["x"] = pd.to_numeric(df["x"], errors="coerce")
#     df["y"] = pd.to_numeric(df["y"], errors="coerce")
#     df = df.dropna(subset=["x", "y"])
#     return df


# def npy_to_pil_rgb(npy_path: str) -> Image.Image:
#     arr = np.load(npy_path)

#     if arr.ndim != 3:
#         raise ValueError(f"{npy_path}: expected 3D array, got shape {arr.shape}")

#     if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
#         arr = arr[:3, ...]
#         arr = np.transpose(arr, (1, 2, 0))
#     elif arr.shape[-1] in (3, 4):
#         arr = arr[..., :3]
#     else:
#         raise ValueError(f"{npy_path}: cannot infer channels from shape {arr.shape}")

#     if arr.dtype != np.uint8:
#         arr = arr.astype(np.float32)
#         if arr.max() <= 1.5:
#             arr = arr * 255.0
#         arr = np.clip(arr, 0, 255).astype(np.uint8)

#     return Image.fromarray(arr, mode="RGB")


# def save_h5(out_h5: str, feats: np.ndarray, coords: np.ndarray):
#     os.makedirs(os.path.dirname(out_h5), exist_ok=True)
#     with h5py.File(out_h5, "w") as f:
#         f.create_dataset("features", data=feats, compression="gzip")
#         f.create_dataset("coords", data=coords, compression="gzip")


# def save_pt(out_pt: str, feats: np.ndarray, coords: np.ndarray):
#     os.makedirs(os.path.dirname(out_pt), exist_ok=True)
#     torch.save(
#         {"features": torch.from_numpy(feats), "coords": torch.from_numpy(coords)},
#         out_pt,
#     )


# # ============================================================
# # Extraction
# # ============================================================
# def extract_one_slide(
#     slide_dir: str,
#     patch_map_name: str,
#     model,
#     preprocess,
#     device: torch.device,
#     out_h5_dir: str,
#     out_pt_dir: str,
#     batch_size: int,
#     max_patches: int,
# ):
#     slide_id = os.path.basename(slide_dir.rstrip("\\/"))

#     patch_map_csv = os.path.join(slide_dir, patch_map_name)
#     if not os.path.exists(patch_map_csv):
#         return {"slide_id": slide_id, "status": "missing_patch_map", "n": 0}

#     dfm = read_patch_map(patch_map_csv)

#     patch_paths, coords = [], []
#     for _, r in dfm.iterrows():
#         p = os.path.join(slide_dir, r["patch_file"])
#         if os.path.exists(p):
#             patch_paths.append(p)
#             coords.append([int(r["x"]), int(r["y"])])

#     if len(patch_paths) == 0:
#         return {"slide_id": slide_id, "status": "no_patches_found", "n": 0}

#     if max_patches > 0 and len(patch_paths) > max_patches:
#         patch_paths = patch_paths[:max_patches]
#         coords = coords[:max_patches]

#     coords = np.asarray(coords, dtype=np.int32)

#     feats_chunks = []
#     model.eval()

#     for i in range(0, len(patch_paths), batch_size):
#         batch_paths = patch_paths[i : i + batch_size]

#         imgs = []
#         for p in batch_paths:
#             pil = npy_to_pil_rgb(p)
#             imgs.append(preprocess(pil))  # normalized tensor

#         x = torch.stack(imgs, dim=0).to(device, non_blocking=True)

#         use_amp = device.type == "cuda"
#         with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
#             z = encode_images(model, x)

#         feats_chunks.append(z.detach().cpu().numpy().astype(np.float32))

#     feats = np.concatenate(feats_chunks, axis=0).astype(np.float32)

#     out_h5 = os.path.join(out_h5_dir, f"{slide_id}.h5")
#     out_pt = os.path.join(out_pt_dir, f"{slide_id}.pt")
#     save_h5(out_h5, feats, coords)
#     save_pt(out_pt, feats, coords)

#     return {"slide_id": slide_id, "status": "ok", "n": int(feats.shape[0]), "dim": int(feats.shape[1])}


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--slides_root", type=str, required=True)
#     ap.add_argument("--patch_map_name", type=str, default="patch_map.csv")
#     ap.add_argument("--out_h5_dir", type=str, required=True)
#     ap.add_argument("--out_pt_dir", type=str, required=True)

#     ap.add_argument("--project_root", type=str, required=True)
#     ap.add_argument("--model_cfg", type=str, default="conch_ViT-B-16")

#     # v1.5 options
#     ap.add_argument("--checkpoint_path", type=str, default="",
#                 help="Local CONCH checkpoint path. Leave empty to use HF.")
#     ap.add_argument("--hf_ckpt", type=str, default="hf-hub:MahmoodLab/conch",
#                 help='HF checkpoint identifier (passed as checkpoint_path to conch loader).')

#     # ap.add_argument(
#     #     "--checkpoint_path",
#     #     type=str,
#     #     default="",
#     #     help="Path to local CONCH v1.5 checkpoint (.bin). Leave empty to use HF fallback.",
#     # )
#     ap.add_argument(
#         "--hf_repo",
#         type=str,
#         default="hf-hub:MahmoodLab/conch",
#         help='HF repo id, e.g. "hf-hub:MahmoodLab/conch" (adjust if your repo uses a different name).',
#     )
#     ap.add_argument(
#         "--hf_revision",
#         type=str,
#         default="",
#         help="Optional HF revision/tag/commit for v1.5 if needed.",
#     )
#     ap.add_argument(
#         "--use_hf",
#         action="store_true",
#         help="Force HF even if checkpoint_path exists.",
#     )

#     ap.add_argument("--device", type=str, default="cuda:0")
#     ap.add_argument("--batch_size", type=int, default=64)
#     ap.add_argument("--max_patches", type=int, default=0)
#     args = ap.parse_args()

#     if args.device.startswith("cuda") and not torch.cuda.is_available():
#         print("CUDA not available; falling back to CPU")
#         device = torch.device("cpu")
#     else:
#         device = torch.device(args.device)

#     # ckpt = None
#     # if (not args.use_hf) and args.checkpoint_path.strip():
#     #     ckpt = args.checkpoint_path.strip()

#     # hf_revision = args.hf_revision.strip() or None

#     # print("Loading CONCH v1.5 (local ckpt if provided, else HF)...")
#     # model, preprocess = load_conch15_model_and_preprocess(
#     #     project_root=args.project_root,
#     #     device=device,
#     #     model_cfg=args.model_cfg,
#     #     checkpoint_path=ckpt,
#     #     hf_repo=args.hf_repo,
#     #     hf_revision=hf_revision,
#     # )

#     ckpt = args.checkpoint_path.strip() or None

#     print("Loading CONCH v1.5...")
#     model, preprocess = load_conch_model_and_preprocess(
#         project_root=args.project_root,
#         device=device,
#         model_cfg=args.model_cfg,     # keep this as config name
#         checkpoint_path=ckpt,
#         # hf_ckpt=args.hf_ckpt,
#     )

#     slide_dirs = sorted(
#         [
#             p
#             for p in glob.glob(os.path.join(args.slides_root, "**"), recursive=True)
#             if os.path.isdir(p) and os.path.exists(os.path.join(p, args.patch_map_name))
#         ]
#     )
#     print(f"Found {len(slide_dirs)} slide dirs under {args.slides_root}")

#     rows = []
#     for sd in tqdm(slide_dirs, desc="Extract slides"):
#         try:
#             rows.append(
#                 extract_one_slide(
#                     slide_dir=sd,
#                     patch_map_name=args.patch_map_name,
#                     model=model,
#                     preprocess=preprocess,
#                     device=device,
#                     out_h5_dir=args.out_h5_dir,
#                     out_pt_dir=args.out_pt_dir,
#                     batch_size=args.batch_size,
#                     max_patches=args.max_patches,
#                 )
#             )
#         except Exception as e:
#             rows.append({"slide_id": os.path.basename(sd), "status": f"error: {e}", "n": 0})

#     df = pd.DataFrame(rows)
#     os.makedirs(os.path.dirname(args.out_h5_dir), exist_ok=True)
#     out_csv = os.path.join(os.path.dirname(args.out_h5_dir), "conch15_image_feature_extraction_summary.csv")
#     df.to_csv(out_csv, index=False)
#     print(f"Saved summary: {out_csv}")
#     print(df["status"].value_counts(dropna=False))


# if __name__ == "__main__":
#     main()

# ============================================ NEW 
import os
import glob
import argparse
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoModel


# ============================================================
# CONCH v1.5 via TITAN
# ============================================================
def load_conch15_model_and_preprocess(device: torch.device):
    titan = AutoModel.from_pretrained(
        "MahmoodLab/TITAN",
        trust_remote_code=True,
    )
    model, preprocess = titan.return_conch()
    model = model.to(device).eval()
    return model, preprocess


# @torch.inference_mode()
# def encode_images_conch15(model, x: torch.Tensor) -> torch.Tensor:
#     # raw image features for MIL
#     z = model.encode_image(x, proj_contrast=False, normalize=False)

#     if isinstance(z, dict):
#         for k in ["image_embeds", "embeds", "features", "proj"]:
#             if k in z:
#                 z = z[k]
#                 break
#     if isinstance(z, (tuple, list)):
#         z = z[0]

#     return z.float()

@torch.inference_mode()
def encode_images_conch15(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "encode_image"):
        z = model.encode_image(x, proj_contrast=False, normalize=False)
    else:
        z = model(x)

    if isinstance(z, dict):
        for k in ["image_embeds", "embeds", "features", "proj"]:
            if k in z:
                z = z[k]
                break
    if isinstance(z, (tuple, list)):
        z = z[0]

    return z.float()
# ============================================================
# I/O helpers
# ============================================================
def read_patch_map(patch_map_csv: str) -> pd.DataFrame:
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


def save_h5(out_h5: str, feats: np.ndarray, coords: np.ndarray, patch_size_level0: int | None = None):
    os.makedirs(os.path.dirname(out_h5), exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("features", data=feats, compression="gzip")
        dset = f.create_dataset("coords", data=coords, compression="gzip")
        if patch_size_level0 is not None:
            dset.attrs["patch_size_level0"] = int(patch_size_level0)


def save_pt(out_pt: str, feats: np.ndarray, coords: np.ndarray):
    os.makedirs(os.path.dirname(out_pt), exist_ok=True)
    torch.save(
        {
            "features": torch.from_numpy(feats),
            "coords": torch.from_numpy(coords),
        },
        out_pt,
    )


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
    patch_size_level0: int | None = None,
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
        batch_paths = patch_paths[i:i + batch_size]

        imgs = []
        for p in batch_paths:
            pil = npy_to_pil_rgb(p)
            imgs.append(preprocess(pil))

        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            z = encode_images_conch15(model, x)

        feats_chunks.append(z.detach().cpu().numpy().astype(np.float32))

    feats = np.concatenate(feats_chunks, axis=0).astype(np.float32)

    out_h5 = os.path.join(out_h5_dir, f"{slide_id}.h5")
    out_pt = os.path.join(out_pt_dir, f"{slide_id}.pt")
    save_h5(out_h5, feats, coords, patch_size_level0=patch_size_level0)
    save_pt(out_pt, feats, coords)

    return {
        "slide_id": slide_id,
        "status": "ok",
        "n": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides_root", type=str, required=True,
                    help="Root dir with per-slide folders containing .npy patches + patch_map.csv")
    ap.add_argument("--patch_map_name", type=str, default="patch_map.csv")
    ap.add_argument("--out_h5_dir", type=str, required=True)
    ap.add_argument("--out_pt_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_patches", type=int, default=0,
                    help="0 = all patches")
    ap.add_argument("--patch_size_level0", type=int, default=224,
                    help="512 for 20x slides, 1024 for 40x slides if patch size is 512 at target mag")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("Loading CONCH v1.5 via TITAN...")
    model, preprocess = load_conch15_model_and_preprocess(device)

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
                    patch_size_level0=args.patch_size_level0,
                )
            )
        except Exception as e:
            rows.append({
                "slide_id": os.path.basename(sd),
                "status": f"error: {e}",
                "n": 0,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_h5_dir), exist_ok=True)
    out_csv = os.path.join(os.path.dirname(args.out_h5_dir), "conch15_image_feature_extraction_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")
    print(df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

# python test_text_encoder\extract_conch15_feats.py --slides_root "C:\Users\Vivian\Documents\CONCH\patches_tiled\patches_10x\40x" --out_h5_dir "C:\Users\Vivian\Documents\CONCH\run2_conch15_img_feats\h5" --out_pt_dir "C:\Users\Vivian\Documents\CONCH\run2_conch15_img_feats\pt" --device cuda:0 --batch_size 32 --patch_size_level0 224