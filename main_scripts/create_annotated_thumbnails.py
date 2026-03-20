#!/usr/bin/env python
# make_annotated_thumbs.py

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# <-- your functions (same ones you already use)
from overlay_ann import parse_annotation_file, apply_offset_to_annotations

# Optional Bio-Formats fallback
def _try_start_jvm():
    try:
        import javabridge, bioformats
        try:
            javabridge.get_env()
            return True
        except:
            javabridge.start_vm(class_path=bioformats.JARS)
            return True
    except Exception:
        return False

def _kill_jvm():
    try:
        import javabridge
        javabridge.kill_vm()
    except Exception:
        pass

def load_series_image_or_thumb(slide_path, series_index=6, thumb_max_width=2000, use_bioformats=False):
    """
    Return a PIL RGB image for display:
      1) If a PNG/JPG with the same basename exists, use it (fast).
      2) Else, if use_bioformats=True, read the series via Bio-Formats.
    """
    base, _ = os.path.splitext(slide_path)
    for ext in (".png", ".jpg", ".jpeg"):
        thumb = base + ext
        if os.path.exists(thumb):
            img = Image.open(thumb).convert("RGB")
            if img.width > thumb_max_width:
                r = thumb_max_width / img.width
                img = img.resize((thumb_max_width, int(img.height * r)), Image.LANCZOS)
            return img, None  # None = unknown original size (assume coords already match)
    if not use_bioformats:
        raise FileNotFoundError(f"No thumbnail next to slide and Bio-Formats disabled: {slide_path}")

    import bioformats
    arr = bioformats.load_image(slide_path, series=series_index, rescale=False)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = (255 * (arr - arr.min()) / (arr.ptp() + 1e-6)).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    orig_size = (img.width, img.height)
    if img.width > thumb_max_width:
        r = thumb_max_width / img.width
        img = img.resize((thumb_max_width, int(img.height * r)), Image.LANCZOS)
    return img, orig_size  # orig_size used for scaling polygons

def draw_polygons_on_image(img, polygons_xy, outline=(255, 0, 0), fill_alpha=60, outline_width=3):
    """
    polygons_xy: list of Nx2 arrays in img pixel coordinates.
    """
    if img.mode != "RGBA":
        base_rgba = img.convert("RGBA")
    else:
        base_rgba = img

    overlay = Image.new("RGBA", base_rgba.size, (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay, "RGBA")
    fill = (255, 0, 0, int(fill_alpha))

    for poly in polygons_xy:
        if poly is None or len(poly) < 3:
            continue
        pts = [tuple(map(float, p)) for p in poly]
        odraw.polygon(pts, fill=fill)

    composed = Image.alpha_composite(base_rgba, overlay).convert("RGB")

    draw = ImageDraw.Draw(composed)
    for poly in polygons_xy:
        if poly is None or len(poly) < 3:
            continue
        pts = [tuple(map(float, p)) for p in poly]
        draw.line(pts + [pts[0]], fill=outline, width=outline_width, joint="curve")
    return composed

def batch_save_slide_thumbnails_with_annotations(
    annotation_dir,
    slide_root,
    output_dir,
    metadata_csv,
    pixel_size_x, pixel_size_y,
    downsample_factor,
    series_index=6,
    thumb_max_width=1800,
    use_bioformats=False,
    outline_width=3,
    fill_alpha=60
):
    os.makedirs(output_dir, exist_ok=True)
    meta = pd.read_csv(metadata_csv)

    # Your reference (unchanged from your other script)
    series0_offset = (-107248.19992049833, -73463.97964187959)

    jvm_started = False
    if use_bioformats:
        jvm_started = _try_start_jvm()
        if not jvm_started:
            print("⚠️ Could not start JVM; Bio-Formats disabled. Will only use pre-exported thumbnails.")

    try:
        for fname in os.listdir(annotation_dir):
            if not fname.endswith(".annotations"):
                continue

            base = os.path.splitext(fname)[0]  # e.g., "FA 56B"
            slide_path = os.path.join(slide_root, base + ".vsi")
            if not os.path.exists(slide_path):
                print(f"⚠️ Missing slide: {slide_path}")
                continue

            out_img = os.path.join(output_dir, f"{base}_thumb_annotated.jpg")
            if os.path.exists(out_img):
                print(f"⏭️ Skipping {base}: overlay already exists")
                continue

            row = meta[meta['Filename'] == (base + ".vsi")]
            if row.empty:
                print(f"⚠️ Metadata missing for {base}.vsi")
                continue

            try:
                off = row.iloc[0]['Series_6'].strip("()")
                off_x, off_y = map(float, off.split(","))
                offset_x_um = series0_offset[0] - off_x
                offset_y_um = series0_offset[1] - off_y
            except Exception as e:
                print(f"⚠️ Offset parse error for {base}: {e}")
                continue

            # Parse annotations (your format)
            ann_path = os.path.join(annotation_dir, fname)
            annotations = parse_annotation_file(ann_path)

            # Convert to pixel coords at the same level you visualize (series_index)
            shifted_annotations = apply_offset_to_annotations(
                annotations,
                offset_x_um, offset_y_um,
                pixel_size_x, pixel_size_y,
                downsample_factor
            )  # -> list of (N,2) arrays in *series* pixel coords

            # Load display image and (maybe) original series size
            try:
                img, orig_series_wh = load_series_image_or_thumb(
                    slide_path,
                    series_index=series_index,
                    thumb_max_width=thumb_max_width,
                    use_bioformats=use_bioformats
                )
            except Exception as e:
                print(f"❌ Could not load overview for {base}: {e}")
                continue

            # Scale polygons if we resized from original series size
            if orig_series_wh is not None:
                sx = img.width  / float(orig_series_wh[0])
                sy = img.height / float(orig_series_wh[1])
            else:
                # Assume the thumbnail is already at the same pixel scale as shifted_annotations
                sx = sy = 1.0

            scaled_polys = []
            for poly in shifted_annotations:
                poly = np.asarray(poly, dtype=float)
                poly = np.stack([poly[:, 0] * sx, poly[:, 1] * sy], axis=1)
                scaled_polys.append(poly)

            composed = draw_polygons_on_image(
                img,
                scaled_polys,
                outline=(255, 0, 0),
                fill_alpha=fill_alpha,
                outline_width=outline_width
            )
            composed.save(out_img, quality=95)
            print(f"✅ Saved: {out_img}")
    finally:
        if jvm_started:
            _kill_jvm()

def main():
    ap = argparse.ArgumentParser(description="Save slide thumbnails with annotation overlays")
    ap.add_argument("--annotation-dir", required=True)
    ap.add_argument("--slide-root",     required=True, help="Folder with .vsi slides")
    ap.add_argument("--output-dir",     required=True)
    ap.add_argument("--metadata-csv",   required=True)
    ap.add_argument("--pixel-size-x",   type=float, required=True, help="µm per pixel X at series 0")
    ap.add_argument("--pixel-size-y",   type=float, required=True, help="µm per pixel Y at series 0")
    ap.add_argument("--downsample-factor", type=float, required=True, help="factor to series you visualize (e.g., series 6)")
    ap.add_argument("--series-index",   type=int, default=6)
    ap.add_argument("--thumb-max-width", type=int, default=1800)
    ap.add_argument("--use-bioformats", action="store_true", help="Use Bio-Formats if no thumbnail exists")
    ap.add_argument("--outline-width",  type=int, default=3)
    ap.add_argument("--fill-alpha",     type=int, default=60)
    args = ap.parse_args()

    batch_save_slide_thumbnails_with_annotations(
        annotation_dir=args.annotation_dir,
        slide_root=args.slide_root,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        pixel_size_x=args.pixel_size_x,
        pixel_size_y=args.pixel_size_y,
        downsample_factor=args.downsample_factor,
        series_index=args.series_index,
        thumb_max_width=args.thumb_max_width,
        use_bioformats=args.use_bioformats,
        outline_width=args.outline_width,
        fill_alpha=args.fill_alpha
    )

if __name__ == "__main__":
    main()
