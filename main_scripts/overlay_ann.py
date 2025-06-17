import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import cv2
import javabridge
import os
from tqdm import tqdm
import bioformats
from bioformats import load_image
import pandas as pd
import cv2

def load_vsi_image(vsi_path, series=0):
    javabridge.start_vm(class_path=bioformats.JARS)
    try:
        image = load_image(vsi_path, series=series, rescale=False)
        image_np = np.array(image)
    finally:
        javabridge.kill_vm()
    return image_np

def load_slide_in_tiles(vsi_path, tile_size=1024, series=8):
    javabridge.start_vm(class_path=os.environ.get("CLASSPATH", bioformats.JARS))
    ImageReader = javabridge.JClassWrapper("loci.formats.ImageReader")
    reader = ImageReader()

    reader.setId(vsi_path)
    reader.setSeries(series)

    width = reader.getSizeX()
    height = reader.getSizeY()
    channels = reader.getRGBChannelCount()

    full_image = np.zeros((height, width, channels), dtype=np.uint8)


    for y in tqdm(range(0, height, tile_size), desc="Reading tiles"):
        for x in range(0, width, tile_size):
            tile_width = min(tile_size, width - x)
            tile_height = min(tile_size, height - y)

            # Skip if tile size is invalid
            if tile_width <= 0 or tile_height <= 0:
                continue

            try:
                byte_array = reader.openBytes(0, x, y, tile_width, tile_height)
                tile = np.frombuffer(byte_array, dtype=np.uint8).reshape((tile_height, tile_width, channels))
                full_image[y:y + tile_height, x:x + tile_width] = tile
            except javabridge.jutil.JavaException as e:
                print(f"Tile at ({x}, {y}) caused error: {e}")

    reader.close()
    javabridge.kill_vm()
    return full_image

def parse_all_annotations(annotation_path):
    # without scaling down
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    all_annotations = []

    for annot in root.findall("Annotation"):
        name = annot.attrib.get("Name", "Unnamed")
        regions = annot.find("Regions")
        if regions is None:
            continue
        for region in regions.findall("Region"):
            polygon = []
            for v in region.findall(".//Vertices/V"):
                x = float(v.get("X")) 
                y = float(v.get("Y"))
                polygon.append((x, y))
            all_annotations.append({
                "name": name,
                "polygon": polygon
            })
    return all_annotations

def visualize_annotations(image, annotations):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)

    # Assign each annotation a unique color
    color_map = {}
    cmap = plt.cm.get_cmap("tab10")  # 10 distinct colors

    for i, annot in enumerate(annotations):
        name = annot["name"]
        polygon = annot["polygon"]
        xs, ys = zip(*polygon)

        if name not in color_map:
            color_map[name] = cmap(i % 10)

        ax.plot(xs, ys, color=color_map[name], linewidth=2, label=name)

    ax.axis("off")
    ax.legend()
    plt.show()

def apply_offset_to_annotations(annotations, offset_x_um, offset_y_um, pixel_size_x, pixel_size_y, downsample_factor=1.0):
    """
    Shift and scale annotation points based on physical offset and downsampling.

    Args:
        annotations: List of dicts with polygon points.
        offset_x_um: X offset in microns (top-left corner of cropped image).
        offset_y_um: Y offset in microns.
        pixel_size_x: Microns per pixel (horizontal).
        pixel_size_y: Microns per pixel (vertical).
        downsample_factor: Additional downsampling applied to the target image.
    
    Returns:
        List of shifted annotations.
    """
    shifted = []
    for annot in annotations:
        name = annot["name"]
        polygon = annot["polygon"]
        new_polygon = []
        for x, y in polygon:
            print(f"Original (full-res pixels): ({x:.2f}, {y:.2f})")
            x_um = x * pixel_size_x # convert pixel to microns (offset is in microns)
            y_um = y * pixel_size_y
            x_new = (x_um + offset_x_um) / (pixel_size_x*downsample_factor) # apply offset, convert back to pixels, and downsample
            y_new = (y_um + offset_y_um) / (pixel_size_y*downsample_factor)
            print(f"Shifted (Series N pixels): ({x_new:.2f}, {y_new:.2f})")
            new_polygon.append((x_new, y_new))
        shifted.append({"name": name, "polygon": new_polygon})
    return shifted

# ------------------ added functions for mask generation ------------------
from shapely.geometry import Point, Polygon
from tqdm import tqdm
def parse_annotation_file(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    all_polygons = []
    for annot in root.findall("Annotation"):

        name = annot.attrib.get("Name", "Unnamed")

        if name.lower() == "border" or name.lower() == "normal/ other":
            continue  # ✅ Skip testing/debug border annotations

        regions = annot.find("Regions")
        if regions is None:
            continue
        for region in regions.findall("Region"):
            polygon = []
            for v in region.findall(".//Vertices/V"):
                x = float(v.get("X"))
                y = float(v.get("Y"))
                polygon.append((x, y))
        
            if len(polygon) < 3:
                continue  # Skip invalid polygons

            # ✅ Close the polygon
            if polygon[0] != polygon[-1]:
                polygon.append(polygon[0])
            
            all_polygons.append(polygon)

    return all_polygons

def apply_offset_to_annotations(annotations, offset_x_um, offset_y_um,
                                 pixel_size_x, pixel_size_y, downsample_factor=1.0):
    shifted = []
    for polygon in annotations:
        new_polygon = []
        for x, y in polygon:
            x_um = x * pixel_size_x
            y_um = y * pixel_size_y
            x_new = (x_um + offset_x_um) / (pixel_size_x * downsample_factor)
            y_new = (y_um + offset_y_um) / (pixel_size_y * downsample_factor)
            new_polygon.append((x_new, y_new))
        shifted.append(new_polygon)
    return shifted

def generate_patch_mask_csv(patch_map_csv, shifted_annotations):
    df = pd.read_csv(patch_map_csv)
    df['InsideAnnotation'] = False

    patch_polygons = [Polygon(p) for p in shifted_annotations]

    for i, row in df.iterrows():
        x, y = row['x'], row['y']
        patch_center = Point(x + 112, y + 112)  # Assuming patch size is 224x224
        for poly in patch_polygons:
            if poly.contains(patch_center):
                df.at[i, 'InsideAnnotation'] = True
                break

    return df[['patch_file', 'x', 'y', 'InsideAnnotation']]

def batch_generate_patch_csvs_with_offset(annotation_dir, slide_root, output_dir, metadata_csv,
                                          pixel_size_x, pixel_size_y, downsample_factor):
    os.makedirs(output_dir, exist_ok=True)
    metadata_df = pd.read_csv(metadata_csv)
    series0_offset = (-107248.19992049833, -73463.97964187959)

    for fname in tqdm(os.listdir(annotation_dir)):
        if not fname.endswith(".annotations"):
            continue

        base = os.path.splitext(fname)[0]
        slide_name = base + ".vsi"
        row = metadata_df[metadata_df['Filename'] == slide_name]
        if row.empty:
            print(f"⚠️ Metadata missing for {slide_name}")
            continue
        
        output_path = os.path.join(output_dir, f"{base}_patch_mask.csv")
        if os.path.exists(output_path):
            print(f"⏭️ Skipping {base}: CSV already exists")
            continue  # ✅ Skip if already generated
        
        try:
            offset_str = row.iloc[0]['Series_6'].strip("()")
            offset_values = tuple(map(float, offset_str.split(",")))
            offset_x_um = series0_offset[0] - offset_values[0]
            offset_y_um = series0_offset[1] - offset_values[1]
        except:
            print(f"⚠️ Offset parse error for {slide_name}")
            continue

        annotation_path = os.path.join(annotation_dir, fname)
        annotations = parse_annotation_file(annotation_path)
        shifted_annotations = apply_offset_to_annotations(
            annotations, offset_x_um, offset_y_um,
            pixel_size_x, pixel_size_y, downsample_factor
        )

        patch_map_path = os.path.join(slide_root, base, "patch_map.csv")
        if not os.path.exists(patch_map_path):
            print(f"⚠️ Missing patch_map.csv for {base}")
            continue

        patch_mask_df = generate_patch_mask_csv(patch_map_path, shifted_annotations)
        output_path = os.path.join(output_dir, f"{base}_patch_mask.csv")
        patch_mask_df.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")


def main():
    # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 52 B.vsi" # series 8, scale 4
    # annotation_path = r"C:\Users\Vivian\OneDrive - Queen's University\research2025\breast_project\Amoon_annotation_code\halo_annotations\PT 52 B.annotations"
    
    # # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi"
    # # annotation_path = r"C:\Users\Vivian\OneDrive - Queen's University\research2025\breast_project\Amoon_annotation_code\halo_annotations\FA 57B.annotations"
    
    # # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 35 B.vsi"
    # # annotation_path = r"C:\Users\Vivian\OneDrive - Queen's University\research2025\breast_project\Amoon_annotation_code\halo_annotations\PT 35 B.annotations"
    

    # # scale_factor = 10.09  # adjust based on downsample level
    # # series = 0  # select correct resolution level
    # # tile_size = 100

    # # --- Parameters ---
    # series = 8  # downsampled series
    # # scale_factor = 4  # original to 20x full-res
    # downsample_factor = 4  # series 8 is 4x downsampled from 20x
    # pixel_size_x = 0.3433209  # in microns
    # pixel_size_y = 0.3433189
    # # offset_x_um = ((-107248.19992049833)-(-89145.87))  # FA 57B # offset of cropped 20x image in microns
    # # offset_y_um = ((-73463.97964187959)-(-66619.44)) # Fa 57B Origin	
    # offset_x_um = ((-107248.19992049833)-(-96073.36))  # PT 52 B (-107248.19992049833, -73463.97964187959) (107248.19992049833 - 96073.36)
    # offset_y_um = ((-73463.97964187959)-(-73150.42))  # PT 52 B
    # # offset_x_um = ((-107248.19992049833)-(-96357.44876760358))  # PT 35 B # offset of cropped 20x image in microns
    # # offset_y_um = ((-73463.97964187959)-(-72208.0167180309)) # PT 35 B Origin
    # # offset_x_um = 0
    # # offset_y_um = 0

    # # Load image
    # # image = load_vsi_image(vsi_path, series=series)
    # # image = load_slide_in_tiles(vsi_path, tile_size=tile_size, series=series)

    # # # Parse annotations
    # annotations = parse_all_annotations(annotation_path)
    # # --------------------------
    # # --- Parse and adjust annotations ---
    # annotations_shifted = apply_offset_to_annotations(
    #     annotations,
    #     offset_x_um, offset_y_um,
    #     pixel_size_x, pixel_size_y,
    #     downsample_factor
    # )
    # # --------------------------
    # # testing with reconstructed image 
    # from PIL import Image
    # import numpy as np
    # Image.MAX_IMAGE_PIXELS = None
    # # Load the saved PNG image
    # reconstructed_image = np.array(Image.open(r"C:\Users\Vivian\Documents\CONCH\PT 52B_reconstructed_image.png"))


    # # --- Visualize ---
    # # visualize_annotations(image, annotations)
    # # visualize_annotations(image, annotations_shifted)
    # visualize_annotations(reconstructed_image, annotations_shifted)


    # ----------------------------------------------------------------
    # --- Generate masks and save patch mask csv ---
    # annotation_dir = r"C:\Users\Vivian\OneDrive - Queen's University\research2025\breast_project\Amoon_annotation_code\halo_annotations"
    # offset_csv = r"C:\Users\Vivian\Documents\CONCH\PT_plane_metadata.csv"
    # save_dir = r"path\to\save\masks"
    batch_generate_patch_csvs_with_offset(
    annotation_dir=r"C:\Users\Vivian\OneDrive - Queen's University\research2025\breast_project\Amoon_annotation_code\halo_annotations",
    slide_root=r"C:\Users\Vivian\Documents\CONCH\all_patches\patches_5x\20x\FA",
    output_dir=r"C:\Users\Vivian\Documents\CONCH\series8_5x_masks",
    metadata_csv=r"C:\Users\Vivian\Documents\CONCH\FA_plane_metadata.csv",
    pixel_size_x=0.3433209,
    pixel_size_y=0.3433189,
    downsample_factor=4
    )


if __name__ == "__main__":
    main()
