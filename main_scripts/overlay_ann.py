
# -------------------------------------------------
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



def parse_all_annotations(annotation_path, scale_factor=1.0):
    # to visualize og annotations without offset
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
                x = float(v.get("X")) / scale_factor
                y = float(v.get("Y")) / scale_factor
                polygon.append((x, y))
            all_annotations.append({
                "name": name,
                "polygon": polygon
            })
    return all_annotations

def parse_all_annotations2(annotation_path):
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
    # ax.imshow(image, extent=(x_offset, x_offset + image.shape[1],
    #                      y_offset + image.shape[0], y_offset),
    #       origin='upper')


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
            # x_new = (x_um - 10000) / (pixel_size_x * downsample_factor) # apply offset, convert back to pixels, and downsample
            # y_new = (y_um - 10000) / (pixel_size_y * downsample_factor)
            print(f"Shifted (Series N pixels): ({x_new:.2f}, {y_new:.2f})")
            new_polygon.append((x_new, y_new))
        shifted.append({"name": name, "polygon": new_polygon})
    return shifted

def main():

    # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 52 B.vsi" # series 8, scale 4
    # annotation_path = r"C:\Users\Vivian\OneDrive - Queen's University\research2025\breast_project\Amoon_annotation_code\halo_annotations\PT 52 B (1).annotations"
    
    # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi"
    # annotation_path = r"C:\Users\Vivian\OneDrive - Queen's University\research2025\breast_project\Amoon_annotation_code\halo_annotations\FA 57B (2).annotations"
    
    vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 35 B.vsi"
    annotation_path = r"C:\Users\Vivian\OneDrive - Queen's University\research2025\breast_project\Amoon_annotation_code\halo_annotations\PT 35 B.annotations"
    

    # scale_factor = 10.09  # adjust based on downsample level
    # series = 0  # select correct resolution level
    # tile_size = 100

    # --- Parameters ---
    series = 8  # downsampled series
    # scale_factor = 4  # original to 20x full-res
    downsample_factor = 4  # series 8 is 4x downsampled from 20x
    pixel_size_x = 0.3433209  # in microns
    pixel_size_y = 0.3433189
    # offset_x_um = ((-107248.19992049833)-(-89145.87))  # FA 57B # offset of cropped 20x image in microns
    # offset_y_um = ((-73463.97964187959)-(-66619.44)) # Fa 57B Origin	
    # offset_x_um = ((-107248.19992049833)-(-96073.36))  # PT 52 B (-107248.19992049833, -73463.97964187959) (107248.19992049833 - 96073.36)
    # offset_y_um = ((-73463.97964187959)-(-73150.42))  # PT 52 B
    offset_x_um = ((-107248.19992049833)-(-96357.44876760358))  # PT 35 B # offset of cropped 20x image in microns
    offset_y_um = ((-73463.97964187959)-(-72208.0167180309)) # PT 35 B Origin

    # Load image
    image = load_vsi_image(vsi_path, series=series)
    # image = load_slide_in_tiles(vsi_path, tile_size=tile_size, series=series)

    # # Parse annotations
    annotations = parse_all_annotations(annotation_path, scale_factor=downsample_factor)
    annotations2 = parse_all_annotations2(annotation_path)
    # --------------------------
    # --- Parse and adjust annotations ---
    annotations_shifted = apply_offset_to_annotations(
        annotations2,
        offset_x_um, offset_y_um,
        pixel_size_x, pixel_size_y,
        downsample_factor
    )
    # --------------------------

    # --- Visualize ---
    # visualize_annotations(image, annotations)
    visualize_annotations(image, annotations_shifted)

if __name__ == "__main__":#
    main()
