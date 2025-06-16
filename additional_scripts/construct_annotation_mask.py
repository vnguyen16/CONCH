import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import javabridge
import bioformats
import numpy as np

def parse_annotation(annotation_path, scale_factor=1.0):
    """
    Parse XML annotation and scale the polygon points.
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    coords = []
    for vertex in root.findall(".//Vertices/V"):
        x = int(vertex.get("X")) / scale_factor
        y = int(vertex.get("Y")) / scale_factor
        coords.append((x, y))
    return coords

def polygon_centroid(coords):
    """
    Accurate centroid calculation for arbitrary 2D polygon.
    """
    coords = np.array(coords)
    x = coords[:, 0]
    y = coords[:, 1]
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * np.sum(a)
    Cx = (1 / (6 * area)) * np.sum((x[:-1] + x[1:]) * a)
    Cy = (1 / (6 * area)) * np.sum((y[:-1] + y[1:]) * a)
    return Cx, Cy

def adjust_annotation_offset(image, coords):
    """
    Align the polygon centroid to the image center.
    """
    image_height, image_width = image.shape[:2]
    image_center = (image_width / 2, image_height / 2)
    annotation_center = polygon_centroid(coords)
    offset_x = image_center[0] - annotation_center[0]
    offset_y = image_center[1] - annotation_center[1]
    adjusted_coords = [(x + offset_x, y + offset_y) for x, y in coords]
    return adjusted_coords

def create_polygon_mask(image_shape, polygon_coords):
    """
    Create binary mask: white inside polygon, black outside.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(polygon_coords, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=255)  # Fill with white inside
    return mask

def display_overlay(image, polygon_coords, mask):
    """
    Show the tissue image, annotation outline, and overlay mask.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)
    xs, ys = zip(*polygon_coords)
    ax.plot(xs, ys, color='white', linewidth=2)  # Overlay polygon
    ax.imshow(mask, cmap='gray', alpha=0.3)      # Overlay mask (semi-transparent)
    ax.axis('off')
    plt.show()


def load_vsi_image(vsi_path, series=0):
    """
    Load a specific series (resolution level) from a VSI file using Bio-Formats.
    """
    javabridge.start_vm(class_path=bioformats.JARS)
    try:
        image = bioformats.load_image(vsi_path, series=series, rescale=False)
        image_np = np.array(image)
    finally:
        javabridge.kill_vm()

    return image_np

def align_to_top_left_origin(image, coords):
    """
    Aligns the annotation polygon by shifting it so its top-left corner
    matches the top-left of the image (origin alignment).
    """
    image_height, image_width = image.shape[:2]

    # Bounding box of the annotation
    min_x = min(x for x, _ in coords)
    min_y = min(y for _, y in coords)

    # Align annotation so its top-left starts at (0, 0)
    adjusted_coords = [(x - min_x, y - min_y) for x, y in coords]

    # Optional: Center it on the image if it needs to be placed visually in context
    # offset_x = (image_width - (max(x for x, _ in adjusted_coords))) // 2
    # offset_y = (image_height - (max(y for _, y in adjusted_coords))) // 2
    offset_x = 0  # No offset for top-left alignment
    offset_y = 0  # No offset for top-left alignment

    aligned_coords = [(x + offset_x, y + offset_y) for x, y in adjusted_coords]

    return aligned_coords


def main():
    import matplotlib.image as mpimg  # needed only if using PNG elsewhere
    import cv2

    # --- Input paths ---
    # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi" # series 7, scale 2
    # annotation_path = r"C:\Users\Vivian\Downloads\FA 57B.annotations" 
    # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 41 B.vsi"
    # annotation_path = r"C:\Users\Vivian\Downloads\PT 41 B.annotations"
    vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 52 B.vsi"  # series 8, scale 4
    annotation_path = r"C:\Users\Vivian\Downloads\PT 52 B.annotations"

    # --- Parameters ---
    series = 8             # Select desired resolution level from .vsi
    scale_factor = 4      # Scale for annotations (adjust to match series level)

    # --- Load image from VSI ---
    print(f"Loading VSI image from series {series}...")
    image = load_vsi_image(vsi_path, series=series)
    print(f"Loaded image shape: {image.shape}")

    # --- Load and scale annotation ---
    coords = parse_annotation(annotation_path, scale_factor=scale_factor)
    print(f"Parsed {len(coords)} annotation points.")

    # --- Adjust polygon alignment ---
    adjusted_coords = adjust_annotation_offset(image, coords)
    # adjusted_coords = align_to_top_left_origin(image, coords)

    # --- Create mask ---
    mask = create_polygon_mask(image.shape, adjusted_coords)
    # mask = create_polygon_mask(image.shape, coords)

    # debugging
    print("First polygon point:", coords[0])
    print("Image shape:", image.shape)


    # --- Overlay and display ---
    display_overlay(image, adjusted_coords, mask)
    # display_overlay(image, coords, mask)

    # --- Save mask image ---
    # cv2.imwrite("annotation_mask.png", mask)
    # print("Saved binary mask to annotation_mask.png")


if __name__ == "__main__":
    main()