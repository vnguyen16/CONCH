"""
This script overlays the ground truth annotation on a downsampled whole-slide image (WSI).
"""
# #  almost works --> slightly offcentered
# import xml.etree.ElementTree as ET
# import numpy as np
# import matplotlib.pyplot as plt
# import bioformats
# import javabridge

# def load_vsi_image(vsi_path, series=2):  # Downsampled level (e.g., level 2 = 4× down)
#     """
#     Load a VSI image using Bio-Formats and convert it to a NumPy array.
#     """
#     javabridge.start_vm(class_path=bioformats.JARS)

#     try:
#         image = bioformats.load_image(vsi_path, series=series, rescale=False)
#         image_np = np.array(image)
#     finally:
#         javabridge.kill_vm()

#     return image_np

# def parse_annotation(annotation_path, scale_factor=1.0):
#     """
#     Parse the .annotations XML file and extract scaled polygon coordinates.
#     """
#     tree = ET.parse(annotation_path)
#     root = tree.getroot()

#     coords = []
#     for vertex in root.findall(".//Vertices/V"):
#         x = int(vertex.get("X")) / scale_factor
#         y = int(vertex.get("Y")) / scale_factor
#         coords.append((x, y))
    
#     return coords

# def overlay_annotation(image, coords):
#     """
#     Overlay annotation coordinates on the image using matplotlib.
#     """
#     xs, ys = zip(*coords)

#     plt.figure(figsize=(15, 15))
#     # need another type of image conversion
#     # try in lower dimension level
    
#     plt.imshow(image)
#     plt.plot(xs, ys, 'r-', linewidth=2, label="Annotation Polygon")
#     # plt.scatter(xs, ys, color='blue', s=5)
#     # plt.title("Annotation Overlay on Downsampled WSI")
#     plt.axis("off")
#     # plt.legend()
#     plt.show()

# # adding function to calculate polygon centroid and adjust offset
# def polygon_centroid(coords):
#     """
#     Compute the accurate centroid of a 2D polygon using the area-weighted formula.
#     Assumes the polygon is closed (first and last points are not the same).
#     """
#     coords = np.array(coords)
#     x = coords[:, 0]
#     y = coords[:, 1]

#     # Shift the points so (x[n+1], y[n+1]) = (x[0], y[0])
#     x = np.append(x, x[0])
#     y = np.append(y, y[0])

#     a = x[:-1] * y[1:] - x[1:] * y[:-1]
#     area = 0.5 * np.sum(a)

#     Cx = (1 / (6 * area)) * np.sum((x[:-1] + x[1:]) * a)
#     Cy = (1 / (6 * area)) * np.sum((y[:-1] + y[1:]) * a)

#     return Cx, Cy


# def adjust_annotation_offset(image, coords):
#     """
#     Adjust annotation coordinates to properly align with the WSI image.
#     If annotations are centered incorrectly, this will shift them accordingly.
#     """
#     image_height, image_width = image.shape[:2]
#     annotation_height = max(y for _, y in coords) - min(y for _, y in coords)
#     annotation_width = max(x for x, _ in coords) - min(x for x, _ in coords)
    
#     # Calculate the difference in center position (centroid)
#     image_center = (image_width / 2, image_height / 2)
#     # annotation_center = (sum(x for x, _ in coords) / len(coords), sum(y for _, y in coords) / len(coords))

#     # using the polygon centroid function***
#     annotation_center = annotation_center = polygon_centroid(coords)

#     # Calculate offset
#     offset_x = image_center[0] - annotation_center[0]
#     offset_y = image_center[1] - annotation_center[1]

#     # Apply the offset
#     adjusted_coords = [(x + offset_x, y + offset_y) for x, y in coords]
#     return adjusted_coords

# def main():
#     # Paths
#     vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi" 
#     annotation_path = r"C:\Users\Vivian\Downloads\FA 57B.annotations"  # <-- adjust as needed
#     # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 41 B.vsi"
#     # annotation_path = r"C:\Users\Vivian\Downloads\PT 41 B.annotations"

#     # Parameters
#     series = 7  # Downsampled level (e.g., level 2 for 4× downsampling)
#     scale_factor = 2  # Scale down annotations to match series 2 resolution

#     # Load image
#     print(f"Loading VSI image from series {series}...")
#     image = load_vsi_image(vsi_path, series=series)
#     print(f"Loaded image with shape: {image.shape}")

#     # Load and scale annotation
#     print("Parsing and scaling annotation...")
#     coords = parse_annotation(annotation_path, scale_factor=scale_factor)
#     print(f"Loaded {len(coords)} annotation points (scaled).")

#     # Adjust the annotation offset if misaligned
#     print("Adjusting annotation offset...")
#     adjusted_coords = adjust_annotation_offset(image, coords)
#     print(f"Adjusted {len(adjusted_coords)} annotation points.")

#     # Show overlay
#     print("Displaying overlay...")
#     overlay_annotation(image, adjusted_coords)

# if __name__ == "__main__":
#     main()

# ------------------------------------------------

#  testing out different methods to load the image and overlay mask

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

    # manual offset
    offset_x = abs(-89145.87) / (0.3433209*4)
    offset_y = abs(-66619.44) / (0.3433189*4)

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

def shift_annotation_to_downsampled(coords, offset_x_um, offset_y_um, px_size_x, px_size_y, downsample_factor):
    """
    Shift annotation coordinates to align with a downsampled 20x image.
    """
    adjusted_coords = []
    for x, y in coords:
        # Convert full-res pixel coordinates to microns
        x_um = x * px_size_x
        y_um = y * px_size_y

        # Shift relative to 20x origin and convert back to 20x pixels
        x_shifted_20x = (x_um - offset_x_um) / px_size_x
        y_shifted_20x = (y_um - offset_y_um) / px_size_y

        # Downsample to match the image resolution (e.g., 4x smaller)
        x_down = x_shifted_20x / downsample_factor
        y_down = y_shifted_20x / downsample_factor

        adjusted_coords.append((x_down, y_down))
    return adjusted_coords

def main():
    import matplotlib.image as mpimg  # needed only if using PNG elsewhere
    import cv2

    # # # --- Input paths ---
    # # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi"
    # # annotation_path = r"C:\Users\Vivian\Downloads\FA 57B.annotations"
    # # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 41 B.vsi"
    # # annotation_path = r"C:\Users\Vivian\Downloads\PT 41 B.annotations"
    # # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 52 B.vsi"  # series 8, scale 4
    # # annotation_path = r"C:\Users\Vivian\Downloads\PT 52 B.annotations"
    
    vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi"  # series 8, scale 4
    annotation_path = r"C:\Users\Vivian\OneDrive - Queen's University\research2025\breast_project\Amoon_annotation_code\halo_annotations\FA 57B (2).annotations"


    # --- Parameters ---
    # series = 0             # Select desired resolution level from .vsi
    # scale_factor = 10.09      # Scale for annotations (adjust to match series level)
    series = 8
    scale_factor = 4  # Adjust based on the series level (e.g., 4 for series 8)

    # --- Load image from VSI ---
    print(f"Loading VSI image from series {series}...")
    image = load_vsi_image(vsi_path, series=series)
    print(f"Loaded image shape: {image.shape}")

    # --- Load and scale annotation ---
    coords = parse_annotation(annotation_path, scale_factor=scale_factor)
    print(f"Parsed {len(coords)} annotation points.")

    # --- Adjust polygon alignment ---
    # adjusted_coords = adjust_annotation_offset(image, coords)
    # adjusted_coords = align_to_top_left_origin(image, coords)

    # -----
    offset_x_um = -89145.87
    offset_y_um = -66619.44
    px_size_x = 0.3433209
    px_size_y = 0.3433189
    downsample_factor = 4.0  # for Series 8

    adjusted_coords = shift_annotation_to_downsampled(
        coords, offset_x_um, offset_y_um, px_size_x, px_size_y, downsample_factor
    )

    # --- Create mask ---
    mask = create_polygon_mask(image.shape, adjusted_coords)
    # mask = create_polygon_mask(image.shape, coords)

    # debugging
    print("First polygon point:", coords[0])
    print("First adjusted polygon point:", adjusted_coords[0])
    print("Image shape:", image.shape)


    # --- Overlay and display ---
    display_overlay(image, adjusted_coords, mask)
    # display_overlay(image, coords, mask)

    # --- Save mask image ---
    # cv2.imwrite("annotation_mask.png", mask)
    # print("Saved binary mask to annotation_mask.png")

    # --------------------------------------------



if __name__ == "__main__":
    main()

# ------------------------------------------------
# using polygon dragger

# import xml.etree.ElementTree as ET
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection
# import bioformats
# import javabridge

# def load_vsi_image(vsi_path, series=2):
#     javabridge.start_vm(class_path=bioformats.JARS)
#     try:
#         image = bioformats.load_image(vsi_path, series=series, rescale=False)
#         image_np = np.array(image)
#     finally:
#         javabridge.kill_vm()
#     return image_np

# def parse_annotation(annotation_path, scale_factor=1.0):
#     tree = ET.parse(annotation_path)
#     root = tree.getroot()
#     coords = []
#     for vertex in root.findall(".//Vertices/V"):
#         x = int(vertex.get("X")) / scale_factor
#         y = int(vertex.get("Y")) / scale_factor
#         coords.append([x, y])  # use list so it's mutable for dragging
#     return np.array(coords)

# class PolygonDragger:
#     def __init__(self, image, coords):
#         self.image = image
#         self.coords = coords
#         self.offset = np.array([0.0, 0.0])

#         self.fig, self.ax = plt.subplots(figsize=(15, 15))
#         self.ax.imshow(image)

#         self.poly = Polygon(coords, closed=True, edgecolor='red', facecolor='none', linewidth=2)
#         self.ax.add_patch(self.poly)

#         self.press = None
#         self.cid_press = self.poly.figure.canvas.mpl_connect('button_press_event', self.on_press)
#         self.cid_release = self.poly.figure.canvas.mpl_connect('button_release_event', self.on_release)
#         self.cid_motion = self.poly.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

#         plt.title("Drag the annotation to align. Close window when done.")
#         plt.axis('off')
#         plt.show()

#     def on_press(self, event):
#         if event.inaxes != self.ax:
#             return
#         if self.poly.contains_point((event.x, event.y)):
#             self.press = (event.xdata, event.ydata)

#     def on_motion(self, event):
#         if self.press is None or event.inaxes != self.ax:
#             return
#         dx = event.xdata - self.press[0]
#         dy = event.ydata - self.press[1]
#         self.offset += np.array([dx, dy])
#         self.coords += [dx, dy]
#         self.poly.set_xy(self.coords)
#         self.press = (event.xdata, event.ydata)
#         self.poly.figure.canvas.draw()

#     def on_release(self, event):
#         if self.press is None:
#             return
#         print(f"Final offset applied: x={self.offset[0]:.2f}, y={self.offset[1]:.2f}")
#         self.press = None

# def main():
#     # Set your paths
#     vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi"
#     annotation_path = r"C:\Users\Vivian\Downloads\FA 57B.annotations"
    
#     # Parameters
#     series = 7
#     scale_factor = 2  # Adjust if needed

#     print("Loading image...")
#     image = load_vsi_image(vsi_path, series=series)
#     print(f"Image shape: {image.shape}")

#     print("Loading annotation...")
#     coords = parse_annotation(annotation_path, scale_factor=scale_factor)

#     print("Launching UI — drag the annotation polygon to align it")
#     PolygonDragger(image, coords)

# if __name__ == "__main__":
#     main()

# ------------------------------------------------
# #  centering annotation + polygon dragger

# import xml.etree.ElementTree as ET
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# import cv2
# import javabridge
# import bioformats

# # ------------------------------
# # Annotation Parsing
# # ------------------------------
# def parse_annotation(annotation_path):
#     tree = ET.parse(annotation_path)
#     root = tree.getroot()
#     coords = []
#     for vertex in root.findall(".//Vertices/V"):
#         x = int(vertex.get("X"))
#         y = int(vertex.get("Y"))
#         coords.append([x, y])  # list for mutability
#     return np.array(coords)

# # ------------------------------
# # Mask Creation
# # ------------------------------
# def create_polygon_mask(image_shape, polygon_coords):
#     mask = np.zeros(image_shape[:2], dtype=np.uint8)
#     pts = np.array(polygon_coords, dtype=np.int32)
#     cv2.fillPoly(mask, [pts], color=255)
#     return mask

# # ------------------------------
# # Image Loading (VSI)
# # ------------------------------
# def load_vsi_image(vsi_path, series=0):
#     javabridge.start_vm(class_path=bioformats.JARS)
#     try:
#         image = bioformats.load_image(vsi_path, series=series, rescale=False)
#         image_np = np.array(image)
#     finally:
#         javabridge.kill_vm()
#     return image_np

# def center_polygon_on_image(image_shape, coords):
#     """
#     Shift polygon coords so that its centroid is aligned to image center.
#     """
#     image_h, image_w = image_shape[:2]
#     image_center = np.array([image_w / 2, image_h / 2])

#     polygon_centroid = np.mean(coords, axis=0)
#     shift = image_center - polygon_centroid
#     return coords + shift


# # ------------------------------
# # Display + Dragging Polygon
# # ------------------------------
# class PolygonDragger:
#     def __init__(self, image, coords):
#         self.image = image
#         self.coords = coords.astype(np.float32)
#         self.offset = np.array([0.0, 0.0])

#         self.fig, self.ax = plt.subplots(figsize=(15, 15))
#         self.ax.imshow(image)

#         self.poly = Polygon(self.coords, closed=True, edgecolor='red', facecolor='none', linewidth=2)
#         self.ax.add_patch(self.poly)

#         self.press = None
#         self.cid_press = self.poly.figure.canvas.mpl_connect('button_press_event', self.on_press)
#         self.cid_release = self.poly.figure.canvas.mpl_connect('button_release_event', self.on_release)
#         self.cid_motion = self.poly.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

#         plt.title("Drag the annotation to align. Close window when done.")
#         plt.axis('off')
#         plt.show()

#     def on_press(self, event):
#         if event.inaxes != self.ax:
#             return
#         if self.poly.contains_point((event.x, event.y)):
#             self.press = (event.xdata, event.ydata)

#     def on_motion(self, event):
#         if self.press is None or event.inaxes != self.ax:
#             return
#         dx = event.xdata - self.press[0]
#         dy = event.ydata - self.press[1]
#         self.offset += np.array([dx, dy])
#         self.coords += [dx, dy]
#         self.poly.set_xy(self.coords)
#         self.press = (event.xdata, event.ydata)
#         self.poly.figure.canvas.draw()

#     def on_release(self, event):
#         if self.press is None:
#             return
#         print(f"✅ Final offset applied: x={self.offset[0]:.2f}, y={self.offset[1]:.2f}")
#         self.press = None


# # ------------------------------
# # Main
# # ------------------------------
# def main():
#     # Paths
#     # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi"
#     # annotation_path = r"C:\Users\Vivian\Downloads\FA 57B.annotations"
#     vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 41 B.vsi"
#     annotation_path = r"C:\Users\Vivian\Downloads\PT 41 B.annotations"
#     series = 7

#     # Load image
#     print(f"Loading image from series {series}...")
#     image = load_vsi_image(vsi_path, series=series)
#     print(f"Loaded image shape: {image.shape}")

#     # Parse annotation (full-res coordinates)
#     coords = parse_annotation(annotation_path)
#     print(f"Parsed {len(coords)} annotation points.")

#     # Downsample for display if needed (e.g., if annotation is full-res and image is downsampled)
#     # downsample_factor = 2
#     # coords_display = coords / downsample_factor

#     # Downsample for display and center polygon
#     downsample_factor = 2
#     coords_display = coords / downsample_factor
#     coords_display = center_polygon_on_image(image.shape, coords_display)

#     # Interactive adjustment
#     print("Launching drag UI to adjust polygon...")
#     dragger = PolygonDragger(image, coords_display.copy())
#     adjusted_coords = dragger.coords

#     # Create mask from adjusted coordinates
#     mask = create_polygon_mask(image.shape, adjusted_coords)

#     # Final visualization
#     fig, ax = plt.subplots(figsize=(15, 15))
#     ax.imshow(image)
#     ax.plot(adjusted_coords[:, 0], adjusted_coords[:, 1], color='white', linewidth=2)
#     ax.imshow(mask, cmap='gray', alpha=0.3)
#     ax.axis("off")
#     plt.title("Final Adjusted Overlay")
#     plt.show()

#     # Optionally save mask
#     cv2.imwrite("PT41B_mask.png", mask)

# if __name__ == "__main__":
#     main()

# ------------------------------------------------
# automatically calculate scale factor based on image size and annotation coordinates
# import xml.etree.ElementTree as ET
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import bioformats
# import javabridge

# def load_vsi_image(vsi_path, series=7):
#     javabridge.start_vm(class_path=bioformats.JARS)
#     try:
#         image = bioformats.load_image(vsi_path, series=series, rescale=False)
#         image_np = np.array(image)
#     finally:
#         javabridge.kill_vm()
#     return image_np

# def parse_annotation(annotation_path):
#     tree = ET.parse(annotation_path)
#     root = tree.getroot()
#     coords = []
#     for vertex in root.findall(".//Vertices/V"):
#         x = int(vertex.get("X"))
#         y = int(vertex.get("Y"))
#         coords.append((x, y))
#     return coords

# def analyze_coordinates(coords, image_shape):
#     xs = [x for x, y in coords]
#     ys = [y for x, y in coords]
#     img_height, img_width = image_shape[:2]

#     print("--- Annotation Coordinate Range ---")
#     print(f"X: min={min(xs)}, max={max(xs)}, image width={img_width}")
#     print(f"Y: min={min(ys)}, max={max(ys)}, image height={img_height}")

#     scale_x = max(xs) / img_width
#     scale_y = max(ys) / img_height
#     print(f"Suggested X scale factor: {scale_x:.2f}")
#     print(f"Suggested Y scale factor: {scale_y:.2f}")

#     return scale_x, scale_y

# def main():
#     # Paths
#     vsi_path = r"Z:\\mirage\\med-i_data\\Data\\Amoon\\Pathology Raw\\FA scans\\FA 57B.vsi"
#     annotation_path = r"C:\\Users\\Vivian\\Downloads\\FA 57B.annotations"
#     series = 7  # Full resolution

#     print("Loading image...")
#     image = load_vsi_image(vsi_path, series=series)
#     print(f"Loaded image shape: {image.shape}")

#     print("Parsing annotation...")
#     coords = parse_annotation(annotation_path)
#     print(f"Parsed {len(coords)} annotation points.")

#     # Analyze coordinates and detect if scaling is necessary
#     analyze_coordinates(coords, image.shape)

# if __name__ == "__main__":
#     main()

# ------------------------------------------------