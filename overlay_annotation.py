"""
This script overlays the ground truth annotation on a downsampled whole-slide image (WSI).
"""
# # ------------------------------------------------

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
#         x = int(vertex.get("X")) // scale_factor
#         y = int(vertex.get("Y")) // scale_factor
#         coords.append((x, y))
    
#     return coords

# def overlay_annotation(image, coords):
#     """
#     Overlay annotation coordinates on the image using matplotlib.
#     """
#     xs, ys = zip(*coords)

#     plt.figure(figsize=(15, 15))
#     plt.imshow(image)
#     plt.plot(xs, ys, 'r-', linewidth=2, label="Annotation Polygon")
#     plt.scatter(xs, ys, color='blue', s=5)
#     plt.title("Annotation Overlay on Downsampled WSI")
#     plt.axis("off")
#     plt.legend()
#     plt.show()

# def main():
#     # Paths
#     vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi"
#     annotation_path = r"C:\Users\Vivian\Downloads\FA 57B.annotations"  # <-- adjust as needed

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

#     # Show overlay
#     print("Displaying overlay...")
#     overlay_annotation(image, coords)

# if __name__ == "__main__":
#     main()

# ------------------------------------------------

#  almost works --> slightly offcentered
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import bioformats
import javabridge

def load_vsi_image(vsi_path, series=2):  # Downsampled level (e.g., level 2 = 4× down)
    """
    Load a VSI image using Bio-Formats and convert it to a NumPy array.
    """
    javabridge.start_vm(class_path=bioformats.JARS)

    try:
        image = bioformats.load_image(vsi_path, series=series, rescale=False)
        image_np = np.array(image)
    finally:
        javabridge.kill_vm()

    return image_np

def parse_annotation(annotation_path, scale_factor=1.0):
    """
    Parse the .annotations XML file and extract scaled polygon coordinates.
    """
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    coords = []
    for vertex in root.findall(".//Vertices/V"):
        x = int(vertex.get("X")) / scale_factor
        y = int(vertex.get("Y")) / scale_factor
        coords.append((x, y))
    
    return coords

def overlay_annotation(image, coords):
    """
    Overlay annotation coordinates on the image using matplotlib.
    """
    xs, ys = zip(*coords)

    plt.figure(figsize=(15, 15))
    # need another type of image conversion
    # try in lower dimension level
    
    plt.imshow(image)
    plt.plot(xs, ys, 'r-', linewidth=2, label="Annotation Polygon")
    # plt.scatter(xs, ys, color='blue', s=5)
    # plt.title("Annotation Overlay on Downsampled WSI")
    plt.axis("off")
    # plt.legend()
    plt.show()

# adding function to calculate polygon centroid and adjust offset
def polygon_centroid(coords):
    """
    Compute the accurate centroid of a 2D polygon using the area-weighted formula.
    Assumes the polygon is closed (first and last points are not the same).
    """
    coords = np.array(coords)
    x = coords[:, 0]
    y = coords[:, 1]

    # Shift the points so (x[n+1], y[n+1]) = (x[0], y[0])
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    a = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * np.sum(a)

    Cx = (1 / (6 * area)) * np.sum((x[:-1] + x[1:]) * a)
    Cy = (1 / (6 * area)) * np.sum((y[:-1] + y[1:]) * a)

    return Cx, Cy


def adjust_annotation_offset(image, coords):
    """
    Adjust annotation coordinates to properly align with the WSI image.
    If annotations are centered incorrectly, this will shift them accordingly.
    """
    image_height, image_width = image.shape[:2]
    annotation_height = max(y for _, y in coords) - min(y for _, y in coords)
    annotation_width = max(x for x, _ in coords) - min(x for x, _ in coords)
    
    # Calculate the difference in center position (centroid)
    image_center = (image_width / 2, image_height / 2)
    # annotation_center = (sum(x for x, _ in coords) / len(coords), sum(y for _, y in coords) / len(coords))

    # using the polygon centroid function***
    annotation_center = annotation_center = polygon_centroid(coords)

    # Calculate offset
    offset_x = image_center[0] - annotation_center[0]
    offset_y = image_center[1] - annotation_center[1]

    # Apply the offset
    adjusted_coords = [(x + offset_x, y + offset_y) for x, y in coords]
    return adjusted_coords

def main():
    # Paths
    vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi"
    annotation_path = r"C:\Users\Vivian\Downloads\FA 57B.annotations"  # <-- adjust as needed
    # vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 41 B.vsi"
    # annotation_path = r"C:\Users\Vivian\Downloads\PT 41 B.annotations"

    # Parameters
    series = 7  # Downsampled level (e.g., level 2 for 4× downsampling)
    scale_factor = 2  # Scale down annotations to match series 2 resolution

    # Load image
    print(f"Loading VSI image from series {series}...")
    image = load_vsi_image(vsi_path, series=series)
    print(f"Loaded image with shape: {image.shape}")

    # Load and scale annotation
    print("Parsing and scaling annotation...")
    coords = parse_annotation(annotation_path, scale_factor=scale_factor)
    print(f"Loaded {len(coords)} annotation points (scaled).")

    # Adjust the annotation offset if misaligned
    print("Adjusting annotation offset...")
    adjusted_coords = adjust_annotation_offset(image, coords)
    print(f"Adjusted {len(adjusted_coords)} annotation points.")

    # Show overlay
    print("Displaying overlay...")
    overlay_annotation(image, adjusted_coords)

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
