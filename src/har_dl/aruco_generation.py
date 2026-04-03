import cv2
import cv2.aruco as aruco
from src.har_dl.definitions import get_project_root
import numpy as np


def generate_aruco(marker_id=23, marker_size_cm=15, dpi=300):
    marker_size_pixels = int((marker_size_cm / 2.54) * dpi)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)

    a4_width_pixels = int((21.0 / 2.54) * dpi)
    a4_height_pixels = int((29.7 / 2.54) * dpi)

    a4_canvas = np.ones((a4_height_pixels, a4_width_pixels), dtype=np.uint8) * 255

    y_offset = (a4_height_pixels - marker_size_pixels) // 2
    x_offset = (a4_width_pixels - marker_size_pixels) // 2

    a4_canvas[y_offset:y_offset + marker_size_pixels, x_offset:x_offset + marker_size_pixels] = marker_image

    artifacts_path = get_project_root() / "artifacts"
    artifacts_path.mkdir(exist_ok=True)
    file_path = artifacts_path / f"aruco_{marker_id}_15cm_300dpi.png"
    cv2.imwrite(str(file_path), a4_canvas)

    print(f"Aruco marker saved at {file_path}")
    print(f"Print at 300 DPI for exact 15cm x 15cm size")


if __name__ == '__main__':
    generate_aruco()