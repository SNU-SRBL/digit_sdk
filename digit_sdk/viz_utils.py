import numpy as np
import cv2
def force_field_to_rgb(normal, shear):
    """
    Convert force field arrays to RGB image using (R,G,B)=(Fx,Fy,Fz).

    Args:
        normal: np.array [H, W]; normal force component (Fz), expected in [0, 1].
        shear: np.array [H, W, 2]; shear components (Fx, Fy), expected in [-1, 1].

    Returns
    -------
    np.array [H, W, 3] uint8 RGB image.

    """
    normal_arr = np.asarray(normal, dtype=np.float32)
    shear_arr = np.asarray(shear, dtype=np.float32)

    normal_norm = np.clip(normal_arr, 0.0, 1.0)
    shear_x_norm = np.clip((shear_arr[..., 0] + 1.0) / 2.0, 0.0, 1.0)
    shear_y_norm = np.clip((shear_arr[..., 1] + 1.0) / 2.0, 0.0, 1.0)

    red = (shear_x_norm * 255.0).astype(np.uint8)
    green = (shear_y_norm * 255.0).astype(np.uint8)
    blue = (normal_norm * 255.0).astype(np.uint8)

    return np.stack([red, green, blue], axis=-1)


def visualize_force_vector(
        fx, fy, fz, image, arrow_scale=50.0, arrow_color=(0, 255, 0),
        arrow_thickness=2, show_magnitude=True):
    """
    Visualize force vector as arrow overlay on image.

    Args:
        fx: float; horizontal shear force component.
        fy: float; vertical shear force component.
        fz: float; normal force component.
        image: np.array [H, W, 3] or [H, W]; background image.
        arrow_scale: float; scaling factor for arrow length.
        arrow_color: tuple (B, G, R); color for the arrow.
        arrow_thickness: int; thickness of the arrow line.
        show_magnitude: bool; whether to show magnitude text.

    Returns
    -------
    np.array [H, W, 3] uint8; image with force vector overlay.

    """
    # Copy image to avoid modifying original
    if len(image.shape) == 2:
        viz_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        viz_image = image.copy()

    h, w = viz_image.shape[:2]
    center = (w // 2, h // 2)

    # Calculate arrow endpoint
    # Note: OpenCV coordinates are (x, y) where y increases downward
    arrow_end_x = int(center[0] + fx * arrow_scale)
    arrow_end_y = int(center[1] + fy * arrow_scale)  # fy positive = downward
    arrow_end = (arrow_end_x, arrow_end_y)

    # Draw arrow for in-plane forces (fx, fy)
    if abs(fx) > 0.01 or abs(fy) > 0.01:  # Only draw if significant
        cv2.arrowedLine(
            viz_image, center, arrow_end, arrow_color,
            arrow_thickness, tipLength=0.3,
        )

    # Draw circle for normal force (fz) - size proportional to magnitude
    normal_radius = int(abs(fz) * arrow_scale)
    if normal_radius > 5:
        # Red for positive, blue for negative
        normal_color = (0, 0, 255) if fz > 0 else (255, 0, 0)
        cv2.circle(viz_image, center, normal_radius, normal_color, 2)

    # Add text showing force magnitudes
    if show_magnitude:
        magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
        text_lines = [
            f"Fx: {fx:+.3f}",
            f"Fy: {fy:+.3f}",
            f"Fz: {fz:+.3f}",
            f"|F|: {magnitude:.3f}"
        ]

        y_offset = 30
        for i, line in enumerate(text_lines):
            cv2.putText(
                viz_image, line, (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            cv2.putText(
                viz_image, line, (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
            )

    return viz_image
