import numpy as np
import cv2


def plot_gradients(fig, ax, gx, gy=None, mask=None, mode="rgb", **kwargs):
    """
    Plot the gradients.

    :params fig: plt.figure; the figure to plot the gradients.
    :params ax: plt.axis; the axis to plot the gradients.
    :params gx: np.array (H, W) or dict; the x gradient, or dict with gradient data.
    :params gy: np.array (H, W) or None; the y gradient. If gx is dict, this is ignored.
    :params mask: np.array (H, W); the mask for gradients to be plotted
    :params mode: str {"rgb", "quiver"}; the mode to plot the gradients.
    """
    # Handle dict format (e.g., from result_dict['gradient'])
    if isinstance(gx, dict):
        gradient_data = gx
        if 'gradient' in gradient_data:
            grad = gradient_data['gradient']
            gx = grad[..., 0]
            gy = grad[..., 1]
        else:
            raise ValueError("Dict input must contain 'gradient' key")
    elif isinstance(gx, np.ndarray) and gx.ndim == 3 and gx.shape[-1] == 2:
        # Handle [H, W, 2] format directly
        gy = gx[..., 1]
        gx = gx[..., 0]
    elif gy is None:
        raise ValueError("gy must be provided if gx is not a dict or [H, W, 2] array")

    if mode == "rgb":
        # Plot the gradient in red and blue
        grad_range = kwargs.get("grad_range", 3.0)
        red = gx * 255 / grad_range + 127
        red = np.clip(red, 0, 255)
        blue = gy * 255 / grad_range + 127
        blue = np.clip(blue, 0, 255)
        image = np.stack((red, np.zeros_like(red), blue), axis=-1).astype(np.uint8)
        if mask is not None:
            image[np.logical_not(mask)] = np.array([127, 0, 127])
        ax.imshow(image)
    elif mode == "quiver":
        # Plot the gradient in quiver
        n_skip = kwargs.get("n_skip", 5)
        quiver_scale = kwargs.get("quiver_scale", 10.0)
        imgh, imgw = gx.shape
        X, Y = np.meshgrid(np.arange(imgw)[::n_skip], np.arange(imgh)[::n_skip])
        U = gx[::n_skip, ::n_skip] * quiver_scale
        V = -gy[::n_skip, ::n_skip] * quiver_scale
        if mask is None:
            mask = np.ones_like(gx, dtype=bool)
        else:
            mask = np.copy(mask).astype(bool)
        mask = mask[::n_skip, ::n_skip]
        ax.quiver(X[mask], Y[mask], U[mask], V[mask], units="xy", scale=1, color="red")
        ax.set_xlim(0, imgw)
        ax.set_ylim(imgh, 0)
    else:
        raise ValueError("Unknown plot gradient mode %s" % mode)


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


def visualize_force_field(normal, shear, overlay_image=None, alpha=0.6):
    """
    Visualize force field as RGB heatmap.

    Args:
        normal: np.array [H, W]; normal force component (Fz).
        shear: np.array [H, W, 2]; shear force components (Fx, Fy).
        overlay_image: np.array [H, W, 3] or None; optional background image to overlay on.
        alpha: float; transparency for overlay (0=fully transparent, 1=fully opaque).

    Returns
    -------
    np.array [H, W, 3] uint8; RGB visualization of force field.

    """
    force_viz = force_field_to_rgb(normal, shear)

    # Overlay on image if provided
    if overlay_image is not None:
        # Resize force field to match overlay image if needed
        if force_viz.shape[:2] != overlay_image.shape[:2]:
            force_viz = cv2.resize(force_viz, (overlay_image.shape[1], overlay_image.shape[0]))

        # Convert overlay to RGB if needed
        if len(overlay_image.shape) == 2:
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)

        # Blend images
        force_viz = cv2.addWeighted(overlay_image, 1 - alpha, force_viz, alpha, 0)

    return force_viz


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
