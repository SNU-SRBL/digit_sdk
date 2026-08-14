#!/usr/bin/env python3
"""
mm_to_ppmm.py.

This script measures the pixels-per-millimeter (ppmm) of a camera sensor by
capturing live video frames and allowing the user to click two points on the
image to measure the pixel distance.
The user must know the physical distance between these two points in millimeters.

Borrows from https://github.com/vocdex/digit-depth/blob/main/scripts/mm_to_pix.py

DO NOT PRESS THE GEL WITH THE SHARP POINTS OF THE CALIPER.
This will damage the sensor. Try to press the gel with the flat side of the caliper.

Usage:
    python mm_to_ppmm.py --serial SERIAL --distance_mm DISTANCE_MM
    [--frames N] [--sensors_root SENSORS_ROOT]
"""
import argparse
import os
import cv2
import numpy as np
from digit_sdk.camera import Camera
from digit_sdk.utils import load_config

DEFAULT_SENSORS_ROOT = os.path.join(os.path.dirname(__file__), "../sensors")


def main():
    parser = argparse.ArgumentParser(
        description="Measure pixels-per-millimeter (ppmm) interactively with live stream."
    )
    parser.add_argument(
        "--serial",
        type=str,
        required=True,
        help="sensor serial number (directory name under sensors/)",
    )
    parser.add_argument(
        "--distance_mm", type=float, required=True,
        help="Known physical distance between two clicks, in mm"
    )
    parser.add_argument(
        "--frames", type=int, default=5,
        help="Number of measurements to take (default: 5)"
    )
    parser.add_argument(
        "--sensors_root",
        type=str,
        default=DEFAULT_SENSORS_ROOT,
        help="root directory containing sensors/<serial>/",
    )
    args = parser.parse_args()

    # Load sensor configuration
    sensor_dir = os.path.join(args.sensors_root, args.serial)
    config_path = os.path.join(sensor_dir, f"{args.serial}.yaml")
    load_config(config_path=config_path)
    # Camera will use all config fields internally

    # Connect to the sensor
    device = Camera(serial=args.serial, sensors_root=args.sensors_root)
    device.connect()

    distances = []
    print("Streaming... Press 'c' to capture a frame measurement, 'q' to quit.")

    cv2.namedWindow("stream", cv2.WINDOW_NORMAL)
    while len(distances) < args.frames:
        frame = device.get_image()
        cv2.imshow("stream", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            pts = []
            display = frame.copy()

            def on_click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
                    pts.append((x, y))
                    cv2.circle(display, (x, y), 3, (0, 255, 0), -1)
                    cv2.imshow("capture", display)

            cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
            cv2.imshow("capture", display)
            cv2.setMouseCallback("capture", on_click)
            # Wait for two clicks or cancel
            while len(pts) < 2:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow("capture")
            if len(pts) < 2:
                print("Capture cancelled, continue streaming.")
                continue

            # Compute and store pixel distance
            d_px = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
            distances.append(d_px)
            print(f"Captured {len(distances)}/{args.frames}: {d_px:.2f} px")
        elif key == ord('q'):
            print("Quitting.")
            break

    # Cleanup
    device.release()
    cv2.destroyAllWindows()

    if len(distances) == 0:
        print("No measurements taken.")
        return

    # Compute average and ppmm
    avg_px = float(np.mean(distances))
    ppmm = avg_px / args.distance_mm
    print(f"\nAvg pixel distance: {avg_px:.2f} px")
    print(f"Computed ppmm: {ppmm:.3f} pixels/mm")


if __name__ == "__main__":
    main()
