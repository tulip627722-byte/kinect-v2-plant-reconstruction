"""
Manual annotation tool for pot and plant masks on aligned 512x424 images.

Usage:
  - LEFT CLICK:  add vertex to current polygon
  - 'p': finalize POT polygon (blue)
  - 'l': finalize PLANT polygon (green)
  - 'r': reset current polygon
  - 'R': reset ALL polygons
  - 'c': confirm and save
  - 'q': quit without saving
  - 'n': next frame (after confirming)

Workflow:
  1. Draw pot polygon → press 'p' to confirm
  2. Draw plant polygon → press 'l' to confirm
  3. Press 'c' to save masks → then 'n' for next frame
"""
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import load_depth, load_color_rgb
from step1_align import (
    build_undistort_maps, build_alignment_lut,
    undistort_depth, compute_alignment_with_depth
)

# Global state
pot_polygon = []      # list of (u, v) tuples for pot
plant_polygon = []    # list of (u, v) tuples for plant
current_polygon = []  # points being drawn now
pot_masks = {}        # frame_idx → binary mask (424, 512)
plant_masks = {}      # frame_idx → binary mask (424, 512)
current_frame = 0
display_img = None
original_img = None
window_name = "Annotate — Pot(blue) Plant(green)"


def draw_all():
    """Redraw the display image with all annotations."""
    global display_img, pot_polygon, plant_polygon, current_polygon
    display_img = original_img.copy()

    # Draw pot polygons (blue)
    for poly in ([pot_polygon] if pot_polygon else []):
        if len(poly) >= 2:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, (255, 0, 0), 2)
            cv2.fillPoly(display_img, [pts], (255, 0, 0, 60))  # semi-transparent fill

    # Draw plant polygons (green)
    for poly in ([plant_polygon] if plant_polygon else []):
        if len(poly) >= 2:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)
            cv2.fillPoly(display_img, [pts], (0, 255, 0, 60))

    # Draw current polygon being edited (yellow)
    if len(current_polygon) >= 1:
        pts = np.array(current_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(display_img, [pts], False, (0, 255, 255), 2)
        for pt in current_polygon:
            cv2.circle(display_img, pt, 4, (0, 255, 255), -1)

    # Status text
    cv2.putText(display_img, f"Frame {current_frame:03d}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_img, "p=pot | l=plant | r=reset | c=save | n=next | q=quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow(window_name, display_img)


def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for polygon drawing."""
    global current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point to current polygon
        current_polygon.append((x, y))
        draw_all()


def polygons_to_masks():
    """Convert pot and plant polygons to binary masks.
    Polygon points are in DISPLAY coords (1024x848) → scale to (512x424).
    """
    h, w = DEPTH_HEIGHT, DEPTH_WIDTH  # 424, 512
    scale_x = w / 1024.0  # 512 / 1024 = 0.5
    scale_y = h / 848.0   # 424 / 848 = 0.5

    pot_mask = np.zeros((h, w), dtype=np.uint8)
    if len(pot_polygon) >= 3:
        pts = np.array(pot_polygon, np.float32)
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(pot_mask, [pts], 255)

    plant_mask = np.zeros((h, w), dtype=np.uint8)
    if len(plant_polygon) >= 3:
        pts = np.array(plant_polygon, np.float32)
        pts[:, 0] *= scale_x
        pts[:, 1] *= scale_y
        pts = pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(plant_mask, [pts], 255)

    return pot_mask, plant_mask


def save_masks():
    """Save all annotated masks to disk."""
    out_dir = os.path.join(OUTPUT_DIR, "annotations")
    os.makedirs(out_dir, exist_ok=True)

    # Collect all annotated frame indices
    all_frames = set(pot_masks.keys()) | set(plant_masks.keys())

    for fidx in sorted(all_frames):
        if fidx in pot_masks and pot_masks[fidx] is not None:
            ok, buf = cv2.imencode('.png', pot_masks[fidx])
            if ok:
                buf.tofile(os.path.join(out_dir, f"pot_mask_{fidx:03d}.png"))
        if fidx in plant_masks and plant_masks[fidx] is not None:
            ok, buf = cv2.imencode('.png', plant_masks[fidx])
            if ok:
                buf.tofile(os.path.join(out_dir, f"plant_mask_{fidx:03d}.png"))

    print(f"\nSaved masks for {len(all_frames)} frames → {out_dir}")


def annotation_loop(map1, map2, lut_u, lut_v):
    """Main annotation loop — annotate any frames the user wants."""
    global pot_polygon, plant_polygon, current_polygon, current_frame
    global display_img, original_img, pot_masks, plant_masks

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 848)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "=" * 60)
    print("ANNOTATION TOOL")
    print("=" * 60)
    print("LEFT CLICK = add vertex   p = finalize POT (blue)")
    print("l = finalize PLANT (green)  r = reset current polygon")
    print("R = reset ALL    c = save masks    q = quit")
    print("=" * 60)

    # Start with frame 0
    current_frame = 0
    load_frame(current_frame, map1, map2, lut_u, lut_v)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('p'):
            # Finalize pot polygon
            if len(current_polygon) >= 3:
                pot_polygon = current_polygon.copy()
                print(f"  Pot: {len(pot_polygon)} vertices")
            else:
                print("  Need >= 3 vertices for polygon")
            current_polygon = []
            draw_all()

        elif key == ord('l'):
            # Finalize plant polygon
            if len(current_polygon) >= 3:
                plant_polygon = current_polygon.copy()
                print(f"  Plant: {len(plant_polygon)} vertices")
            else:
                print("  Need >= 3 vertices for polygon")
            current_polygon = []
            draw_all()

        elif key == ord('r'):
            # Reset current polygon
            current_polygon = []
            draw_all()

        elif key == ord('R'):
            # Reset ALL
            pot_polygon = []
            plant_polygon = []
            current_polygon = []
            draw_all()

        elif key == ord('c'):
            # Save current frame's masks to memory AND disk immediately
            pot_m, plant_m = polygons_to_masks()
            pot_masks[current_frame] = pot_m
            plant_masks[current_frame] = plant_m

            out_dir = os.path.join(OUTPUT_DIR, "annotations")
            os.makedirs(out_dir, exist_ok=True)
            if pot_m is not None:
                # Use imencode + tofile to avoid Chinese path issues
                ok, buf = cv2.imencode('.png', pot_m)
                if ok:
                    buf.tofile(os.path.join(out_dir, f"pot_mask_{current_frame:03d}.png"))
            if plant_m is not None:
                ok, buf = cv2.imencode('.png', plant_m)
                if ok:
                    buf.tofile(os.path.join(out_dir, f"plant_mask_{current_frame:03d}.png"))

            print(f"  Saved masks for frame {current_frame:03d} → disk")

        elif key == ord('n'):
            # Next frame
            current_frame = (current_frame + 1) % NUM_FRAMES
            pot_polygon = []
            plant_polygon = []
            current_polygon = []
            load_frame(current_frame, map1, map2, lut_u, lut_v)
            print(f"\n  Frame {current_frame:03d}")

        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    save_masks()


def load_frame(fidx, map1, map2, lut_u, lut_v):
    """Load and prepare a frame for annotation."""
    global original_img, display_img, pot_polygon, plant_polygon, current_polygon

    depth_mm = load_depth(fidx)
    color_rgb = load_color_rgb(fidx)
    depth_undist_mm = undistort_depth(depth_mm, map1, map2)
    depth_undist_m = depth_undist_mm / 1000.0
    aligned_color = compute_alignment_with_depth(depth_undist_m, lut_u, lut_v, color_rgb)

    # Resize for display (512x424 → 1024x848 for easier annotation)
    original_img = cv2.resize(aligned_color, (1024, 848),
                              interpolation=cv2.INTER_NEAREST)

    # Try to load masks: memory first, then disk
    pot_polygon = []
    plant_polygon = []

    # From memory
    if fidx in pot_masks and pot_masks[fidx] is not None:
        pot_polygon = masks_to_polygon(pot_masks[fidx])
    if fidx in plant_masks and plant_masks[fidx] is not None:
        plant_polygon = masks_to_polygon(plant_masks[fidx])

    # From disk (if not in memory)
    anno_dir = os.path.join(OUTPUT_DIR, "annotations")
    if not pot_polygon and os.path.exists(anno_dir):
        pot_path = os.path.join(anno_dir, f"pot_mask_{fidx:03d}.png")
        if os.path.exists(pot_path):
            raw = np.fromfile(pot_path, dtype=np.uint8)
            mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                pot_masks[fidx] = mask
                pot_polygon = masks_to_polygon(mask)
    if not plant_polygon and os.path.exists(anno_dir):
        plant_path = os.path.join(anno_dir, f"plant_mask_{fidx:03d}.png")
        if os.path.exists(plant_path):
            raw = np.fromfile(plant_path, dtype=np.uint8)
            mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                plant_masks[fidx] = mask
                plant_polygon = masks_to_polygon(mask)

    if pot_polygon:
        print(f"  Loaded pot mask: {len(pot_polygon)} vertices")
    if plant_polygon:
        print(f"  Loaded plant mask: {len(plant_polygon)} vertices")

    current_polygon = []
    draw_all()


def masks_to_polygon(mask_424_512):
    """Convert a binary mask back to polygon points (for display at 1024x848)."""
    # Find contour of the mask
    contours, _ = cv2.findContours(mask_424_512, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)
    # Scale from 512x424 to 1024x848
    pts = largest.reshape(-1, 2).astype(np.float32)
    pts[:, 0] *= 2  # 1024/512
    pts[:, 1] *= 2  # 848/424
    return [(int(p[0]), int(p[1])) for p in pts]


def main():
    print("=" * 60)
    print("MANUAL ANNOTATION: Pot & Plant Masks")
    print("=" * 60)
    print("Building alignment maps...")

    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    annotation_loop(map1, map2, lut_u, lut_v)


if __name__ == "__main__":
    main()
