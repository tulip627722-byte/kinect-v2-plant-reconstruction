"""
Step 3: 逐帧花盆/植物分割（支持人工标注掩码）

分割优先级：
1. 如果存在人工标注掩码 → 先作为空间 ROI 约束
2. 在 ROI 内用 HSV 颜色 refine
3. 无标注掩码 → 纯 HSV 颜色分割

花盆：非绿色 + Y 范围 [0.08, 0.20]m
植物：绿色（HSV H=[35°,85°], S>30） + Y<0.10m

掩码文件：pipeline_output/annotations/pot_mask_XXX.png / plant_mask_XXX.png
"""
import numpy as np
import cv2
import open3d as o3d
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import (
    load_depth, load_color_rgb, depth_to_points,
    bilateral_filter_depth, save_ply,
    is_green_hsv, sor_filter, voxel_downsample, estimate_normals,
    visualize_point_clouds, make_colored_pcd
)
from step1_align import build_undistort_maps, build_alignment_lut, undistort_depth, compute_alignment_with_depth


def load_annotations():
    """Load all annotation masks from disk.

    Returns:
        pot_masks: dict {frame_idx: (424,512) uint8 mask} or empty dict
        plant_masks: dict {frame_idx: (424,512) uint8 mask} or empty dict
    """
    pot_masks = {}
    plant_masks = {}
    anno_dir = os.path.join(OUTPUT_DIR, "annotations")

    if not os.path.exists(anno_dir):
        return pot_masks, plant_masks

    for f in sorted(glob.glob(os.path.join(anno_dir, "pot_mask_*.png"))):
        basename = os.path.basename(f)
        fidx = int(basename.replace("pot_mask_", "").replace(".png", ""))
        raw = np.fromfile(f, dtype=np.uint8)
        mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            pot_masks[fidx] = mask

    for f in sorted(glob.glob(os.path.join(anno_dir, "plant_mask_*.png"))):
        basename = os.path.basename(f)
        fidx = int(basename.replace("plant_mask_", "").replace(".png", ""))
        raw = np.fromfile(f, dtype=np.uint8)
        mask = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            plant_masks[fidx] = mask

    return pot_masks, plant_masks


def segment_frame(depth_filtered_m, aligned_color, pot_mask=None, plant_mask=None):
    """
    Segment one frame into pot and plant point clouds.

    Args:
        depth_filtered_m: (424, 512) float, depth in meters
        aligned_color: (424, 512, 3) uint8 RGB
        pot_mask: (424, 512) uint8, 255 = pot region, optional
        plant_mask: (424, 512) uint8, 255 = plant region, optional

    Returns:
        pot_pcd, plant_pcd
    """
    h, w = depth_filtered_m.shape

    # Build full point cloud
    vv, uu = np.mgrid[0:h, 0:w]
    valid = (depth_filtered_m > DEPTH_MIN_M) & (depth_filtered_m < DEPTH_MAX_M)
    z = depth_filtered_m[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)

    x = (u - DEPTH_CX) * z / DEPTH_FX
    y = (v - DEPTH_CY) * z / DEPTH_FY
    pts = np.stack([x, y, z], axis=1).astype(np.float64)

    clr_uint8 = aligned_color[valid]
    clr_rgb = clr_uint8.astype(np.float64) / 255.0

    # HSV 绿色检测：将 RGB 转为 HSV 后判断是否在绿色阈值区间
    # H∈[35°,85°]（OpenCV 半范围 17~42），S>30% 排除白色/灰色区域
    clr_bgr_uint8 = cv2.cvtColor(clr_uint8.reshape(1, -1, 3), cv2.COLOR_RGB2BGR)
    clr_hsv = cv2.cvtColor(clr_bgr_uint8, cv2.COLOR_BGR2HSV)
    clr_hsv = clr_hsv.reshape(-1, 3)
    green_mask = is_green_hsv(clr_hsv)

    # 空间掩码：如果有标注则用标注 ROI，否则全 1（不做空间约束）
    spatial_pot = np.ones(valid.sum(), dtype=bool)
    spatial_plant = np.ones(valid.sum(), dtype=bool)

    if pot_mask is not None:
        spatial_pot = pot_mask[valid] > 0
    if plant_mask is not None:
        spatial_plant = plant_mask[valid] > 0

    # ============================================================
    # 花盆分割：非绿色 + Y 在花盆高度范围 + 可选空间 ROI
    # 花盆是白色的，HSV 中饱和度低，通过排除绿色 + Y 范围定位
    # ============================================================
    # With annotation: spatial ROI + NOT green
    # Without annotation: NOT green + Y + depth (as before)
    if pot_mask is not None:
        pot_mask_final = (
            spatial_pot &
            (~green_mask) &
            (y >= POT_Y_MIN) & (y <= POT_Y_MAX)
        )
    else:
        pot_mask_final = (
            (~green_mask) &
            (y >= POT_Y_MIN) & (y <= POT_Y_MAX)
        )

    pot_pts = pts[pot_mask_final]
    pot_clr = clr_rgb[pot_mask_final]

    pot_pcd = o3d.geometry.PointCloud()
    if len(pot_pts) > 0:
        pot_pcd.points = o3d.utility.Vector3dVector(pot_pts)
        pot_pcd.colors = o3d.utility.Vector3dVector(pot_clr)
        pot_pcd = voxel_downsample(pot_pcd)
        pot_pcd = sor_filter(pot_pcd, POT_SOR_NB, POT_SOR_STD)
        pot_pcd = estimate_normals(pot_pcd)

    # ============================================================
    # 植物分割：HSV 绿色 + Y 在叶片范围 + 可选空间 ROI
    # 植物叶片的主要特征是高饱和度绿色
    # ============================================================
    # With annotation: spatial ROI + IS green
    # Without annotation: IS green + Y (as before)
    if plant_mask is not None:
        plant_mask_final = (
            spatial_plant &
            green_mask &
            (y < PLANT_Y_MAX)
        )
    else:
        plant_mask_final = (
            green_mask &
            (y < PLANT_Y_MAX)
        )

    plant_pts = pts[plant_mask_final]
    plant_clr = clr_rgb[plant_mask_final]

    plant_pcd = o3d.geometry.PointCloud()
    if len(plant_pts) > 0:
        plant_pcd.points = o3d.utility.Vector3dVector(plant_pts)
        plant_pcd.colors = o3d.utility.Vector3dVector(plant_clr)
        plant_pcd = voxel_downsample(plant_pcd)
        plant_pcd = sor_filter(plant_pcd, PLANT_SOR_NB, PLANT_SOR_STD)
        plant_pcd = estimate_normals(plant_pcd)

    return pot_pcd, plant_pcd


def process_all_frames():
    """Segment all 36 frames, using annotations if available."""
    print("Building undistortion maps and alignment LUT...")
    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    # Load annotations
    pot_masks, plant_masks = load_annotations()
    if pot_masks:
        print(f"Loaded pot masks for {len(pot_masks)} frames: {sorted(pot_masks.keys())}")
    else:
        print("No pot masks found — using HSV-only segmentation")
    if plant_masks:
        print(f"Loaded plant masks for {len(plant_masks)} frames: {sorted(plant_masks.keys())}")
    else:
        print("No plant masks found — using HSV-only segmentation")

    pot_pcds = []
    plant_pcds = []

    for i in range(NUM_FRAMES):
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)

        depth_undist_mm = undistort_depth(depth_mm, map1, map2)
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(
            depth_undist_m, lut_u, lut_v, color_rgb)

        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)
        depth_filtered_m = depth_filtered_mm / 1000.0

        # Use frame-specific mask or propagate frame 0 mask
        pot_m = pot_masks.get(i, pot_masks.get(0))
        plant_m = plant_masks.get(i, plant_masks.get(0))

        pot_pcd, plant_pcd = segment_frame(
            depth_filtered_m, aligned_color,
            pot_mask=pot_m, plant_mask=plant_m)
        pot_pcds.append(pot_pcd)
        plant_pcds.append(plant_pcd)

        if i % 6 == 0:
            mask_info = ""
            if pot_m is not None:
                mask_info += " [masked]"
            print(f"  Frame {i:03d}: pot={len(pot_pcd.points)} pts, "
                  f"plant={len(plant_pcd.points)} pts{mask_info}")

    # Save
    out_dir = os.path.join(OUTPUT_DIR, "segmented")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(NUM_FRAMES):
        o3d.io.write_point_cloud(
            os.path.join(out_dir, f"pot_{i:03d}.ply"), pot_pcds[i])
        o3d.io.write_point_cloud(
            os.path.join(out_dir, f"plant_{i:03d}.ply"), plant_pcds[i])

    print(f"\nSaved segmented point clouds to {out_dir}")
    return pot_pcds, plant_pcds


def main():
    print("=" * 60)
    print("STEP 3: Pot / Plant segmentation")
    print("=" * 60)

    pot_pcds, plant_pcds = process_all_frames()

    # Verify
    print("\nVerification: frame 0 segmentation")
    print("  BLUE = Pot    GREEN = Plant")

    pot_pts = np.asarray(pot_pcds[0].points)
    plant_pts = np.asarray(plant_pcds[0].points)

    pot_viz = make_colored_pcd(pot_pts, [0.3, 0.4, 0.9])
    plant_viz = make_colored_pcd(plant_pts, [0.2, 0.8, 0.3])

    visualize_point_clouds(
        [pot_viz, plant_viz],
        window_name="Step 3 — Pot (Blue) + Plant (Green) — Frame 0 (close to confirm)")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()
