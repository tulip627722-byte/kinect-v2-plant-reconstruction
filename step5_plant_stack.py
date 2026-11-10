"""
Step 5: 植物纯旋转堆叠验证（不做 ICP）

仅用 rot_around_center(i*10°) 将 36 帧植物点云旋转到世界坐标系。
不做 ICP、不做 TSDF，纯几何验证旋转中心是否正确。

目的：在 Step6 ICP 之前确认旋转中心对植物也适用。
- 如果堆叠后呈紧凑球状 → 中心正确，进入 Step6
- 如果炸散（跨度>0.5m）→ 旋转中心需重新估计
"""
import numpy as np
import open3d as o3d
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import (
    load_depth, load_color_rgb, bilateral_filter_depth,
    rotate_points_around_center, rot_around_center,
    visualize_point_clouds, make_colored_pcd, save_ply
)
from step1_align import build_undistort_maps, build_alignment_lut, undistort_depth, compute_alignment_with_depth
from step3_segment import segment_frame


def main():
    print("=" * 60)
    print("STEP 5: Plant pure rotation stacking verification")
    print("=" * 60)
    print("NO ICP. Only rotation. Verifying center before ICP.")
    print("=" * 60)

    # Build alignment maps
    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    all_plant_world = []
    frame_counts = []

    for i in range(NUM_FRAMES):
        # Load and preprocess
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(depth_undist_m, lut_u, lut_v, color_rgb)
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)
        depth_filtered_m = depth_filtered_mm / 1000.0

        # Segment plant
        _, plant_pcd = segment_frame(depth_filtered_m, aligned_color)
        plant_pts = np.asarray(plant_pcd.points)

        if len(plant_pts) < 10:
            print(f"  Frame {i:03d}: no plant pts, skip")
            frame_counts.append(0)
            continue

        # 仅做旋转，不做 ICP！
    # 如果旋转中心正确，36 帧植物点云应叠成一个紧凑区域
    # 如果炸散说明旋转中心对植物不适用
    # Rotate to world frame (NO ICP!)
        angle = i * DEG_PER_FRAME
        plant_world = rotate_points_around_center(plant_pts, angle)

        all_plant_world.append(plant_world)
        frame_counts.append(len(plant_world))

        if i % 6 == 0:
            print(f"  Frame {i:03d}: {len(plant_world)} plant pts → world @ {angle:.0f}°")

    # Merge all
    merged = np.vstack(all_plant_world)
    print(f"\nTotal plant points (world): {len(merged)}")
    print(f"  X range: [{merged[:,0].min():.3f}, {merged[:,0].max():.3f}]")
    print(f"  Y range: [{merged[:,1].min():.3f}, {merged[:,1].max():.3f}]")
    print(f"  Z range: [{merged[:,2].min():.3f}, {merged[:,2].max():.3f}]")

    # 检测是否炸散：X 或 Z 方向跨度 > 0.5m 判断为炸散
    # Check for explosion
    x_span = merged[:, 0].max() - merged[:, 0].min()
    z_span = merged[:, 2].max() - merged[:, 2].min()
    is_exploded = (x_span > 0.5 or z_span > 0.5)
    # 典型盆栽植物冠幅约 0.25m，若 >0.5m 判定为炸散

    # Save merged
    out_dir = os.path.join(OUTPUT_DIR, "plant_stack")
    os.makedirs(out_dir, exist_ok=True)
    save_ply(os.path.join(out_dir, "plant_stack_nocp.ply"), merged)

    # Visualize
    print("\nVerification: pure rotation stack — close window to continue")
    if is_exploded:
        print("  [WARNING] WARNING: Point cloud appears exploded!")
        print("  Rotation center may not be valid for plant.")
        print("  Need to re-estimate center from plant centroids.")
    else:
        print("  [OK] Stack looks coherent. Proceed to Step 6.")

    pcd = make_colored_pcd(merged, [0.2, 0.7, 0.3])  # Green
    visualize_point_clouds(
        [pcd],
        window_name="Step 5 — Plant Pure Rotation Stack (close to continue)")

    print(f"\nStep 5 complete.")
    print(f"  Merged: {len(merged)} pts → pipeline_output/plant_stack/plant_stack_nocp.ply")
    print(f"  Exploded: {is_exploded}")

    if is_exploded:
        print("\n  [WARNING] DO NOT proceed to Step 6. Re-estimate center first.")


if __name__ == "__main__":
    main()
