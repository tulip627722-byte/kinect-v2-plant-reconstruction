"""
Step 2: 逐帧三维彩色点云生成

对 36 帧逐帧处理：
1. 加载深度图 + 彩色图
2. 深度图畸变校正
3. 彩色图对齐到深度图（外参变换）
4. 双边滤波平滑深度
5. 深度范围滤波 (0.3m~1.5m)
6. 反投影到三维点云（深度相机内参）

输出：36 帧彩色点云 → pipeline_output/frames/
"""
import numpy as np
import cv2
import open3d as o3d
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import (
    load_depth, load_color_rgb, depth_to_points, save_ply,
    bilateral_filter_depth, visualize_point_clouds, make_colored_pcd
)
from step1_align import build_undistort_maps, build_alignment_lut, undistort_depth, compute_alignment_with_depth


def process_all_frames():
    """Process all 36 frames: align → filter → unproject → save."""
    out_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(out_dir, exist_ok=True)

    # 预处理畸变校正映射和颜色对齐 LUT（只需计算一次，所有帧共用）
    print("Building undistortion maps and alignment LUT...")
    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    point_clouds = []

    for i in range(NUM_FRAMES):
        # 加载原始深度图（uint16 mm）和彩色图（RGB 1920x1080）
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)

        # 深度图径向畸变校正（使用预计算的映射表）
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)

        # 将彩色图对齐到深度图分辨率（512x424）
        # 通过深度相机→彩色相机的外参变换 + 深度依赖的基线校正
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(
            depth_undist_m, lut_u, lut_v, color_rgb)

        # 双边滤波：保持边缘的同时平滑深度噪声（sigma_color=30, sigma_space=30）
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)
        depth_filtered_m = depth_filtered_mm / 1000.0

        # 深度图反投影生成三维点云（针孔相机模型）
        # X=(u-cx)*Z/fx, Y=(v-cy)*Z/fy
        pts, clr = depth_to_points(depth_filtered_m, aligned_color)

        # Save
        save_ply(
            os.path.join(out_dir, f"frame_{i:03d}.ply"), pts, clr)

        point_clouds.append((pts, clr))

        if i % 6 == 0:
            print(f"  Frame {i:03d}: {len(pts)} points "
                  f"(depth range [{depth_filtered_m[depth_filtered_m>0].min():.3f}, "
                  f"{depth_filtered_m[depth_filtered_m>0].max():.3f}] m)")

    print(f"\nSaved {NUM_FRAMES} point clouds to {out_dir}")
    return point_clouds


def main():
    print("=" * 60)
    print("STEP 2: Per-frame 3D point cloud generation")
    print("=" * 60)

    point_clouds = process_all_frames()

    # Verify: visualize 3 random frames
    print("\nVerification: visualizing frames 0, 12, 24...")
    print("  Pot should be white, plant should be green")

    for fidx in [0, 12, 24]:
        pts, clr = point_clouds[fidx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(clr)
        print(f"\n  Frame {fidx}: {len(pts)} points — close window to continue")
        visualize_point_clouds(
            [pcd],
            window_name=f"Step 2 — Frame {fidx:03d} (close to continue)")

    print("\nStep 2 complete.")
    print(f"  {NUM_FRAMES} point clouds → pipeline_output/frames/")


if __name__ == "__main__":
    main()
