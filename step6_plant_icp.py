"""
Step 6: 植物点云 PointToPoint ICP 配准

在 Step5 纯旋转初值基础上，逐帧做 PointToPoint ICP 精配准。
（不同于花盆用 PointToPlane，植物叶片法线不可靠，故用 PointToPoint）

参数：
  - max_correspondence_distance = 0.03m（3cm）
  - max_iteration = 50
  - fitness < 0.2 或 inlier_rmse > 0.01 → 跳过该帧

如果超过 10 帧被跳过 → 放弃 ICP，回退到纯旋转位姿。

输出：pipeline_output/plant_icp/plant_merged.ply
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
    icp_point_to_point,
    visualize_point_clouds, make_colored_pcd, save_ply
)
from step1_align import build_undistort_maps, build_alignment_lut, undistort_depth, compute_alignment_with_depth
from step3_segment import segment_frame


def main():
    print("=" * 60)
    print("STEP 6: Plant PointToPoint ICP registration")
    print("=" * 60)

    # Build alignment maps
    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    # Collect plant point clouds (in camera frame)
    plant_pcds_cam = []
    for i in range(NUM_FRAMES):
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(depth_undist_m, lut_u, lut_v, color_rgb)
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)
        depth_filtered_m = depth_filtered_mm / 1000.0
        _, plant_pcd = segment_frame(depth_filtered_m, aligned_color)
        plant_pcds_cam.append(plant_pcd)

    # Register frame-by-frame with PointToPoint ICP
    # Strategy: register each frame to the accumulated world cloud
    print("\nPointToPoint ICP (threshold=0.03m, max_iter=50)...")
    print(f"  Quality checks: fitness >= {PLANT_ICP_MIN_FITNESS}, "
          f"inlier_rmse <= {PLANT_ICP_MAX_RMSE}")

    world_cloud = o3d.geometry.PointCloud()
    # Frame 0 as reference at identity
    frame0_world = o3d.geometry.PointCloud(plant_pcds_cam[0])
    world_cloud += frame0_world

    num_skipped = 0
    icp_results = []

    for i in range(1, NUM_FRAMES):
        src = plant_pcds_cam[i]
        n_pts = len(src.points)

        if n_pts < 20:
            print(f"  Frame {i:03d}: too few pts ({n_pts}), SKIP")
            num_skipped += 1
            continue

        # Initial: undo turntable rotation
        init = rot_around_center(-i * DEG_PER_FRAME)

        try:
            reg = icp_point_to_point(
                src, world_cloud, init,
                PLANT_ICP_THRESHOLD, PLANT_ICP_MAX_ITER)

            fitness = reg.fitness
            rmse = reg.inlier_rmse

            # 质量检查：fitness 太低或 RMSE 太高 → 弃用该帧
            # 叶片无稳定几何特征，误匹配帧宁缺毋滥
            if fitness < PLANT_ICP_MIN_FITNESS or rmse > PLANT_ICP_MAX_RMSE:
                print(f"  Frame {i:03d}: fit={fitness:.3f} rmse={rmse:.4f}m — "
                      f"BELOW QUALITY, SKIP")
                num_skipped += 1
                continue

            # ICP 成功的帧：变换到世界坐标系后加入累积点云
            # 逐帧累加的策略确保每帧只与已对齐的全局点云配准
            src_w = o3d.geometry.PointCloud(src)
            src_w.transform(reg.transformation)
            world_cloud += src_w

            if i % 6 == 0:
                print(f"  Frame {i:03d}: fit={fitness:.3f} rmse={rmse:.4f}m OK "
                      f"| total={len(world_cloud.points)} pts")

            icp_results.append({
                "frame": i,
                "fitness": float(fitness),
                "rmse": float(rmse),
                "accepted": True
            })

        except Exception as e:
            print(f"  Frame {i:03d}: ICP error ({e}), SKIP")
            num_skipped += 1

    # Summary
    print(f"\nResults: {NUM_FRAMES - 1 - num_skipped} accepted, "
          f"{num_skipped} skipped (out of {NUM_FRAMES - 1})")

    if num_skipped > 10:
        print("\n  [WARNING] 超过 10 帧被跳过，放弃 ICP！")
        print("  回退到纯旋转位姿（Step5 的结果）")
        # Fall back to pure rotation
        all_plant_world = []
        for i in range(NUM_FRAMES):
            pts = np.asarray(plant_pcds_cam[i].points)
            if len(pts) > 0:
                angle = i * DEG_PER_FRAME
                all_plant_world.append(rotate_points_around_center(pts, angle))
        merged = np.vstack(all_plant_world)
    else:
        merged = np.asarray(world_cloud.points)

    # Save
    out_dir = os.path.join(OUTPUT_DIR, "plant_icp")
    os.makedirs(out_dir, exist_ok=True)
    save_ply(os.path.join(out_dir, "plant_merged.ply"), merged)

    print(f"\nPlant merged: {len(merged)} pts")
    print(f"  X range: [{merged[:,0].min():.3f}, {merged[:,0].max():.3f}]")
    print(f"  Y range: [{merged[:,1].min():.3f}, {merged[:,1].max():.3f}]")
    print(f"  Z range: [{merged[:,2].min():.3f}, {merged[:,2].max():.3f}]")

    # Visualize
    print("\nVerification: ICP-registered plant cloud — close window to continue")
    print("  Should be spherical cluster, no explosion")

    pcd = make_colored_pcd(merged, [0.2, 0.7, 0.3])
    visualize_point_clouds(
        [pcd],
        window_name="Step 6 — Plant ICP Result (close to continue)")

    # Also save individual ICP results
    import json
    with open(os.path.join(out_dir, "icp_report.json"), "w") as f:
        json.dump({
            "num_accepted": NUM_FRAMES - 1 - num_skipped,
            "num_skipped": num_skipped,
            "abandoned_icp": num_skipped > 10,
            "results": icp_results
        }, f, indent=2)

    print("\nStep 6 complete.")
    print(f"  Output: pipeline_output/plant_icp/plant_merged.ply")


if __name__ == "__main__":
    main()
