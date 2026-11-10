"""
Step 4: Pot registration (FPFH+RANSAC → PointToPlane ICP + Pose Graph) + TSDF reconstruction.

For each frame's pot point cloud:
1. FPFH feature extraction (Search Radius=25mm)         ← 报告 Step 4
2. RANSAC coarse registration (4,000,000 iterations)    ← 报告 Step 4
3. Point-to-Plane ICP (max_correspondence_distance=3mm) ← 报告 Step 4
4. Pose graph optimization (35 adjacent + 1 loop closure)
5. TSDF fusion with optimized poses
6. Extract mesh, vertex clustering, Taubin smooth x3
7. Save pose.json with per-frame RMSE / R / t           ← 报告 Rule 7

Output: pipeline_output/pot_mesh.ply
"""
import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import (
    load_depth, load_color_rgb, bilateral_filter_depth,
    rot_around_center, sor_filter, voxel_downsample, estimate_normals,
    visualize_mesh, save_mesh
)
from step1_align import build_undistort_maps, build_alignment_lut, undistort_depth, compute_alignment_with_depth
from step3_segment import segment_frame


# ============================================================
# Step 4 核心流程：
# 1. 加载 36 帧 RGB-D 数据
# 2. 深度图畸变校正 + 颜色对齐 + 双边滤波
# 3. 每帧分割出花盆点云
# 4. FPFH+RANSAC 粗配准 + Point-to-Plane ICP 精配准
# 5. 位姿图优化（含闭环约束）
# 6. TSDF 体素融合重建
# 7. 输出 pot_mesh.ply + pose.json（每帧 RMSE）
# ============================================================


def extract_pot_clouds():
    """Extract pot point clouds from all 36 frames."""
    print("Extracting pot clouds from all frames...")
    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    pot_pcds = []

    for i in range(NUM_FRAMES):
        # 逐帧加载深度和彩色图像
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(depth_undist_m, lut_u, lut_v, color_rgb)
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)
        depth_filtered_m = depth_filtered_mm / 1000.0

        pot_pcd, _ = segment_frame(depth_filtered_m, aligned_color)
        pot_pcds.append(pot_pcd)

        if i % 6 == 0:
            print(f"  Frame {i:03d}: pot={len(pot_pcd.points)} pts")

    return pot_pcds


def register_pot_fpfh_ransac_icp(pot_pcds):
    """
    Register pot clouds using adjacent-frame PointToPlane ICP.

    Returns:
        world_poses: list of 4x4 numpy arrays (world-from-camera for each frame)
    """
    print("\n" + "-" * 40)
    print("FPFH feature extraction -> RANSAC coarse -> Point-to-Plane ICP...")
    print("  (FPFH radius=0.025m, RANSAC iter=4,000,000, ICP threshold=0.003m)")
    print("-" * 40)

    world_poses = [np.eye(4)]  # Frame 0 at identity
    icp_records = [{"frame": 0, "angle_deg": 0.0,
                     "rotation_matrix": np.eye(3).tolist(),
                     "translation": [0.0, 0.0, 0.0],
                     "icp_rmse_m": 0.0, "method": "reference"}]

    for i in range(1, NUM_FRAMES):
        src = pot_pcds[i]
        tgt = pot_pcds[i - 1]

        n_src, n_tgt = len(src.points), len(tgt.points)
        if n_src < 50 or n_tgt < 50:
            print(f"  Frame {i:03d}: too few points ({n_src}, {n_tgt}), using nominal pose")
            T = rot_around_center(-DEG_PER_FRAME)
            icp_records.append({
                "frame": i, "angle_deg": round(i * DEG_PER_FRAME, 1),
                "rotation_matrix": T[:3, :3].tolist(),
                "translation": T[:3, 3].tolist(),
                "icp_rmse_m": -1.0, "method": "nominal"})
        else:
            try:
                # --- Step A: FPFH 特征提取（Search Radius=25mm）---
                # FPFH（Fast Point Feature Histograms）对每个点计算其邻域的
                # 几何特征直方图，描述局部表面形状。半径 25mm 可捕获花盆圆柱面特征
                fpfh_src = o3d.pipelines.registration.compute_fpfh_feature(
                    src, o3d.geometry.KDTreeSearchParamRadius(0.025))
                fpfh_tgt = o3d.pipelines.registration.compute_fpfh_feature(
                    tgt, o3d.geometry.KDTreeSearchParamRadius(0.025))

                # --- Step B: RANSAC 粗配准（4,000,000 次迭代，5mm 阈值）---
                # 在 FPFH 特征空间中随机采样 3 组对应点估计刚体变换
                # 400 万次迭代确保找到全局最优初值，避免 ICP 落入局部最优
                result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    src, tgt, fpfh_src, fpfh_tgt, True, 0.005,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    3,
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.005)],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

                # --- Step C: Point-to-Plane ICP 精配准（3mm 阈值）---
                # 以 RANSAC 输出为初值，用点到面距离优化精调位姿
                # 点到面比点到点收敛更快，对花盆圆柱面效果更好
                reg = o3d.pipelines.registration.registration_icp(
                    src, tgt, 0.003, result_ransac.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6,
                        max_iteration=POT_ICP_MAX_ITER))
                T = reg.transformation
                R = T[:3, :3]
                t_vec = T[:3, 3].tolist()
                r_x = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
                r_y = float(np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))))
                r_z = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

                print(f"  Frame {i:03d}: R=({r_x:.2f}, {r_y:.2f}, {r_z:.2f})"
                      f" | t=({t_vec[0]:.4f}, {t_vec[1]:.4f}, {t_vec[2]:.4f})"
                      f" | rmse={reg.inlier_rmse:.4f}m fit={reg.fitness:.3f}"
                      f" | src={n_src} tgt={n_tgt}")

                icp_records.append({
                    "frame": i,
                    "angle_deg": round(i * DEG_PER_FRAME, 1),
                    "rotation_matrix": R.tolist(),
                    "translation": t_vec,
                    "icp_rmse_m": round(reg.inlier_rmse, 4),
                    "fitness": round(reg.fitness, 3),
                    "method": "fpfh_ransac_p2plane"})
            except Exception as e:
                print(f"  Frame {i:03d}: FPFH+RANSAC+ICP failed ({e}), using nominal")
                T = rot_around_center(-DEG_PER_FRAME)
                icp_records.append({
                    "frame": i, "angle_deg": round(i * DEG_PER_FRAME, 1),
                    "rotation_matrix": T[:3, :3].tolist(),
                    "translation": T[:3, 3].tolist(),
                    "icp_rmse_m": -2.0, "method": "fallback_nominal"})

        world_poses.append(world_poses[-1] @ T)

    return world_poses, icp_records


def optimize_pose_graph(pot_pcds, world_poses):
    """
    Build and optimize pose graph with loop closure.
    """
    print("\n" + "-" * 40)
    print("Pose graph optimization...")
    print("-" * 40)

    pg = o3d.pipelines.registration.PoseGraph()

    # Add nodes
    for i in range(NUM_FRAMES):
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(world_poses[i]))

    # Add adjacent edges
    for i in range(1, NUM_FRAMES):
        T_edge = np.linalg.inv(world_poses[i - 1]) @ world_poses[i]
        src = pot_pcds[i]
        tgt = pot_pcds[i - 1]

        if len(src.points) >= 30 and len(tgt.points) >= 30:
            info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                tgt, src, POT_ICP_THRESHOLD, T_edge)
        else:
            info = np.eye(6) * 0.1

        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            i - 1, i, T_edge, info, uncertain=False))

    # Loop closure: frame 35 → frame 0
    print("  Computing loop closure (frame 35 → frame 0)...")
    src_loop = pot_pcds[NUM_FRAMES - 1]
    tgt_loop = pot_pcds[0]
    init_loop = rot_around_center(DEG_PER_FRAME)  # rotate forward by 10°

    try:
        reg_lc = o3d.pipelines.registration.registration_icp(
            src_loop, tgt_loop, POT_ICP_THRESHOLD, init_loop,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=POT_ICP_MAX_ITER))
        info_lc = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            tgt_loop, src_loop, POT_ICP_THRESHOLD, reg_lc.transformation)
        pg.edges.append(o3d.pipelines.registration.PoseGraphEdge(
            NUM_FRAMES - 1, 0, reg_lc.transformation, info_lc, uncertain=False))
        print(f"  Loop closure: fit={reg_lc.fitness:.3f} rmse={reg_lc.inlier_rmse:.4f}m")
    except Exception as e:
        print(f"  Loop closure skipped: {e}")

    # Optimize
    print("  Running pose graph optimization...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=POT_ICP_THRESHOLD,
        edge_prune_threshold=0.25,
        reference_node=0)
    o3d.pipelines.registration.global_optimization(
        pg,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    opt_poses = [node.pose for node in pg.nodes]

    # Print summary
    angle_18 = np.degrees(np.arctan2(opt_poses[18][0, 2], opt_poses[18][0, 0]))
    angle_35 = np.degrees(np.arctan2(opt_poses[35][0, 2], opt_poses[35][0, 0]))
    print(f"  Optimized: frame 18 angle={angle_18:.1f}° (expect ~180°), "
          f"frame 35 angle={angle_35:.1f}° (expect ~350°)")

    return opt_poses


def tsdf_pot(opt_poses):
    """
    Build TSDF volume from pot point clouds using optimized poses.

    Integrates each frame's point cloud into TSDF by:
    1. Transforming points to world frame
    2. Computing camera pose from world pose
    3. Integrating the RGBD image
    """
    print("\n" + "-" * 40)
    print("TSDF pot reconstruction (voxel=3mm, trunc=15mm)...")
    print("-" * 40)

    # Build per-frame data
    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_VOXEL,
        sdf_trunc=TSDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)

    for i in range(NUM_FRAMES):
        # Load and prepare data
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(depth_undist_m, lut_u, lut_v, color_rgb)
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)

        # Create RGBD image
        depth_img = o3d.geometry.Image(depth_filtered_mm.astype(np.uint16))
        color_bgr = cv2.cvtColor(aligned_color, cv2.COLOR_RGB2BGR)
        color_img = o3d.geometry.Image(color_bgr.astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1000.0,
            depth_trunc=TSDF_DEPTH_MAX,
            convert_rgb_to_intensity=False)

        # Camera pose = inverse of world-from-camera
        world_pose = opt_poses[i]
        cam_pose = np.linalg.inv(world_pose)

        tsdf.integrate(rgbd, intrinsic, cam_pose)

        if i % 6 == 0:
            print(f"  Frame {i:03d} integrated")

    # Extract mesh
    print("\nExtracting pot mesh...")
    pot_mesh = tsdf.extract_triangle_mesh()

    # Cleanup
    pot_mesh = pot_mesh.simplify_vertex_clustering(
        VOXEL_SIZE, contraction=o3d.geometry.SimplificationContraction.Average)
    pot_mesh = pot_mesh.filter_smooth_taubin(number_of_iterations=TAUBIN_ITERATIONS)
    pot_mesh.remove_degenerate_triangles()
    pot_mesh.remove_duplicated_vertices()

    print(f"Pot mesh: {len(pot_mesh.vertices)} vertices, {len(pot_mesh.triangles)} triangles")

    return pot_mesh


def main():
    print("=" * 60)
    print("STEP 4: Pot registration (FPFH+RANSAC + ICP + PoseGraph) + TSDF")
    print("=" * 60)

    # Extract pot point clouds
    pot_pcds = extract_pot_clouds()

    # Register with PointToPlane ICP
    world_poses, icp_records = register_pot_fpfh_ransac_icp(pot_pcds)

    # Pose graph optimization
    opt_poses = optimize_pose_graph(pot_pcds, world_poses)

    # TSDF reconstruction
    pot_mesh = tsdf_pot(opt_poses)

    # Save
    out_path = os.path.join(OUTPUT_DIR, "pot_mesh.ply")
    save_mesh(out_path, pot_mesh)
    print(f"\nSaved: {out_path}")

    # Save pose.json with per-frame data
    pose_data = {
        "calibration": {"fx": DEPTH_FX, "fy": DEPTH_FY, "cx": DEPTH_CX, "cy": DEPTH_CY},
        "rotation_center": {"x": ROT_CX, "z": ROT_CZ},
        "deg_per_frame": DEG_PER_FRAME,
        "num_frames": NUM_FRAMES,
        "registration_method": "FPFH -> RANSAC -> PointToPlane ICP -> PoseGraph",
        "frames": icp_records
    }
    pose_path = os.path.join(OUTPUT_DIR, "pose.json")
    with open(pose_path, "w", encoding="utf-8") as f:
        json.dump(pose_data, f, indent=2)
    print(f"Saved: {pose_path} ({len(icp_records)} frames)")

    # Verify
    print("\nVerification: pot mesh — close window to confirm")
    print("  Pot should be complete, no ghosting, no distortion")
    visualize_mesh(pot_mesh, "Step 4 — Pot Mesh (close to confirm)")

    print("\nStep 4 complete. pot_mesh.ply is LOCKED — do not re-run.")


if __name__ == "__main__":
    main()
