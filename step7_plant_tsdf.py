"""
Step 7: 全场景 TSDF 体素融合重建（体素=3mm，截断=15mm）

将 36 帧 RGB-D 数据通过固定旋转位姿逐帧集成到 TSDF 体积中。
TSDF（Truncated Signed Distance Function）通过加权平均多帧观测值，
自然地融合所有视角的点云数据，输出连续三角网格。
核心步骤：RGB-D 图像创建 → TSDF 积分 → 网格抽取 → 顶点聚类简化
"""
import numpy as np
import cv2
import open3d as o3d
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import (
    load_depth, load_color_rgb, bilateral_filter_depth,
    rot_around_center, visualize_mesh, save_mesh
)
from step1_align import (
    build_undistort_maps, build_alignment_lut,
    undistort_depth, compute_alignment_with_depth
)


def main():
    print("=" * 60)
    print("STEP 7: Full-scene TSDF reconstruction")
    print("=" * 60)

    map1, map2 = build_undistort_maps()
    lut_u, lut_v = build_alignment_lut()

    # 创建 TSDF 体素体积：voxel_length=3mm 控制网格细节分辨率，
    # sdf_trunc=15mm 控制融合截断距离（越小表面越锐利）
    # color_type=RGB8 表示同时融合颜色信息
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF_VOXEL,
        sdf_trunc=TSDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    # 深度相机内参矩阵（512x424，fx,fy,cx,cy = 报告实验参数）
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        DEPTH_WIDTH, DEPTH_HEIGHT, DEPTH_FX, DEPTH_FY, DEPTH_CX, DEPTH_CY)

    for i in range(NUM_FRAMES):
        # 加载第 i 帧的深度图（uint16 mm）和彩色图（1920x1080 RGB）
        depth_mm = load_depth(i)
        color_rgb = load_color_rgb(i)
        depth_undist_mm = undistort_depth(depth_mm, map1, map2)
        depth_undist_m = depth_undist_mm / 1000.0
        aligned_color = compute_alignment_with_depth(
            depth_undist_m, lut_u, lut_v, color_rgb)
        depth_filtered_mm = bilateral_filter_depth(depth_undist_mm)

        # 创建 Open3D 图像对象，构建 RGB-D 图像
        # depth_scale=1000.0 将 uint16 mm 转换为米
        # depth_trunc=1.5m 截断远距离无效深度
        depth_img = o3d.geometry.Image(depth_filtered_mm.astype(np.uint16))
        color_bgr = cv2.cvtColor(aligned_color, cv2.COLOR_RGB2BGR)
        color_img = o3d.geometry.Image(color_bgr.astype(np.uint8))

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img,
            depth_scale=1000.0, depth_trunc=TSDF_DEPTH_MAX,
            convert_rgb_to_intensity=False)

        # 使用纯旋转位姿（与 Step5/6 保持一致）
        # 旋转中心为 (-0.0346, 0.7861)，每帧旋转 10°
        # 将 36 帧从相机坐标系变换到统一的世界坐标系
        world_pose = rot_around_center(i * DEG_PER_FRAME)
        # TSDF 需要相机位姿（world-to-camera），即世界坐标系的逆变换
        cam_pose = np.linalg.inv(world_pose)

        tsdf.integrate(rgbd, intrinsic, cam_pose)

        if i % 6 == 0:
            print(f"  Frame {i:03d} integrated")

    # ========== 网格抽取 ==========
    # 从 TSDF 体素中抽取零等值面（zero-crossing），生成三角网格
    print("\nExtracting mesh...")
    mesh = tsdf.extract_triangle_mesh()

    # 顶点聚类简化：将体素大小内的顶点合并为一个，减少网格顶点数
    # 同时去除退化三角形和重复顶点，保证网格拓扑正确
    mesh = mesh.simplify_vertex_clustering(
        VOXEL_SIZE, contraction=o3d.geometry.SimplificationContraction.Average)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()

    # 移除包含 NaN/Inf 的无效顶点（深度噪声可能导致非法值）
    verts = np.asarray(mesh.vertices)
    fin = np.all(np.isfinite(verts), axis=1)
    if not fin.all():
        mesh.remove_vertices_by_mask(~fin)

    # No Taubin — leave for step 8 fusion
    mesh.compute_vertex_normals()

    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    out_path = os.path.join(OUTPUT_DIR, "plant_mesh.ply")
    save_mesh(out_path, mesh)
    print(f"Saved: {out_path}")

    print("\nStep 7 complete.")


if __name__ == "__main__":
    main()
