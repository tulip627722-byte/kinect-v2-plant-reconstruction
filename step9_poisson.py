"""
Step 9: Poisson 曲面重建（匹配报告第 4.5 节核心代码）

基于融合后的彩色点云（final_pointcloud_clean.ply），
执行 Poisson 曲面重建生成光滑三角网格。

流程：
1. 加载彩色点云
2. 估计法线（Poisson 重建前置条件）
3. 法线方向一致性修正
4. Poisson 曲面重建（depth=9 控制八叉树深度）
5. 低密度三角面剔除（去除边界噪声）
6. 输出 merge_model_clean.obj

输入：final_pointcloud_clean.ply（Step8 输出）
输出：merge_model_clean.obj（三角网格）
"""
import numpy as np
import open3d as o3d
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


def main():
    print("=" * 60)
    print("STEP 9: Poisson 曲面重建（匹配报告第 4.5 节）")
    print("=" * 60)

    # 加载 Step8 输出的彩色点云
    pcd_path = os.path.join(OUTPUT_DIR, "final_pointcloud_clean.ply")
    if not os.path.exists(pcd_path):
        print(f"错误：找不到 {pcd_path}，请先运行 Step8")
        return

    print(f"加载点云：{pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"  点数：{len(pcd.points)}")

    # ========== 法线估计（Poisson 重建前置条件）==========
    # 报告第 4.5 节：estimate_normals + orient_normals_consistent_tangent_plane
    print("\n估计法线...")
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    # 法线方向一致性修正：确保相邻点的法线朝向一致
    # 对 Poisson 重建至关重要，否则会产生面片穿插
    pcd.orient_normals_consistent_tangent_plane(100)
    print("  法线估计完成，已做一致性修正")

    # ========== Poisson 曲面重建 ==========
    # depth=9：八叉树深度，控制网格细节级别
    # depth 越大网格越精细，但计算量和噪声也增加
    print("\nPoisson 曲面重建（depth=9）...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    print(f"  重建完成：{len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # ========== 低密度三角面剔除 ==========
    # Poisson 重建会在边界处生成低密度噪声三角面
    # 剔除密度低于 5% 分位数的三角面，保留主体表面
    print("\n剔除低密度噪声三角面（阈值：5% 分位数）...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"  剔除后：{len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # ========== 后处理 ==========
    # 移除退化三角面和重复顶点
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()

    # 如果有点云颜色，传递给网格
    if pcd.has_colors():
        # 最近邻颜色传递
        from scipy.spatial import cKDTree
        tree = cKDTree(np.asarray(pcd.points))
        verts = np.asarray(mesh.vertices)
        _, idx = tree.query(verts, k=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[idx])

    print(f"\n最终网格：{len(mesh.vertices)} 顶点, {len(mesh.triangles)} 三角面")

    # ========== 保存 ==========
    out_obj = os.path.join(OUTPUT_DIR, "merge_model_clean.obj")
    out_ply = os.path.join(OUTPUT_DIR, "merge_model_clean.ply")
    o3d.io.write_triangle_mesh(out_obj, mesh)
    o3d.io.write_triangle_mesh(out_ply, mesh)
    print(f"保存：{out_obj}")
    print(f"保存：{out_ply}")

    # 可视化验证
    print("\n可视化验证 — 关闭窗口继续")
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh], window_name="Step 9 — Poisson 重建结果",
        width=1280, height=720, mesh_show_back_face=True)

    print("\nStep 9 完成。")


if __name__ == "__main__":
    main()
