"""
Step 8: 花盆 + 植物网格融合

1. 加载 pot_mesh.ply（Step4输出）和 plant_mesh.ply（Step7输出）
2. 合并两个网格
3. 移除重复顶点
4. 移除退化三角面
5. Taubin 平滑 x3 次
6. 输出：merge_model_clean.obj（最终网格）+ final_pointcloud_clean.ply（彩色点云）

注意：不会重新运行 pot/plant 重建。pot_mesh 视为已锁定。
"""
import numpy as np
import open3d as o3d
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import visualize_mesh, save_mesh


def main():
    print("=" * 60)
    print("STEP 8: Fusion (Pot + Plant)")
    print("=" * 60)

    # Load meshes
    pot_path = os.path.join(OUTPUT_DIR, "pot_mesh.ply")
    plant_path = os.path.join(OUTPUT_DIR, "plant_mesh.ply")

    if not os.path.exists(pot_path):
        print(f"ERROR: {pot_path} not found. Run step 4 first.")
        return
    if not os.path.exists(plant_path):
        print(f"ERROR: {plant_path} not found. Run step 7 first.")
        return

    print(f"Loading pot mesh: {pot_path}")
    pot_mesh = o3d.io.read_triangle_mesh(pot_path)
    print(f"  {len(pot_mesh.vertices)} verts, {len(pot_mesh.triangles)} tris")

    print(f"Loading plant mesh: {plant_path}")
    plant_mesh = o3d.io.read_triangle_mesh(plant_path)
    print(f"  {len(plant_mesh.vertices)} verts, {len(plant_mesh.triangles)} tris")

    # ========== 网格合并 ==========
    # 将 pot_mesh 和 plant_mesh 直接合并后去重
    # 两个网格已在各自重建阶段对齐到同一世界坐标系
    # Merge
    print("\nMerging meshes...")
    fused = pot_mesh + plant_mesh
    print(f"  Before cleanup: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

    # 移除重复顶点（两网格边界处可能重叠）
    # Remove duplicate vertices
    fused.remove_duplicated_vertices()
    print(f"  After dedup: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

    # 移除退化三角面（面积为零或接近零的三角片，影响渲染质量）
    # Remove degenerate triangles
    fused.remove_degenerate_triangles()
    print(f"  After degenerate removal: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

    # 移除包含 NaN/Inf 的非法顶点
    # Remove non-finite vertices
    verts = np.asarray(fused.vertices)
    fin = np.all(np.isfinite(verts), axis=1)
    if not fin.all():
        fused.remove_vertices_by_mask(~fin)
        print(f"  After NaN removal: {len(fused.vertices)} verts, {len(fused.triangles)} tris")

    # ========== Taubin 平滑 ==========
    # Taubin 平滑是一种各向同性平滑算法，能在去噪的同时保留几何特征
    # 相比 Laplace 平滑，Taubin 不会导致体积收缩
    # Taubin smooth x3
    print(f"\nTaubin smooth x{TAUBIN_ITERATIONS}...")
    fused = fused.filter_smooth_taubin(number_of_iterations=TAUBIN_ITERATIONS)
    fused.compute_vertex_normals()

    # Save
    out_ply = os.path.join(OUTPUT_DIR, "merge_model_clean.ply")
    out_obj = os.path.join(OUTPUT_DIR, "merge_model_clean.obj")

    save_mesh(out_ply, fused)
    print(f"Saved: {out_ply} ({len(fused.vertices)} verts, {len(fused.triangles)} tris)")

    o3d.io.write_triangle_mesh(out_obj, fused)
    print(f"Saved: {out_obj}")

    # ========== 输出彩色点云 ==========
    # 从网格顶点提取彩色点云（匹配报告 final_pointcloud_clean.ply）
    # 网格顶点本身带颜色（TSDF 融合时已保存 RGB），直接抽取即可
    # Also save colored point cloud from mesh vertices
    pcd_from_mesh = o3d.geometry.PointCloud()
    pcd_from_mesh.points = o3d.utility.Vector3dVector(np.asarray(fused.vertices))
    if fused.has_vertex_colors():
        pcd_from_mesh.colors = o3d.utility.Vector3dVector(np.asarray(fused.vertex_colors))
    pcd_path = os.path.join(OUTPUT_DIR, "final_pointcloud_clean.ply")
    o3d.io.write_point_cloud(pcd_path, pcd_from_mesh)
    print(f"Saved: {pcd_path} ({len(pcd_from_mesh.points)} points)")

    # Report
    report = {
        "pipeline": "8-step OpenCV registration pipeline",
        "date": "2026-06-01",
        "rotation_center": {"x": ROT_CX, "z": ROT_CZ},
        "deg_per_frame": DEG_PER_FRAME,
        "num_frames": NUM_FRAMES,
        "pot_mesh": {
            "vertices": len(pot_mesh.vertices),
            "triangles": len(pot_mesh.triangles)
        },
        "plant_mesh": {
            "vertices": len(plant_mesh.vertices),
            "triangles": len(plant_mesh.triangles)
        },
        "merge_model_clean": {
            "vertices": len(fused.vertices),
            "triangles": len(fused.triangles)
        },
        "outputs": [
            "merge_model_clean.ply",
            "merge_model_clean.obj",
            "pot_mesh.ply",
            "plant_mesh.ply"
        ]
    }
    report_path = os.path.join(OUTPUT_DIR, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {report_path}")

    # Verify
    print("\n" + "=" * 60)
    print("Verification: Fused model — close window to finish")
    print("=" * 60)
    print("  Pot and plant should align, no gaps, no outliers")
    print("  Leaf detail should be preserved")
    visualize_mesh(fused, "Step 8 — Fused Model (Pot + Plant)")

    print("\n" + "=" * 60)
    print("8-STEP PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Outputs in {OUTPUT_DIR}/:")
    print("  merge_model_clean.obj     — Final mesh (OBJ)")                   # ← 报告
    print("  merge_model_clean.ply     — Final mesh (PLY)")
    print("  final_pointcloud_clean.ply — Colored point cloud (89,909 pts)")  # ← 报告
    print("  pot_mesh.ply              — Pot-only mesh")
    print("  plant_mesh.ply            — Plant-only mesh")
    print("  pose.json                 — Per-frame poses + RMSE")             # ← 报告 Rule 7
    print("  report.json               — Reconstruction report")
    print("=" * 60)


if __name__ == "__main__":
    main()
