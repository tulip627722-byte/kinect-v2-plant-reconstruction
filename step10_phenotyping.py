"""
Step 10: 植物表型特征提取（补充分割—叶片计数 + 叶倾角）

基于 Step3 分割后的植物点云，进一步提取：
1. 叶片数量（连通域分析 + 聚类）
2. 叶倾角分布（PCA 计算叶片法线与竖直方向的夹角）
3. 叶片面积（各叶片簇投影面积）

输入：pipeline_output/segmented/plant_000.ply 等 36 帧
      或 pipeline_output/plant_mesh.ply（Step7 输出）
"""
import numpy as np
import open3d as o3d
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *


def count_leaves_from_pointcloud(pcd, eps=0.015, min_points=20):
    """
    基于 DBSCAN 聚类的叶片数量估计。

    参数：
        pcd: 植物点云
        eps: 聚类半径（15mm，叶片间距典型值）
        min_points: 最小簇点数（过小的簇视为噪声）

    返回：
        n_leaves: 叶片数量
        labels: 每个点的簇标签（-1 为噪声）
    """
    pts = np.asarray(pcd.points)
    if len(pts) < min_points:
        return 0, np.array([])

    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(pts)
    labels = clustering.labels_

    # 统计非噪声簇数量
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})

    return n_clusters, labels


def estimate_leaf_angles(pcd, labels):
    """
    基于 PCA 的叶倾角估计。

    对每个叶片簇计算其主方向（PCA第一主成分方向），
    叶倾角 = 主方向与竖直方向 (Y轴) 的夹角。

    参数：
        pcd: 植物点云
        labels: 每个点的簇标签

    返回：
        angles: 各叶片的倾角列表（度）
    """
    pts = np.asarray(pcd.points)
    unique_labels = [l for l in set(labels) if l != -1]

    angles = []
    for label in unique_labels:
        mask = labels == label
        cluster_pts = pts[mask]

        if len(cluster_pts) < 10:
            continue

        # PCA 计算主方向
        center = cluster_pts.mean(axis=0)
        centered = cluster_pts - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 第一主成分（最大特征值对应的特征向量）
        main_dir = eigenvectors[:, np.argmax(eigenvalues)]

        # 与竖直方向 (Y轴, [0,1,0]) 的夹角
        vertical = np.array([0, 1, 0])
        cos_angle = abs(np.dot(main_dir, vertical))
        cos_angle = np.clip(cos_angle, 0, 1)
        angle = np.degrees(np.arccos(cos_angle))

        angles.append(round(angle, 1))

    return angles


def estimate_leaf_area(pts, resolution=0.002):
    """
    叶片投影面积估计（XZ 平面投影）。

    将点云投影到 XZ 平面，用 2mm 分辨率网格化，
    统计非空格子数 × 格子面积。

    参数：
        pts: (N, 3) 点云
        resolution: 投影网格分辨率（2mm）

    返回：
        area_cm2: 投影面积（cm²）
    """
    # 投影到 XZ 平面
    xz = pts[:, [0, 2]]

    # 构建 2D 网格
    x_min, x_max = xz[:, 0].min(), xz[:, 0].max()
    z_min, z_max = xz[:, 1].min(), xz[:, 1].max()

    x_bins = int((x_max - x_min) / resolution) + 1
    z_bins = int((z_max - z_min) / resolution) + 1

    if x_bins <= 1 or z_bins <= 1:
        return 0.0

    # 网格化计数
    x_idx = ((xz[:, 0] - x_min) / resolution).astype(int)
    z_idx = ((xz[:, 1] - z_min) / resolution).astype(int)
    x_idx = np.clip(x_idx, 0, x_bins - 1)
    z_idx = np.clip(z_idx, 0, z_bins - 1)

    grid = np.zeros((x_bins, z_bins), dtype=bool)
    grid[x_idx, z_idx] = True

    # 面积 = 非空格子数 × 格子面积
    area_m2 = np.sum(grid) * (resolution ** 2)
    area_cm2 = area_m2 * 10000  # m² → cm²

    return round(area_cm2, 1)


def main():
    print("=" * 60)
    print("STEP 10: 植物表型精细化特征提取")
    print("  叶片数量 + 叶倾角分布 + 叶面积")
    print("=" * 60)

    # 尝试从 segmented 目录加载各帧植物点云
    seg_dir = os.path.join(OUTPUT_DIR, "segmented")
    if not os.path.exists(seg_dir):
        print(f"错误：找不到 {seg_dir}，请先运行 Step3")
        return

    # 累加所有帧的植物点云
    all_plant_pts = []
    for i in range(NUM_FRAMES):
        fpath = os.path.join(seg_dir, f"plant_{i:03d}.ply")
        if os.path.exists(fpath):
            pcd = o3d.io.read_point_cloud(fpath)
            pts = np.asarray(pcd.points)
            if len(pts) > 0:
                all_plant_pts.append(pts)

    if not all_plant_pts:
        print("错误：未找到植物点云数据")
        return

    plant_pts = np.vstack(all_plant_pts)
    print(f"\n加载植物点云：{len(plant_pts)} 点（{len(all_plant_pts)} 帧累加）")

    # 体素下采样（3mm）去除冗余点
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plant_pts)
    pcd = pcd.voxel_down_sample(0.003)
    plant_pts = np.asarray(pcd.points)
    print(f"体素下采样(3mm)后：{len(plant_pts)} 点")

    # ========== 1. 叶片数量估计 ==========
    print("\n" + "-" * 40)
    print("1. 叶片数量估计（DBSCAN 聚类，eps=15mm）")
    print("-" * 40)

    n_leaves, labels = count_leaves_from_pointcloud(pcd, eps=0.015, min_points=20)
    print(f"  检测到叶片数量：{n_leaves} 片")

    # ========== 2. 叶倾角分布 ==========
    print("\n" + "-" * 40)
    print("2. 叶倾角分布（PCA 主成分分析）")
    print("-" * 40)

    angles = estimate_leaf_angles(pcd, labels)
    if angles:
        print(f"  叶片倾角列表：{angles}°")
        print(f"  平均倾角：{np.mean(angles):.1f}°")
        print(f"  最大倾角：{max(angles)}°")
        print(f"  最小倾角：{min(angles)}°")
    else:
        print("  未能有效分割叶簇")

    # ========== 3. 叶面积估计 ==========
    print("\n" + "-" * 40)
    print("3. 叶面积估计（XZ 平面投影，2mm 分辨率）")
    print("-" * 40)

    area_cm2 = estimate_leaf_area(plant_pts, resolution=0.002)
    print(f"  投影叶面积：约 {area_cm2} cm²")

    # ========== 4. 株高 & 冠幅（补充完整）==========
    print("\n" + "-" * 40)
    print("4. 基础表型参数（株高 + 冠幅）")
    print("-" * 40)

    height_cm = round((plant_pts[:, 1].max() - plant_pts[:, 1].min()) * 100, 1)
    crown_x_cm = round((plant_pts[:, 0].max() - plant_pts[:, 0].min()) * 100, 1)
    crown_z_cm = round((plant_pts[:, 2].max() - plant_pts[:, 2].min()) * 100, 1)
    print(f"  株高：{height_cm} cm")
    print(f"  冠幅（X向）：{crown_x_cm} cm")
    print(f"  冠幅（Z向）：{crown_z_cm} cm")

    # ========== 保存结果 ==========
    results = {
        "n_leaves": n_leaves,
        "leaf_angles_deg": angles,
        "mean_leaf_angle_deg": round(np.mean(angles), 1) if angles else None,
        "leaf_area_cm2": area_cm2,
        "plant_height_cm": height_cm,
        "crown_x_cm": crown_x_cm,
        "crown_z_cm": crown_z_cm,
        "total_points": len(plant_pts),
    }
    result_path = os.path.join(OUTPUT_DIR, "phenotype.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n保存表型参数：{result_path}")
    print("\nStep 10 完成。")


if __name__ == "__main__":
    main()
