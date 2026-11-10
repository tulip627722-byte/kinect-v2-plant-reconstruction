"""
Step 1: 深度图→彩色图对齐（替代 pylibfreenect2 Registration）

用 OpenCV + Kinect V2 标定参数实现：
1. 深度图径向畸变校正
2. 深度像素反投影到三维（深度相机坐标系）
3. 变换到彩色相机坐标系（外参：~52mm X向基线）
4. 投影到彩色图像平面 → 采样颜色
5. 输出：512x424 对齐后的深度 + 512x424 RGB

等效于 pylibfreenect2 Registration.apply() 的输出：
  - undistorted: 512x424 float 深度（米）
  - registered:  512x424 uint8 RGB
"""
import numpy as np
import cv2
import open3d as o3d
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from utils import load_depth, load_color_rgb, visualize_point_clouds, depth_to_points, save_ply


def build_undistort_maps():
    """
    构建深度相机畸变校正映射表。
    返回 map1, map2 用于 cv2.remap。
    由于 config.py 中畸变系数已置零，当前为恒等映射。
    """
    # We want to undistort: the output has the same camera matrix
    map1, map2 = cv2.initUndistortRectifyMap(
        K_DEPTH, D_DEPTH, None, K_DEPTH,
        (DEPTH_WIDTH, DEPTH_HEIGHT), cv2.CV_32FC1)
    return map1, map2


def undistort_depth(depth_mm, map1, map2):
    """
    Undistort depth image (mm → mm).
    Uses precomputed rectification maps for speed.
    """
    depth_float = depth_mm.astype(np.float32)
    undistorted = cv2.remap(depth_float, map1, map2, cv2.INTER_NEAREST)
    return undistorted


def build_alignment_lut():
    """
    构建深度像素→彩色像素的查询表 LUT。

    步骤：
    1. 深度像素反投影到三维点（深度相机坐标）
    2. 经外参变换到彩色相机坐标
    3. 投影到彩色图像平面

    返回：
        lut_u: (424, 512) int32, 彩色图像 u 坐标（-1 表示无效）
        lut_v: (424, 512) int32, 彩色图像 v 坐标（-1 表示无效）
    """
    h, w = DEPTH_HEIGHT, DEPTH_WIDTH  # 424, 512

    # Create pixel grids
    vv, uu = np.mgrid[0:h, 0:w]

    # Back-project to 3D with unit depth (z=1)
    x_unit = (uu - DEPTH_CX) / DEPTH_FX
    y_unit = (vv - DEPTH_CY) / DEPTH_FY

    # These are 3D points at z=1 (normalized coords)
    pts_unit = np.stack([x_unit, y_unit, np.ones_like(x_unit)], axis=-1)  # (424, 512, 3)

    # Transform to color camera frame: P_color = R @ P_depth + T
    # Since R ≈ I, T ≈ [-0.052, 0, 0], this shifts x by -0.052m at z=1
    pts_color = pts_unit.copy()
    pts_color[..., 0] += T_DEPTH_TO_COLOR[0]  # x shift (at unit depth)
    pts_color[..., 1] += T_DEPTH_TO_COLOR[1]
    pts_color[..., 2] += T_DEPTH_TO_COLOR[2]

    # Project to color image plane: u = fx * x/z + cx, v = fy * y/z + cy
    z_c = pts_color[..., 2]
    u_c = (pts_color[..., 0] * COLOR_FX / z_c + COLOR_CX).round().astype(np.int32)
    v_c = (pts_color[..., 1] * COLOR_FY / z_c + COLOR_CY).round().astype(np.int32)

    # Mask invalid projections
    valid = (
        (u_c >= 0) & (u_c < COLOR_WIDTH) &
        (v_c >= 0) & (v_c < COLOR_HEIGHT)
    )

    lut_u = np.where(valid, u_c, -1)
    lut_v = np.where(valid, v_c, -1)

    return lut_u, lut_v


def compute_alignment_with_depth(depth_m, lut_u, lut_v, color_rgb):
    """
    根据实际深度值计算深度像素对应的对齐颜色。

    LUT 在 z=1 处构建的基线偏移需要根据实际深度修正。
    对于深度 z 处的点，在彩色相机坐标系中的 x 坐标为：
      x_color = (x_depth_unit * z) + T_x
    投影 u 坐标：
      u_c = (x_color / z) * COLOR_FX + COLOR_CX
          = (x_depth_unit + T_x/z) * COLOR_FX + COLOR_CX

    LUT 在 z=1 构建时用了 T_x/1，实际深度 z 时应该用 T_x/z。
    修正公式：u_c(z) = u_c(z=1) + COLOR_FX * T_x * (1/z - 1)

    参数：
        depth_m: (424, 512) float, 深度值（米）
        lut_u: (424, 512) int32, z=1 处预计算的彩色 u 坐标
        lut_v: (424, 512) int32, z=1 处预计算的彩色 v 坐标
        color_rgb: (1080, 1920, 3) uint8 RGB 彩色图

    返回：
        aligned_color: (424, 512, 3) uint8 RGB 对齐后的彩色图
    """
    h, w = DEPTH_HEIGHT, DEPTH_WIDTH

    # Start with the z=1 LUT
    u_c = lut_u.astype(np.float64)
    v_c = lut_v.astype(np.float64)

    # Apply depth-dependent correction for the baseline
    # u_corrected = u_LUT + fx_color * T_x * (1/z - 1)
    valid_depth = (depth_m > 0.01)
    inv_z = np.zeros_like(depth_m)
    inv_z[valid_depth] = 1.0 / depth_m[valid_depth]

    u_c[valid_depth] += COLOR_FX * T_DEPTH_TO_COLOR[0] * (inv_z[valid_depth] - 1.0)
    v_c[valid_depth] += COLOR_FY * T_DEPTH_TO_COLOR[1] * (inv_z[valid_depth] - 1.0)

    # Round and clip
    u_c = np.round(u_c).astype(np.int32)
    v_c = np.round(v_c).astype(np.int32)
    valid = (
        (u_c >= 0) & (u_c < COLOR_WIDTH) &
        (v_c >= 0) & (v_c < COLOR_HEIGHT) &
        (lut_u >= 0)  # also must be valid in original LUT
    )

    # Build aligned color image
    aligned_color = np.zeros((h, w, 3), dtype=np.uint8)
    aligned_color[valid] = color_rgb[v_c[valid], u_c[valid]]

    return aligned_color


# ============================================================
# Main
# ============================================================
def process_frame(frame_idx, map1, map2, lut_u, lut_v):
    """Process one frame: load → undistort → align."""
    depth_mm = load_depth(frame_idx)
    color_rgb = load_color_rgb(frame_idx)

    # Undistort depth
    depth_undist_mm = undistort_depth(depth_mm, map1, map2)
    depth_undist_m = depth_undist_mm / 1000.0

    # Align color to depth
    aligned_color = compute_alignment_with_depth(
        depth_undist_m, lut_u, lut_v, color_rgb)

    return depth_undist_m, aligned_color


def main():
    print("=" * 60)
    print("STEP 1: Depth-to-color alignment")
    print("=" * 60)

    # Precompute maps and LUTs (done once)
    print("Building undistortion maps...")
    map1, map2 = build_undistort_maps()

    print("Building alignment LUT...")
    lut_u, lut_v = build_alignment_lut()
    valid_lut_pct = np.sum(lut_u >= 0) / (DEPTH_WIDTH * DEPTH_HEIGHT) * 100
    print(f"  LUT coverage: {valid_lut_pct:.1f}% of depth pixels map to color FOV")

    # Process frame 0 for verification
    print("\nProcessing frame 0 for verification...")
    depth_m, aligned_color = process_frame(0, map1, map2, lut_u, lut_v)

    # Save aligned frame 0
    out_dir = os.path.join(OUTPUT_DIR, "aligned")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "depth_000.npy"), depth_m.astype(np.float32))
    cv2.imwrite(os.path.join(out_dir, "color_000.png"),
                cv2.cvtColor(aligned_color, cv2.COLOR_RGB2BGR))

    # Generate point cloud for visual verification
    pts, clr = depth_to_points(depth_m, aligned_color)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(clr)

    save_ply(os.path.join(out_dir, "frame_000_aligned.ply"), pts, clr)

    print(f"  3D points: {len(pts)}")
    print(f"  Depth range: [{depth_m[depth_m>0].min():.3f}, {depth_m[depth_m>0].max():.3f}] m")
    print(f"  Color range: [{aligned_color.min()}, {aligned_color.max()}]")

    # Visualize
    print("\nVerification: aligned point cloud for frame 0")
    print("  Pot edges should align with color edges (no color fringing)")
    print("  Close window to continue...")
    visualize_point_clouds(
        [pcd],
        window_name="Step 1 — Aligned Frame 0 (close to confirm)")

    print("\nStep 1 complete. Output: pipeline_output/aligned/")
    print("  depth_000.npy  — 512x424 undistorted depth (meters)")
    print("  color_000.png  — 512x424 registered color")
    print("  frame_000_aligned.ply — 3D point cloud for verification")


if __name__ == "__main__":
    main()
