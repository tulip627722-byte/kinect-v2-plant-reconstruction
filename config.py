"""
管线全局配置：所有内参、路径、阈值统一管理。

包含：
- Kinect V2 深度/彩色相机内参（报告实验值）
- 旋转台参数（旋转中心、每帧角度）
- 深度滤波参数
- HSV 分割阈值（绿色/花盆）
- 体素滤波、SOR、ICP、TSDF 等参数
- 法线估计参数
"""
import os
import numpy as np

# ============================================================
# 路径设置
# ============================================================
DATA_DIR = r"E:\鑺辩摱3d"
COLOR_DIR = os.path.join(DATA_DIR, "color")
DEPTH_DIR = os.path.join(DATA_DIR, "depth")
OUTPUT_DIR = os.path.join(DATA_DIR, "pipeline_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_FRAMES = 36

# ============================================================
# Kinect V2 标定参数（报告实验值，fx=365.456）
# 深度相机 512×424，彩色相机 1920×1080
# 外参：深度→彩色 约 52mm X 向基线
# ============================================================

# Depth/IR camera intrinsics (512 x 424)
DEPTH_FX = 365.456
DEPTH_FY = 365.456
DEPTH_CX = 254.878
DEPTH_CY = 205.395
DEPTH_WIDTH = 512
DEPTH_HEIGHT = 424

# Depth camera radial distortion (no tangential)
DEPTH_K1 = 0.0
DEPTH_K2 = 0.0
DEPTH_K3 = 0.0
DEPTH_P1 = 0.0
DEPTH_P2 = 0.0

# Camera matrix and distortion vector for OpenCV
K_DEPTH = np.array([
    [DEPTH_FX, 0, DEPTH_CX],
    [0, DEPTH_FY, DEPTH_CY],
    [0, 0, 1]
], dtype=np.float64)

D_DEPTH = np.array([DEPTH_K1, DEPTH_K2, DEPTH_P1, DEPTH_P2, DEPTH_K3], dtype=np.float64)

# Color camera intrinsics (1920 x 1080)
COLOR_FX = 1081.37
COLOR_FY = 1081.37
COLOR_CX = 959.5
COLOR_CY = 539.5
COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080

K_COLOR = np.array([
    [COLOR_FX, 0, COLOR_CX],
    [0, COLOR_FY, COLOR_CY],
    [0, 0, 1]
], dtype=np.float64)

# Extrinsic: depth camera 鈫?color camera
# Approximate: ~52mm baseline along X, near-identity rotation
R_DEPTH_TO_COLOR = np.eye(3, dtype=np.float64)
T_DEPTH_TO_COLOR = np.array([-0.052, 0.0, 0.0], dtype=np.float64)

# ============================================================
# 旋转台参数：旋转中心、每帧旋转角
# 每次 = 10°，36 帧完成 360° 环形采集
# ============================================================
ROT_CX = -0.0346  # Rotation center X (world coords)
ROT_CZ = 0.7861   # Rotation center Z (world coords)
DEG_PER_FRAME = 360.0 / NUM_FRAMES  # 10 degrees

# ============================================================
# 深度滤波：有效范围 0.3~1.5m，双边滤波参数
# ============================================================
DEPTH_MIN_M = 0.3   # Minimum valid depth (meters)
DEPTH_MAX_M = 1.5   # Maximum valid depth (meters)
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 30

# ============================================================
# 分割阈值
# 花盆：Y∈[0.08,0.20]m（花盆高度范围）
# 植物：Y<0.10m + HSV 绿色检测
# 绿色 HSV：H=[35°,85°], S>30%
# ============================================================
# Pot
POT_Y_MIN = 0.08   # meters (above ground)
POT_Y_MAX = 0.20   # meters
POT_DEPTH_MIN = 0.3
POT_DEPTH_MAX = 1.5

# Plant
PLANT_Y_MAX = 0.10  # meters (below this = plant)

# HSV thresholds for green (OpenCV: H鈭圼0,180], S鈭圼0,255], V鈭圼0,255])
# H=35掳鈫?7 in OpenCV half-range, H=85掳鈫?2
GREEN_H_MIN = 17    # 35掳 / 2
GREEN_H_MAX = 42    # 85掳 / 2
GREEN_S_MIN = 77    # 0.3 * 255 鈮?77

# ============================================================
# 体素下采样 & 统计离群点去除
# VOXEL_SIZE=3mm 控制点云密度
# 花盆 SOR 更严格（nb=16），植物更宽松（nb=8）
# ============================================================
VOXEL_SIZE = 0.003  # 3 mm

# Pot SOR (strict)
POT_SOR_NB = 16
POT_SOR_STD = 2.0

# Plant SOR (loose)
PLANT_SOR_NB = 8
PLANT_SOR_STD = 3.0

# ============================================================
# ICP 配准参数
# 花盆（Point-to-Plane）：阈值 5cm，迭代 50 次
# 植物（Point-to-Point）：阈值 3cm，fitness>0.2 且 RMSE<0.01
# ============================================================
# Pot ICP
POT_ICP_THRESHOLD = 0.05  # meters
POT_ICP_MAX_ITER = 50

# Plant ICP
PLANT_ICP_THRESHOLD = 0.03  # meters
PLANT_ICP_MAX_ITER = 50
PLANT_ICP_MIN_FITNESS = 0.2
PLANT_ICP_MAX_RMSE = 0.01

# ============================================================
# TSDF 体素融合参数
# voxel=3mm 控制重建精度，trunc=15mm 控制融合带宽度
# ============================================================
TSDF_VOXEL = 0.003     # 3 mm
TSDF_TRUNC = 0.015     # 15 mm
TSDF_DEPTH_MIN = 0.3   # meters
TSDF_DEPTH_MAX = 1.5   # meters

# ============================================================
# Outlier removal (plant per-frame before TSDF)
# ============================================================
PLANT_RADIUS_NB = 5
PLANT_RADIUS_R = 0.01  # 10 mm

# ============================================================
# Taubin 平滑迭代次数
# ============================================================
TAUBIN_ITERATIONS = 3

# ============================================================
# 法线估计参数（半径 10mm，最多 30 近邻）
# ============================================================
NORMAL_RADIUS = 0.01   # 10 mm
NORMAL_MAX_NN = 30
