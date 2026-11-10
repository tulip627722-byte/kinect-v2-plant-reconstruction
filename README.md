# 基于 Kinect V2 的植物三维重建与表型分析

本项目使用 Kinect V2 采集的 36 组多视角 RGB-D 数据，完成深度与彩色图像对齐、点云生成、花盆与植株分割、多帧配准、TSDF/Poisson 重建，以及株高、冠幅、叶面积、叶片数量和叶倾角等表型参数提取。

## 处理流程

1. 深度图去畸变并对齐彩色图
2. RGB-D 数据反投影为彩色点云
3. 使用空间范围、HSV 颜色和人工掩膜分割花盆与植株
4. 对花盆点云进行旋转台初始位姿估计和 Point-to-Plane ICP 配准
5. 对植物点云进行旋转叠加与 ICP 配准
6. 使用 TSDF 融合植物点云
7. 合并花盆与植物结果并执行全局去噪
8. 使用 Poisson 方法生成三角网格
9. 使用 DBSCAN 与 PCA 提取植物表型参数

## 环境要求

- Python 3.9 或更高版本
- NumPy
- OpenCV
- Open3D
- SciPy
- scikit-learn

安装依赖：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

python -m pip install -r requirements.txt
```

## 数据目录

在 `config.py` 中将 `DATA_DIR` 修改为本地数据根目录。默认期望的目录结构如下：

```text
DATA_DIR/
├── color/             # 000.png ~ 035.png，彩色图像
├── depth/             # 000.png ~ 035.png，16 位深度图（毫米）
└── pipeline_output/   # 运行后自动生成的中间结果与最终结果
```

相机内参、深度范围、旋转中心、分割阈值、ICP、TSDF 和法线估计参数也统一位于 `config.py`。使用不同设备或采集方案时应重新标定这些参数。

## 运行方法

各步骤存在前后依赖，请在项目根目录按顺序运行：

```bash
python step1_align.py
python step2_pointcloud.py
python step3_segment.py
python step4_pot.py
python step5_plant_stack.py
python step6_plant_icp.py
python step7_plant_tsdf.py
python step8_fuse.py
python step9_poisson.py
python step10_phenotyping.py
```

如需手工标注花盆和植物区域，可先运行：

```bash
python annotate.py
```

标注窗口快捷键：`p` 选择花盆、`l` 选择植物、`r` 重置、`c` 保存、`n` 下一帧、`q` 退出。

## 主要输出

所有结果写入 `DATA_DIR/pipeline_output/`，主要文件包括：

- `final_pointcloud_clean.ply`：融合并清理后的彩色点云
- `merge_model_clean.obj`：Poisson 重建网格
- `merge_model_clean.ply`：PLY 格式重建网格
- `phenotype.json`：株高、冠幅、叶面积、叶片数量和叶倾角结果
- 各步骤生成的对齐图像、分割点云、配准矩阵和中间模型

## 脚本说明

| 文件 | 作用 |
| --- | --- |
| `config.py` | 路径、相机标定和算法参数 |
| `utils.py` | 图像、点云、网格和 JSON 工具函数 |
| `annotate.py` | 交互式花盆/植株掩膜标注 |
| `step1_align.py` | 深度去畸变及 Depth-to-Color 对齐 |
| `step2_pointcloud.py` | 单帧彩色点云生成 |
| `step3_segment.py` | 花盆与植物分割 |
| `step4_pot.py` | 花盆多帧配准 |
| `step5_plant_stack.py` | 植物旋转叠加基线 |
| `step6_plant_icp.py` | 植物 ICP 精配准 |
| `step7_plant_tsdf.py` | 植物 TSDF 融合 |
| `step8_fuse.py` | 花盆与植物融合及清理 |
| `step9_poisson.py` | Poisson 曲面重建 |
| `step10_phenotyping.py` | 表型参数提取 |

## 说明

- 原始 RGB-D 数据体积较大，未包含在当前代码包中。
- 部分步骤会打开 Open3D 或 OpenCV 可视化窗口，需要图形桌面环境。
- `花瓶3d.docx` 为项目实验报告，包含实验原理、处理过程、结果与误差分析。
- 项目用于课程学习与实验复现。
