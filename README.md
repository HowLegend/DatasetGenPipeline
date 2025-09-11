# DatasetGenPipeline
输入一段第一人称视角拍摄的短视频（2min左右），轨迹和所见的物体映射到2D地图上，生成一张 local map

![local map](dataset/assert/outputs_demo/result_dir/local_map/Waikiki_0000_map.jpg)


## 环境部署
#### PS：
> 相关开源项目的 git clone 已经放入这个项目了，必要的权重数据也已放在对于位置。所以只需要按下面步骤配置好 conda 环境中的依赖就可以了，不用克隆其他仓库和下载其他权重资源。
### 克隆 DatasetGenPipeline
```
git clone https://github.com/HowLegend/DatasetGenPipeline.git
cd DatasetGenPipeline

```
### 创建 conda 环境
使用 python=3.10 以及 torch >= 2.3.1、torchvision>=0.18.1 和 cuda-12.1 以上

根据自己环境来安装特定的 conda 环境

```
conda create -n dgp python=3.10 -y
conda activate dgp
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 部署 Grounded-SAM-2
```
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
```
下载 Segment Anything 2 权重
```
cd Grounded-SAM-2
cd checkpoints
bash download_ckpts.sh
cd ..
```
下载 Grounding DINO 权重
```
cd gdino_checkpoints
bash download_ckpts.sh
cd ..
```
安装 Segment Anything 2
```
pip install -e .
```
安装 Grounding DINO
```
pip install --no-build-isolation -e grounding_dino
```
返回主目录 DatasetGenPipeline
```
cd ..
```
### 部署 Metric3D
```
git clone https://github.com/YvanYin/Metric3D.git
cd Metric3D
```
这里选用的是 Metric3D 的 ViT 模型，原项目中的 requirements_v2.txt 会破坏现有环境，所以我们使用这里准备的 DatasetGenPipeline/requirements_Metric3D.txt
```
# 在主目录 DatasetGenPipeline 下
pip install -r requirements_Metric3D.txt
```
#### 下载权重文件 
|      |       Encoder       |      Decoder      |                                               Link                                                |
|:----:|:-------------------:|:-----------------:|:-------------------------------------------------------------------------------------------------:|
| v1-T |    ConvNeXt-Tiny    | Hourglass-Decoder | [Download 🤗](https://huggingface.co/JUGGHM/Metric3D/blob/main/convtiny_hourglass_v1.pth)        |
| v1-L |   ConvNeXt-Large    | Hourglass-Decoder | [Download](https://drive.google.com/file/d/1KVINiBkVpJylx_6z1lAC7CQ4kmn-RJRN/view?usp=drive_link) |
| v2-S | DINO2reg-ViT-Small  |    RAFT-4iter     | [Download](https://drive.google.com/file/d/1YfmvXwpWmhLg3jSxnhT7LvY0yawlXcr_/view?usp=drive_link) |
| v2-L | DINO2reg-ViT-Large  |    RAFT-8iter     | [Download](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view?usp=drive_link) |
| v2-g | DINO2reg-ViT-giant2 |    RAFT-8iter     | [Download 🤗](https://huggingface.co/JUGGHM/Metric3D/blob/main/metric_depth_vit_giant2_800k.pth) | 

这里我们下载 v2-L 版本
```
cd Metric3D/weight
gdown 1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr
cd ..
```
返回主目录 DatasetGenPipeline
```
cd ..
```
### 部署 DPVO
```
git clone https://github.com/princeton-vl/DPVO.git --recursive
cd DPVO
```
安装 DPVO 依赖

> 虽然 PyTorch 官方提供了 cu124 的 wheel，但 PyTorch Geometric（torch-scatter 的官方发布方）尚未为 cu124 提供预编译包。
>
> 不过，CUDA 是向后兼容的 —— 可以安全地安装为 cu121 编译的 torch-scatter，它也会在 CUDA 12.4 系统上完美运行
```
# 安装 torch-scatter 包
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# 安装其他 pip 包
pip install tensorboard numba tqdm einops pypose kornia numpy plyfile evo opencv-python yacs
```
### 下载 DPVO 资源包
```
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

# install DPVO
pip install .

# download models and data (~2GB)
./download_models_and_data.sh
```
返回主目录 DatasetGenPipeline
```
cd ..
```
### 下载补充 pip 包
```
pip install supervision pycolmap scikit-learn transformers pycocotools
```

## 代码介绍

管道的核心目标是：给定一个视频和一个文本提示，生成一个带有物体分割掩码、深度信息、相机运动轨迹以及物体在 3D 空间中相对位置（距离）的输出视频和可视化地图。

### 如何使用

代码的使用非常直接，主要通过修改 `__main__` 函数中的配置来实现。

1.  **设置输出目录**:
    在 `__main__` 函数顶部，设置 `OUTPUT_DIR` 变量，指定所有处理结果的根目录。
    ```python
    OUTPUT_DIR = "./outputs/output_1"
    ```

2.  **配置视频处理任务**:
    `VIDEO_CONFIGS` 是一个字典列表，每个字典定义了一个待处理视频及其参数。
    ```python
    VIDEO_CONFIGS = [
        {
            "video_path": "./dataset/videos/your_video.mp4", # [必填] 输入视频路径
            "start_second": 0, # 从视频的第几秒开始处理
            "max_seconds": 120, # 最大处理时长（秒），0表示处理整个视频
            "frames_per_second": 1.0, # 采样率，每秒处理多少帧
            "colmap_interval": 2.0, # 用于自动计算相机焦距的帧间隔（秒）
            "max_distance": 7.0, # 可视化地图中显示物体的最大距离（米）
            "text_prompt": "car. person. traffic light.", # [必填] 检测提示，以点分隔不同物体
            "intrinsic_fx_fy": [751.0000, 640.0000] # [可选] 已知相机焦距 [fx, fy]，若提供则跳过计算
        },
        # 可以添加更多视频配置...
    ]
    ```

3.  **配置模型路径**:
    `PIPELINE_PARAMS` 字典指定了所有预训练模型的本地路径。
    ```python
    PIPELINE_PARAMS = {
        "sam2_checkpoint": "./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "grounding_dino_config": "Grounding-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "grounding_dino_checkpoint": "Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
        "metric3d_config": "Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py",
        "metric3d_ckpt": "Metric3D/weight/metric_depth_vit_large_800k.pth"
    }
    ```

4.  **运行代码**:
    直接运行脚本即可。代码支持多进程并行处理多个视频。
    ```bash
    python pipeline.py
    ```

5.  **查看结果**:
    处理完成后，结果会保存在 `OUTPUT_DIR` 下：
    *   **最终视频**: `./outputs/output_1/result_dir/output_video/{video_name}_video.mp4`
    *   **可视化地图**: `./outputs/output_1/result_dir/local_map/{video_name}_map.jpg`
    *   **详细日志**: 每个视频的专属日志文件位于其输出目录内。

### 代码核心流程

`CombinedPipeline.process_video` 是主控函数，其执行流程如下：

1.  **焦距计算 (`_compute_average_focal_lengths`)**:
    *   如果未提供 `intrinsic_fx_fy`，管道会使用 `COLMAP` 对视频的关键帧进行稀疏重建，以估算相机的平均焦距 `(fx, fy)` 和图像中心点 `(cx, cy)`。
    *   这些内参对于后续的 3D 重建和尺度对齐至关重要。

2.  **视频帧提取 (`_extract_limited_frames`)**:
    *   根据 `start_second`, `max_seconds`, 和 `frames_per_second` 参数，从原始视频中提取指定范围和帧率的图像序列，并保存到 `frames` 目录。

3.  **相机轨迹估计 (`run_dpvo`)**:
    *   使用 **DPVO** (Direct Perspective Visual Odometry) 算法处理提取的图像序列。
    *   DPVO 会输出相机在每一帧的 6DoF 位姿（位置和旋转），并保存为 TUM 格式的轨迹文件 `pose.txt`。
    *   同时，DPVO 也会生成一个稀疏的 3D 点云。

4.  **模型初始化 (`initialize_models`, `_initialize_metric3d`)**:
    *   加载并初始化 SAM 2、Grounding DINO 和 Metric3D 模型。

5.  **对象分割与追踪 (`_process_frame_batch`, `_propagate_and_save_masks`)**:
    *   **批处理**: 代码以批次（例如，每 20 帧）处理视频。
    *   **首帧检测**: 在每个批次的起始帧，使用 **Grounding DINO** 根据 `text_prompt` 检测物体，并生成边界框。
    *   **首帧分割**: 使用 **SAM 2 图像预测器**，根据 DINO 的边界框生成精确的物体分割掩码。
    *   **跨帧追踪**: 使用 **SAM 2 视频预测器**，将起始帧的掩码作为提示，自动追踪该物体在后续帧中的位置和形状，并生成掩码序列。

6.  **深度与点云估计 (`do_scalecano_test_with_custom_data_my_designed`)**:
    *   使用 **Metric3D** 模型处理 `frames` 目录中的所有图像。
    *   为每一帧生成高精度的深度图 (`depth_{frame_id}.npy`) 和对应的 3D 点云文件 (`pointcloud_{frame_id}.ply`)。

7.  **尺度对齐 (`align_scale_with_metric3d`)**:
    *   **核心创新**: DPVO 估计的轨迹和点云是“无尺度”的（即单位是任意的），而 Metric3D 的点云是“有尺度”的（单位是米）。
    *   该函数通过比较 DPVO 点云和 Metric3D 点云在多个视角下的深度值，使用 **RANSAC** 算法计算出一个全局的**尺度因子**。
    *   然后，将这个尺度因子应用到 DPVO 的相机轨迹和点云上，使其单位与 Metric3D 一致（米），从而获得真实的物理尺度。

8.  **结果渲染与可视化 (`_draw_masks_with_position`, `generate_visualization`)**:
    *   **逐帧渲染**: 遍历所有帧，将 SAM 2 的分割掩码、边界框以及计算出的物体距离（基于 Metric3D 点云和对齐后的相机位姿）绘制到原始图像上。
    *   **生成视频**: 将渲染后的图像序列合成为最终的输出视频。
    *   **生成地图**: 根据对齐后的相机轨迹和检测到的物体在世界坐标系中的位置，绘制一个 2D 鸟瞰图（局部地图），清晰展示相机路径和物体分布。

### 关键函数详解

*   `set_model_paths(self, ...)`: 设置模型路径，延迟加载。

*   `initialize_models(self, ...)`: 初始化 SAM 2、Grounding DINO 和 Metric3D 模型。注意，为了兼容性，Grounding DINO 被强制使用 FP32 精度。

*   `process_video(self, ...)`: **主入口函数**。协调整个处理流程，从帧提取到最终视频生成。

*   `_compute_average_focal_lengths(self, video_path: str, colmap_interval:float) -> List[float]`:
    使用 COLMAP 自动计算视频的相机内参（焦距）。这是实现无标定相机 3D 重建的关键一步。

*   `run_dpvo(self, ...)`: 运行 DPVO 算法，生成相机运动轨迹和稀疏点云。

*   `align_scale_with_metric3d(...) -> float`:
    **核心函数**。通过比较 DPVO 点云和 Metric3D 点云，计算并返回全局尺度因子，实现无尺度轨迹到真实尺度的转换。

*   `_apply_scale_to_poses(self, scale: float)` 和 `_apply_scale_to_point_cloud(self, scale: float)`:
    将计算出的尺度因子应用到相机位姿和点云数据上。

*   `_draw_masks_with_position(...) -> Dict[int, Dict[str, Any]]`:
    主渲染函数。它加载每一帧的分割掩码、深度点云和相机位姿，计算物体距离，并将所有信息绘制到图像上。它返回一个包含所有物体位置信息的字典，用于生成地图。

*   `generate_visualization(...)`: 生成最终的 2D 可视化地图，展示相机轨迹和检测到的物体。

*   `process_single_video(...)`: 一个独立的包装函数，用于在多进程中安全地处理单个视频，包含独立的日志记录。
