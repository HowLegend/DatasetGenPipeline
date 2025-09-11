# 在 import 语句之前添加
import os
# os.environ["HF_HOME"] = "/your/path/to/huggingface_cache"  # 指定本地缓存路径
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 离线模式
os.environ["HF_DATASETS_OFFLINE"] = "1"

############################################
# 基本导入
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import os.path as osp
import sys
import cv2
import torch
import numpy as np
import json
import math
from PIL import Image
import supervision as sv
import copy
import re
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchvision.ops import box_convert
import time
from typing import List, Tuple, Dict, Any, Optional
import pycolmap
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.transform import Rotation
import multiprocessing
import traceback

# 我的导入
from my_utils.video_utils import create_video_from_images
from my_utils.do_test import do_scalecano_test_with_custom_data_my_designed
from my_utils.plot_utils import save_ply_my_designed

# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录：当前脚本所在目录的上一级
project_root = os.path.dirname(current_dir)
# DPVO相关导入
dpvo_dir = os.path.abspath(os.path.join(project_root, "DPVO"))
sys.path.append(dpvo_dir)
from multiprocessing import Process, Queue
from DPVO.dpvo.config import cfg
from DPVO.dpvo.dpvo import DPVO
from DPVO.dpvo.plot_utils import plot_trajectory, save_output_for_COLMAP
from DPVO.dpvo.stream import image_stream, video_stream
from DPVO.dpvo.utils import Timer
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from plyfile import PlyData, PlyElement

# Grounded-SAM-2相关导入
gsam2_dir1 = os.path.abspath(os.path.join(current_dir, "Grounded-SAM-2"))
gsam2_dir2 = os.path.abspath(os.path.join(project_root, "Grounded-SAM-2"))
sys.path.insert(0, gsam2_dir1)
sys.path.insert(0, gsam2_dir2)
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict  # 使用本地函数
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

# Metric3D相关导入
metric3d_dir = os.path.abspath(os.path.join(project_root, "Metric3D"))
sys.path.append(metric3d_dir)
try:
    from mmcv.utils import Config
except:
    from mmengine import Config

import mmcv
import torch
import logging
import torch.distributed as dist
from mono.utils.logger import setup_logger
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.custom_data import load_data
from mono.utils.mldb import load_data_info, reset_ckpt_path
from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.visualization import save_val_imgs, create_html, save_raw_imgs, save_normal_val_imgs
from mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud

# 主类
class CombinedPipeline:
    """整合SAM2、Grounding DINO和Metric3D的视频处理管道"""
    
    def __init__(self, pose_file: Optional[str] = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 初始化一个默认的日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # 如果没有处理器，添加一个基本的控制台处理器
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info("Using device: %s", self.device)
        
        self.pose_file = pose_file
        self.poses = []  # 存储解析后的位姿数据
        
        if pose_file and os.path.exists(pose_file):
            self.logger.info(f"Loading poses from: {pose_file}")
            self.poses = self._parse_pose_file(pose_file)
            self.logger.info(f"Loaded {len(self.poses)} poses")
        
        # 模型和配置占位符
        self.model = None
        self.cfg = None
        self.transform = Compose([])  # 空转换
        
        # 焦距相关参数
        self.intrinsic_fx_fy = None  # 焦距参数
        self.fx_values = None
        self.fy_values = None
        self.avg_fx = None
        self.avg_fy = None
        self.cx = None  # 图像中心点x坐标
        self.cy = None  # 图像中心点y坐标
        self.frame_width = None  # 图像宽度
        self.frame_height = None  # 图像高度
        
        # 初始化标志
        self.metric3d_initialized = False
        
        # 模型路径存储
        self.sam2_checkpoint = None
        self.model_cfg = None
        self.grounding_dino_config = None
        self.grounding_dino_checkpoint = None
        self.metric3d_config = None
        self.metric3d_ckpt = None
        
        # 添加类别计数器
        self.class_counter = {}  # 用于跟踪每个类别的计数器

    def set_model_paths(
        self,
        sam2_checkpoint: str, 
        model_cfg: str, 
        grounding_dino_config: str,
        grounding_dino_checkpoint: str,
        metric3d_config: str, 
        metric3d_ckpt: str
    ):
        """设置模型路径，但不立即初始化"""
        self.sam2_checkpoint = sam2_checkpoint
        self.model_cfg = model_cfg
        self.grounding_dino_config = grounding_dino_config
        self.grounding_dino_checkpoint = grounding_dino_checkpoint
        self.metric3d_config = metric3d_config
        self.metric3d_ckpt = metric3d_ckpt

    def _parse_pose_file(self, pose_file_path: str) -> List[Dict]:
        """解析位姿文件，格式为：时间戳 X Y Z qx qy qz qw"""
        poses = []
        with open(pose_file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) < 8:
                    continue
                    
                try:
                    pose = {
                        'timestamp': float(parts[0]),
                        'position': [float(parts[1]), float(parts[2]), float(parts[3])],
                        'rotation': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                    }
                    poses.append(pose)
                except ValueError:
                    continue
        
        return poses

    def _compute_average_focal_lengths(self, video_path: str, colmap_interval:float) -> List[float]:
        """使用COLMAP计算视频的平均焦距(fx, fy)"""
        self.temp_colmap_dir = "./temp_colmap_frames"
        os.makedirs(self.temp_colmap_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error("无法打开视频文件 %s", video_path)
            raise ValueError(f"无法打开视频文件 {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * colmap_interval)
        
        count = 0
        extracted_frames = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                frame_path = os.path.join(self.temp_colmap_dir, f"frame_{extracted_frames:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                extracted_frames += 1
                
            count += 1
        
        cap.release()
        self.logger.info(f"提取了 {extracted_frames} 个关键帧用于焦距计算")
        
        image_dir = Path(self.temp_colmap_dir)
        output_path = Path("./temp_colmap_reconstruction")
        output_path.mkdir(exist_ok=True)
        
        database_path = output_path / "database.db"
        pycolmap.extract_features(database_path, image_dir)
        pycolmap.match_exhaustive(database_path)
        reconstructions = pycolmap.incremental_mapping(database_path, image_dir, output_path)
        
        fx_values = []
        fy_values = []
        
        if reconstructions:
            rec = reconstructions[0]
            for camera_id, camera in rec.cameras.items():
                params = camera.params
                fx = params[0]
                fy = params[1] if len(params) > 1 else params[0]
                fx_values.append(fx)
                fy_values.append(fy)
        
        if fx_values and fy_values:
            avg_fx = np.mean(fx_values)
            avg_fy = np.mean(fy_values)
            self.logger.info("计算的平均焦距: fx=%.4f, fy=%.4f", avg_fx, avg_fy)
        else:
            avg_fx, avg_fy = 1000.0000, 1000.0000
            self.logger.warning("焦距计算失败，使用默认值: fx=%.4f, fy=%.4f", avg_fx, avg_fy)
        
        self.fx_values = fx_values
        self.fy_values = fy_values
        self.avg_fx = avg_fx
        self.avg_fy = avg_fy
        
        try:
            import shutil
            shutil.rmtree(self.temp_colmap_dir, ignore_errors=True)
            shutil.rmtree(output_path, ignore_errors=True)
            self.logger.info("清理焦距计算临时文件")
        except:
            self.logger.warning("警告: 无法清理焦距计算临时文件")
        
        return [avg_fx, avg_fy]
    
    def _get_first_frame_dimensions(self, frames_dir: str):
        """获取第一帧的尺寸并计算中心点"""
        frame_names = sorted([f for f in os.listdir(frames_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not frame_names:
            self.logger.warning("警告: 未找到提取的帧")
            return
            
        first_frame_path = os.path.join(frames_dir, frame_names[0])
        try:
            with Image.open(first_frame_path) as img:
                self.frame_width, self.frame_height = img.size
                self.cx = self.frame_width / 2.0
                self.cy = self.frame_height / 2.0
                self.logger.info(f"获取第一帧尺寸: width={self.frame_width}, height={self.frame_height}")
                self.logger.info(f"计算中心点: cx={self.cx:.2f}, cy={self.cy:.2f}")
        except Exception as e:
            self.logger.info(f"无法获取第一帧尺寸: {e}")
            # 使用默认值
            self.frame_width = 640
            self.frame_height = 480
            self.cx = self.frame_width / 2.0
            self.cy = self.frame_height / 2.0
    
    def _write_intrinsic_file(self, intrinsic_file: str):
        """将内参参数写入文件（包含焦距和中心点）"""
        with open(intrinsic_file, "w") as f:
            f.write("COLMAP计算的内参参数\n")
            f.write("=====================\n\n")
            
            if self.fx_values and self.fy_values:
                f.write("各相机焦距参数:\n")
                for i, (fx, fy) in enumerate(zip(self.fx_values, self.fy_values)):
                    f.write(f"相机 {i+1}: fx = {fx:.4f}, fy = {fy:.4f}\n")
                
                f.write("\n平均焦距:\n")
                f.write(f"fx = {self.avg_fx:.4f}, fy = {self.avg_fy:.4f}\n")
            else:
                f.write("警告: 焦距计算失败，使用默认值\n")
                f.write(f"fx = {self.avg_fx:.4f}, fy = {self.avg_fy:.4f}\n")
            
            # 添加中心点信息
            f.write("\n图像中心点:\n")
            f.write(f"cx = {self.cx:.4f}\n")
            f.write(f"cy = {self.cy:.4f}\n")
            
            # 添加图像尺寸信息
            f.write("\n图像尺寸:\n")
            f.write(f"width = {self.frame_width}\n")
            f.write(f"height = {self.frame_height}\n")
        
        self.logger.info(f"焦距和中心点信息已保存至: {intrinsic_file}")
    
    def _create_dpvo_calibration_file(self, video_output_dir: str):
        """创建dpvo/calib/camera.txt文件并写入内参参数"""
        # 创建dpvo目录
        dpvo_dir = os.path.join(video_output_dir, "dpvo")
        os.makedirs(dpvo_dir, exist_ok=True)
        
        # 创建calib目录
        calib_dir = os.path.join(dpvo_dir, "calib")
        os.makedirs(calib_dir, exist_ok=True)
        
        # 创建camera.txt文件
        camera_file = os.path.join(calib_dir, "camera.txt")
        
        with open(camera_file, "w") as f:
            f.write(f"{self.avg_fx:.4f} {self.avg_fy:.4f} {self.cx:.4f} {self.cy:.4f}")
        
        self.logger.info(f"dpvo相机参数已保存至: {camera_file}")
        self.logger.info(f"参数: fx={self.avg_fx:.4f}, fy={self.avg_fy:.4f}, cx={self.cx:.4f}, cy={self.cy:.4f}")

    def run_dpvo(
        self,
        imagedir: str,
        calib_file: str,
        output_dir: str,
        name: str = "result",
        config: str = "config/default.yaml",
        stride: int = 1,
        skip: int = 0,
        timeit: bool = False,
        viz: bool = False,
        plot: bool = True,
        save_trajectory: bool = True,
        need_save_ply: bool = True,
        save_colmap: bool = False,
        opts: List[str] = []
    ) -> str:
        """
        运行DPVO算法生成相机轨迹
        
        Args:
            imagedir: 包含输入帧的目录
            calib_file: 相机标定文件路径
            output_dir: 输出目录
            name: 运行名称（用于输出文件名）
            config: DPVO配置文件路径
            stride: 帧步长
            skip: 跳过的帧数
            timeit: 是否计时
            viz: 是否可视化
            plot: 是否绘制轨迹图
            save_trajectory: 是否保存轨迹文件
            need_save_ply: 是否保存点云
            save_colmap: 是否保存COLMAP格式输出
            opts: 额外配置选项
            
        Returns:
            生成的轨迹文件路径
        """
        # 在函数内定义运行函数
        @torch.no_grad()
        def run(cfg, network, imagedir, calib, stride=1, skip=0, viz=False, timeit=False):

            slam = None
            queue = Queue(maxsize=8)

            if os.path.isdir(imagedir):
                reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, skip))
            else:
                reader = Process(target=video_stream, args=(queue, imagedir, calib, stride, skip))

            reader.start()

            while 1:
                (t, image, intrinsics) = queue.get()
                if t < 0: break

                image = torch.from_numpy(image).permute(2,0,1).cuda()
                intrinsics = torch.from_numpy(intrinsics).cuda()

                if slam is None:
                    _, H, W = image.shape
                    slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

                with Timer("SLAM", enabled=timeit):
                    slam(t, image, intrinsics)

            reader.join()

            points = slam.pg.points_.cpu().numpy()[:slam.m]
            colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

            return slam.terminate(), (points, colors, (*intrinsics, H, W))

        # 创建输出目录
        dpvo_output_dir = os.path.join(output_dir, "dpvo")
        os.makedirs(dpvo_output_dir, exist_ok=True)
        
        # 创建轨迹目录
        trajectory_dir = os.path.join(dpvo_output_dir, "trajectory")
        os.makedirs(trajectory_dir, exist_ok=True)
        
        # 设置参数
        class Args:
            def __init__(self):
                self.network = 'DPVO/dpvo.pth'
                self.imagedir = imagedir
                self.calib = calib_file
                self.name = name
                self.stride = stride
                self.skip = skip
                self.config = config
                self.timeit = timeit
                self.viz = viz
                self.plot = plot
                self.need_save_ply = need_save_ply
                self.save_colmap = save_colmap
                self.save_trajectory = save_trajectory
                self.opts = opts
        
        args = Args()
        
        # 合并配置
        cfg.merge_from_file(args.config)
        cfg.merge_from_list(args.opts)
        
        self.logger.info("Running DPVO with config...")
        self.logger.info(cfg)
        
        # 运行DPVO
        (poses, tstamps), (points, colors, calib) = run(
            cfg, args.network, args.imagedir, args.calib, 
            args.stride, args.skip, args.viz, args.timeit
        )
        
        # 创建轨迹对象
        trajectory = PoseTrajectory3D(
            positions_xyz=poses[:,:3], 
            orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], 
            timestamps=tstamps
        )
        
        # 保存轨迹文件
        trajectory_file = os.path.join(trajectory_dir, f"{args.name}_pose.txt")
        if args.save_trajectory:
            file_interface.write_tum_trajectory_file(trajectory_file, trajectory)
            self.logger.info(f"轨迹文件已保存至: {trajectory_file}")
        
        # 绘制轨迹图
        plot_file = os.path.join(trajectory_dir, f"{args.name}_picture.pdf")
        if args.plot:
            plot_trajectory(trajectory, 
                            title=f"DPVO Trajectory Prediction for {args.name}", 
                            filename=plot_file)
            self.logger.info(f"轨迹图已保存至: {plot_file}")
        
        # 在运行DPVO后保存点云数据
        if args.need_save_ply:
            # 创建ply_file目录
            ply_dir = os.path.join(dpvo_output_dir, "ply_file")
            os.makedirs(ply_dir, exist_ok=True)
            
            ply_path = save_ply_my_designed(args.name, points, colors, ply_dir)
            # 保存点云数据到实例变量
            self.dpvo_point_cloud = points  # 保存点云
        
        if args.save_colmap:
            save_output_for_COLMAP(args.name, trajectory, points, colors, *calib)
        
        return trajectory_file

    def align_scale_with_metric3d(self, frames_dir: str, depth_data_dir: str, dpvo_point_cloud: np.ndarray, poses: List[Dict]) -> float:
        """
        使用Metric3D点云校正DPVO的尺度
        
        Args:
            frames_dir: 帧图像目录
            depth_data_dir: Metric3D点云目录
            dpvo_point_cloud: DPVO生成的点云 (N,3)
            poses: 位姿列表
            
        Returns:
            全局尺度因子
        """
        if not poses or dpvo_point_cloud.size == 0:
            self.logger.info("Warning: No poses or point cloud available for scale alignment")
            return 1.0
            
        self.logger.info("Starting scale alignment with Metric3D point clouds...")
        
        # 准备内参矩阵
        K = np.array([
            [self.avg_fx, 0, self.cx],
            [0, self.avg_fy, self.cy],
            [0, 0, 1]
        ])
        
        frame_names = self._get_sorted_frame_names(frames_dir)
        scale_factors = []
        
        # 统计计数器
        stats = {
            "total_frames": len(frame_names),
            "no_point_cloud": 0,
            "no_pose": 0,
            "no_points_behind_camera": 0,
            "no_points_in_view": 0,
            "not_enough_points": 0,
            "ransac_failed": 0,
            "success": 0
        }
        
        # 缓存Metric3D点云数据
        metric3d_point_clouds = {}
        
        for frame_name in tqdm(frame_names, desc="Scale Alignment"):
            frame_base = os.path.splitext(frame_name)[0]
            ply_path = os.path.join(depth_data_dir, f"pointcloud_{frame_base}.ply")
            
            if not os.path.exists(ply_path):
                stats["no_point_cloud"] += 1
                self.logger.info(f"Frame {frame_name}: Point cloud not found: {ply_path}")
                continue
                
            # 加载Metric3D点云
            if ply_path not in metric3d_point_clouds:
                try:
                    points, _ = self._get_ply_point_data(ply_path)
                    metric3d_point_clouds[ply_path] = points
                except Exception as e:
                    self.logger.info(f"Frame {frame_name}: Failed to load point cloud - {e}")
                    stats["no_point_cloud"] += 1
                    continue
                    
            points_metric3d = metric3d_point_clouds[ply_path]
            
            # 获取帧索引（假设帧名是索引）
            try:
                frame_index = int(frame_base)
            except ValueError:
                self.logger.info(f"Frame {frame_name}: Invalid frame index '{frame_base}', skipping")
                continue
                
            # 使用帧索引查找位姿
            pose = self._find_closest_pose(frame_index)
            if not pose:
                stats["no_pose"] += 1
                self.logger.info(f"Frame {frame_name}: No pose found for index {frame_index}")
                continue
                
            # 获取当前帧的位姿变换矩阵 (从世界坐标系到相机坐标系)
            T_wc = self._pose_to_matrix(pose)
            T_cw = np.linalg.inv(T_wc)  # 相机坐标系到世界坐标系的变换
            
            # 将全局点云变换到当前相机坐标系
            points_world = np.hstack((dpvo_point_cloud, np.ones((dpvo_point_cloud.shape[0], 1))))  # 齐次坐标
            points_camera = (T_cw @ points_world.T).T[:, :3]  # 非齐次坐标
            
            # 过滤掉相机后面的点 (z <= 0)
            valid_mask = points_camera[:, 2] > 0
            points_camera = points_camera[valid_mask]
            
            if len(points_camera) == 0:
                stats["no_points_behind_camera"] += 1
                self.logger.info(f"Frame {frame_name}: No points behind camera")
                continue
            
            # 将点投影到图像平面
            points_proj = (K @ points_camera.T).T
            u = points_proj[:, 0] / points_proj[:, 2]  # u = (fx * X/Z + cx)
            v = points_proj[:, 1] / points_proj[:, 2]  # v = (fy * Y/Z + cy)
            z_dpvo = points_camera[:, 2]  # DPVO深度 (相机坐标系下的Z值)
            
            # 过滤投影到图像外的点
            in_bounds = (u >= 0) & (u < self.frame_width) & (v >= 0) & (v < self.frame_height)
            u = u[in_bounds].astype(int)
            v = v[in_bounds].astype(int)
            z_dpvo = z_dpvo[in_bounds]
            
            if len(z_dpvo) == 0:
                stats["no_points_in_view"] += 1
                self.logger.info(f"Frame {frame_name}: No points in view")
                continue
            
            # 从Metric3D点云获取对应位置的深度
            z_metric3d = []
            for i in range(len(u)):
                point = self._find_nearest_point(points_metric3d, u[i], v[i], self.frame_width)
                if point is not None:
                    # 计算点到相机的距离
                    distance = np.linalg.norm(point)
                    z_metric3d.append(distance)
            
            if len(z_metric3d) == 0:
                stats["no_points_in_view"] += 1
                self.logger.info(f"Frame {frame_name}: No points matched in Metric3D point cloud")
                continue
                
            z_metric3d = np.array(z_metric3d)
            
            # 在align_scale_with_metric3d方法中增加点云匹配要求
            if len(z_dpvo) < 10:
                stats["not_enough_points"] += 1
                self.logger.info(f"Frame {frame_name}: Not enough valid points ({len(z_dpvo)}) for scale estimation")
                continue
                
            # 打印一些深度统计信息
            self.logger.info(f"Frame {frame_name}: Valid points: {len(z_dpvo)}")
            self.logger.info(f"  DPVO depth range: [{np.min(z_dpvo):.2f}, {np.max(z_dpvo):.2f}]")
            self.logger.info(f"  Metric3D depth range: [{np.min(z_metric3d):.2f}, {np.max(z_metric3d):.2f}]")
            
            # 使用RANSAC估计鲁棒的尺度因子 (线性回归 y = s * x)
            X = z_dpvo.reshape(-1, 1)
            y = z_metric3d
            
            try:
                ransac = RANSACRegressor()
                ransac.fit(X, y)
                scale_factor = ransac.estimator_.coef_[0]
                inliers = np.sum(ransac.inlier_mask_)
                self.logger.info(f"Frame {frame_name}: Estimated scale factor = {scale_factor:.4f}, inliers: {inliers}/{len(X)}")
                
                # 检查尺度因子是否合理
                if 0.1 < scale_factor < 10.0:
                    scale_factors.append(scale_factor)
                    stats["success"] += 1
                else:
                    stats["ransac_failed"] += 1
                    self.logger.info(f"  Warning: Unreasonable scale factor ({scale_factor:.4f}), discarding")
            except Exception as e:
                stats["ransac_failed"] += 1
                self.logger.info(f"Frame {frame_name}: RANSAC failed - {e}")
                
                # 输出前10个点对的值帮助调试
                self.logger.info("  First 10 point pairs:")
                for i in range(min(10, len(z_dpvo))):
                    self.logger.info(f"    DPVO: {z_dpvo[i]:.2f}, Metric3D: {z_metric3d[i]:.2f}, ratio: {z_metric3d[i]/z_dpvo[i]:.4f}")
        
        # 打印统计信息
        self.logger.info("\nScale alignment statistics:")
        self.logger.info(f"  Total frames processed: {stats['total_frames']}")
        self.logger.info(f"  Frames without point cloud: {stats['no_point_cloud']}")
        self.logger.info(f"  Frames without pose: {stats['no_pose']}")
        self.logger.info(f"  Frames with no points behind camera: {stats['no_points_behind_camera']}")
        self.logger.info(f"  Frames with no points in view: {stats['no_points_in_view']}")
        self.logger.info(f"  Frames with not enough points: {stats['not_enough_points']}")
        self.logger.info(f"  Frames with RANSAC failed: {stats['ransac_failed']}")
        self.logger.info(f"  Frames with successful estimation: {stats['success']}")
        
        # 计算全局尺度因子 (所有帧的中位数)
        if scale_factors:
            global_scale = np.median(scale_factors)
            self.logger.info(f"Global scale factor (median): {global_scale:.4f}")
            return global_scale
        else:
            self.logger.info("Warning: No valid scale factors estimated")
            return 1.0

    def _pose_to_matrix(self, pose: Dict) -> np.ndarray:
        """将位姿字典转换为4x4变换矩阵"""
        # 从四元数创建旋转矩阵
        q = pose['rotation']  # [qx, qy, qz, qw]
        R = Rotation.from_quat(q).as_matrix()
        
        # 创建变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pose['position']
        return T

    def initialize_models(
        self, 
        sam2_checkpoint: str, 
        model_cfg: str, 
        grounding_dino_config: str,
        grounding_dino_checkpoint: str,
        metric3d_config: str, 
        metric3d_ckpt: str
    ):
        """
        初始化所有需要的模型
        """
        # 仅对SAM2和Metric3D启用混合精度，Grounding DINO使用FP32
        if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # 初始化SAM2视频预测器（使用混合精度）
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
            
            # 初始化SAM2图像预测器
            sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
            self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        # 初始化Grounding DINO模型 - 使用FP32精度
        self.logger.info(f"Loading Grounding DINO model from local files:")
        self.logger.info(f"  Config: {grounding_dino_config}")
        self.logger.info(f"  Checkpoint: {grounding_dino_checkpoint}")
        
        # 显式使用FP32精度
        with torch.no_grad():
            self.grounding_model = load_model(
                model_config_path=grounding_dino_config,
                model_checkpoint_path=grounding_dino_checkpoint,
                device=self.device
            ).float()  # 确保模型使用FP32

        # 保存Metric3D配置和权重路径
        self.metric3d_config = metric3d_config
        self.metric3d_ckpt = metric3d_ckpt
        self.metric3d_initialized = False

    def _initialize_metric3d(self, test_data_path: str, depth_data_dir: str):
        """初始化Metric3D模型，直接使用最终输出目录"""
        if self.metric3d_initialized:
            self.logger.info("Metric3D already initialized. Skipping re-initialization.")
            return
            
        self.logger.info(f"Loading Metric3D model with test_data_path: {test_data_path}...")
        
        # 直接使用最终输出目录
        self.depth_data_dir = depth_data_dir
        os.makedirs(self.depth_data_dir, exist_ok=True)
        self.logger.info(f"Using depth output directory: {self.depth_data_dir}")
        
        # 创建虚拟参数解析器
        class Args:
            def __init__(self):
                self.config = 'None'
                self.show_dir = 'None'
                self.load_from = 'None'
                self.node_rank = 0
                self.nnodes = 1
                self.options = None
                self.launcher = 'None'
                self.test_data_path = test_data_path
                self.batch_size = 1
                self.timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        
        args = Args()
        args.config = self.metric3d_config
        args.show_dir = self.depth_data_dir  # 直接使用最终目录
        args.load_from = self.metric3d_ckpt
        
        # 初始化 Metric3D 环境
        cfg = Config.fromfile(args.config)
        cfg.load_from = args.load_from
        cfg.batch_size = args.batch_size
        cfg.show_dir = args.show_dir
        
        # 创建输出目录
        os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)

        # 初始化日志
        cfg.log_file = osp.join(cfg.show_dir, f'{args.timestamp}.log')
        logger = setup_logger(cfg.log_file)
        logger.info(f'Config:\n{cfg.pretty_text}')

        # 加载数据信息
        data_info = {}
        load_data_info('data_info', data_info=data_info)
        cfg.mldb_info = data_info
        
        # 更新检查点路径
        reset_ckpt_path(cfg.model, data_info)
        
        # 配置分布式环境
        if args.launcher == 'None':
            cfg.distributed = False
        else:
            cfg.distributed = True
            init_env(args.launcher, cfg)
        logger.info(f'Distributed training: {cfg.distributed}')
        
        # 保存配置
        cfg.dump(osp.join(cfg.show_dir, osp.basename(args.config)))

        # 添加数据
        self.test_data = load_data(args.test_data_path)

        # 构建模型
        model = get_configured_monodepth_model(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model = torch.nn.DataParallel(model)
        model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
        model.eval()

        # 保存模型和配置
        self.local_rank = 0
        self.logger = logger
        self.distributed = cfg.distributed
        self.batch_size = cfg.batch_size
        self.metric3d_cfg = cfg
        self.metric3d_model = model
        
        self.logger.info("Metric3D model loaded successfully.")
        self.metric3d_initialized = True

    def _generate_local_map(
        self,
        video_output_dir: str,
        object_positions: Dict[str, List[Tuple[float, float]]]
    ):
        """
        生成局部地图并保存
        """
        if not self.poses:
            self.logger.info("Warning: No poses available. Cannot generate local map.")
            return
            
        self.logger.info("Generating local map...")
        
        # 创建地图画布
        plt.figure(figsize=(12, 10))
        
        # 绘制相机轨迹
        camera_x = [pose['position'][0] for pose in self.poses]
        camera_z = [pose['position'][2] for pose in self.poses]  # 使用Z作为垂直轴
        
        plt.plot(camera_x, camera_z, 'b-', linewidth=2, alpha=0.7, label='Camera Trajectory')
        plt.scatter(camera_x, camera_z, c=range(len(camera_x)), cmap='viridis', s=30, alpha=0.8)
        
        # 添加起点和终点标记
        plt.scatter(camera_x[0], camera_z[0], s=200, marker='*', color='green', edgecolors='black', label='Start')
        plt.scatter(camera_x[-1], camera_z[-1], s=200, marker='s', color='red', edgecolors='black', label='End')
        
        # 绘制检测到的物体
        color_map = cm.get_cmap('tab10')
        for obj_idx, (obj_class, positions) in enumerate(object_positions.items()):
            obj_x = [pos[0] for pos in positions]
            obj_z = [pos[1] for pos in positions]
            
            if obj_x and obj_z:
                color = color_map(obj_idx % 10)
                plt.scatter(obj_x, obj_z, color=color, s=80, alpha=0.8, label=obj_class)
                
                # 添加物体类别的文本标签（只添加一次）
                if positions:
                    plt.text(obj_x[0], obj_z[0], obj_class, 
                            fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        
        # 设置图表属性
        plt.title('Local Map with Camera Trajectory and Detected Objects', fontsize=14)
        plt.xlabel('X Position (meters)', fontsize=12)
        plt.ylabel('Z Position (meters)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        # 保存局部地图
        local_map_path = os.path.join(video_output_dir, "local_map.jpg")
        plt.savefig(local_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Local map saved to: {local_map_path}")

    def generate_visualization(
        self,
        output_path: str,
        object_positions: Dict[str, Dict[str, Any]],
        self_logger: Any,
        max_distance: float = 7.0,
    ):
        """
        生成轨迹和检测物体的可视化图片
        
        Args:
            output_path: 输出图片路径
            object_positions: 物体位置字典 {物体ID: {'class_name': 类别, 'original_id': 原始ID, 'positions': [位置信息]}}
            self_logger: 日志记录器
            max_distance: 最大考虑距离（米）
        """
        if not self.poses:
            self_logger.info("Warning: No poses available. Cannot generate visualization.")
            return
            
        self_logger.info(f"Generating visualization map to: {output_path}")
        
        # 创建地图画布
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # 绘制相机轨迹（使用世界坐标系）
        camera_x = [pose['position'][0] for pose in self.poses]
        camera_z = [pose['position'][2] for pose in self.poses]
        
        ax.plot(camera_x, camera_z, 'b-', linewidth=2, alpha=0.7, label='Camera Trajectory')
        scatter = ax.scatter(camera_x, camera_z, c=range(len(camera_x)), cmap='viridis', s=30, alpha=0.8)
        
        # 添加起点和终点标记
        ax.scatter(camera_x[0], camera_z[0], s=200, marker='*', color='green', edgecolors='black', label='Start')
        ax.scatter(camera_x[-1], camera_z[-1], s=200, marker='s', color='red', edgecolors='black', label='End')
        
        # 创建扩展的颜色映射
        tab20 = cm.get_cmap('tab20', 20)
        tab20b = cm.get_cmap('tab20b', 20)
        tab20c = cm.get_cmap('tab20c', 20)
        
        all_colors = []
        for i in range(20):
            all_colors.append(tab20(i))
        for i in range(20):
            all_colors.append(tab20b(i))
        for i in range(20):
            all_colors.append(tab20c(i))
        
        # 统计信息
        total_points = 0
        filtered_points = 0
        
        # 创建颜色映射字典
        object_color_map = {}
        
        # 绘制检测到的物体
        for obj_id, obj_data in object_positions.items():
            class_name = obj_data.get('class_name', 'unknown')
            original_id = obj_data.get('original_id', 'unknown')
            positions_list = obj_data.get('positions', [])
            total_points += len(positions_list)
            
            # 使用2D平面距离进行过滤
            filtered_positions = []
            for pos_info in positions_list:
                distance = pos_info.get('distance_2d') or pos_info.get('distance')
                if distance is not None and distance <= max_distance:
                    filtered_positions.append(pos_info)
            filtered_points += len(filtered_positions)
            
            if not filtered_positions:
                continue
                
            composite_key = f"{class_name}_{original_id}"
            
            if composite_key not in object_color_map:
                object_color_map[composite_key] = len(object_color_map) % len(all_colors)
            color_idx = object_color_map[composite_key]
            color = all_colors[color_idx]
            
            # 提取坐标
            obj_x = [pos_info['world_position'][0] for pos_info in filtered_positions if pos_info['world_position'] is not None]
            obj_z = [pos_info['world_position'][2] for pos_info in filtered_positions if pos_info['world_position'] is not None]
            
            # 绘制物体轨迹
            if obj_x and obj_z:
                ax.scatter(obj_x, obj_z, color=color, s=5, alpha=0.8, 
                            label=f"{class_name} (ID: {original_id})")
        
        # 添加统计日志
        self_logger.info(f"可视化过滤统计: 总点数={total_points}, 过滤后点数={filtered_points} (max_distance={max_distance}m)")
        
        # 计算所有点的坐标范围（只包括距离小于max_distance的点）
        all_x = []
        all_z = []
        
        # 相机轨迹点（全部加入，因为相机位置距离为0）
        all_x.extend(camera_x)
        all_z.extend(camera_z)
        
        # 物体点：只加入距离小于max_distance的点
        for obj_data in object_positions.values():
            positions_list = obj_data.get('positions', [])
            for pos_info in positions_list:
                # 检查距离是否小于max_distance
                distance = pos_info.get('distance_2d') or pos_info.get('distance')
                if distance is not None and distance <= max_distance and pos_info['world_position']:
                    all_x.append(pos_info['world_position'][0])
                    all_z.append(pos_info['world_position'][2])
        
        self_logger.info(f"用于计算坐标范围的点数: {len(all_x)}个点")
        
        if all_x and all_z:
            # 计算坐标范围
            min_x, max_x = min(all_x), max(all_x)
            min_z, max_z = min(all_z), max(all_z)
            
            # 记录坐标范围信息
            self_logger.info(f"X轴范围: min_x={min_x:.2f}, max_x={max_x:.2f}, range_x={max_x-min_x:.2f}")
            self_logger.info(f"Z轴范围: min_z={min_z:.2f}, max_z={max_z:.2f}, range_z={max_z-min_z:.2f}")
            
            # 确保坐标范围不为0
            range_x = max_x - min_x
            range_z = max_z - min_z
            
            if range_x == 0:
                range_x = 1.0
                self_logger.info("X轴范围为零，使用默认范围1.0")
            if range_z == 0:
                range_z = 1.0
                self_logger.info("Z轴范围为零，使用默认范围1.0")
                
            # 计算中心点
            center_x = (min_x + max_x) / 2.0
            center_z = (min_z + max_z) / 2.0
            
            # 根据点群分布自适应调整坐标轴范围
            # 使用较小的边距（2%），减少空白区域
            margin_factor = 0.02
            
            # 分别计算X和Z轴的范围
            axis_range_x = range_x * (1 + margin_factor)
            axis_range_z = range_z * (1 + margin_factor)
            
            # 记录最终坐标轴范围
            final_min_x = min_x - margin_factor * range_x
            final_max_x = max_x + margin_factor * range_x
            final_min_z = min_z - margin_factor * range_z
            final_max_z = max_z + margin_factor * range_z
            
            self_logger.info(f"最终X轴范围: {final_min_x:.2f} 到 {final_max_x:.2f}")
            self_logger.info(f"最终Z轴范围: {final_min_z:.2f} 到 {final_max_z:.2f}")
            
            # 设置坐标轴范围，紧密围绕数据点
            ax.set_xlim(final_min_x, final_max_x)
            ax.set_ylim(final_min_z, final_max_z)
            
            # 设置等比例显示
            ax.set_aspect('equal')
        
        # 设置图表属性
        ax.set_title(f'Camera Trajectory and Detected Objects (≤ {max_distance} m)', fontsize=14)
        ax.set_xlabel('X Position (meters)', fontsize=12)
        ax.set_ylabel('Z Position (meters)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 优化legend位置
        n_legend_items = len(ax.get_legend_handles_labels()[1])
        n_cols = max(1, min(3, (n_legend_items + 9) // 10))
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=n_cols)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # 保存图片
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self_logger.info(f"Visualization map saved to: {output_path}")
        return output_path
    
    def process_video(
        self, 
        video_path: str, 
        text_prompt: str, 
        output_dir: str, 
        max_seconds: int = 30,
        frames_per_second: float = 2.0,
        colmap_interval: float = 2.0,
        need_run_dpvo: bool = True,
        max_distance: float = 7.0,
        intrinsic_fx_fy: Optional[List[float]] = None,
        start_second: int = 0,  # 从视频的第几秒开始
        self_logger: Any = None
    ) -> str:
        """
        处理视频的主函数
        
        Args:
            video_path: 输入视频路径
            text_prompt: Grounding DINO的文本提示
            output_dir: 输出目录
            max_seconds: 最大处理秒数
            frames_per_second: 每秒处理的帧数
            colmap_interval: 用于焦距计算的截取帧间隔（秒）
            need_run_dpvo: 是否运行DPVO算法
            max_distance: local map生成时，最大考虑距离（米），用于可视化物体位置
            intrinsic_fx_fy: 可选的内参焦距列表 [fx, fy]，如果提供则跳过COLMAP计算
            start_second: 从视频的第几秒开始
        """

        # 如果提供了内参焦距，直接使用
        if intrinsic_fx_fy is not None and len(intrinsic_fx_fy) == 2:
            self.logger.info(f"使用提供的焦距值: fx={intrinsic_fx_fy[0]:.4f}, fy={intrinsic_fx_fy[1]:.4f}")
            self.intrinsic_fx_fy = intrinsic_fx_fy
            self.fx_values = [intrinsic_fx_fy[0]]
            self.fy_values = [intrinsic_fx_fy[1]]
            self.avg_fx = intrinsic_fx_fy[0]
            self.avg_fy = intrinsic_fx_fy[1]
        else:
            # 计算平均焦距（放在视频处理开始的位置）
            counter = 0
            max_attempts = 3
            
            while counter < max_attempts:
                counter += 1
                self.logger.info(f"进行第 {counter} 次焦距计算...")
                self._compute_average_focal_lengths(video_path, colmap_interval)
                
                if self.avg_fx is not None and self.avg_fx > 0 and self.avg_fx <= 2000:
                    self.logger.info(f"使用计算的平均焦距值: fx={self.avg_fx:.4f}, fy={self.avg_fy:.4f}")
                    break
                else:
                    self.logger.warning(f"警告：第 {counter} 次焦距计算结果无效 ({self.avg_fx}, {self.avg_fy})，重新尝试...")
                    self.avg_fx = None
                    self.avg_fy = None
                    time.sleep(2)  # 等待后重试

            if self.avg_fx is None:
                self.avg_fx, self.avg_fy = 1000.0, 1000.0
                self.logger.info(f"焦距计算失败，使用默认值: fx={self.avg_fx}, fy={self.avg_fy}")
                
            self.intrinsic_fx_fy = [self.avg_fx, self.avg_fy]

        # 创建视频输出目录和相关子目录
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        optimized_dirname = self._generate_output_dirname(video_basename)
        video_output_dir = os.path.join(output_dir, optimized_dirname)
        
        # 创建统一存储目录
        result_dir = os.path.join(output_dir, "result_dir")
        os.makedirs(result_dir, exist_ok=True)
        
        # 创建视频统一存储目录
        output_video_dir = os.path.join(result_dir, "output_video")
        os.makedirs(output_video_dir, exist_ok=True)
        
        # 创建地图统一存储目录
        local_map_dir = os.path.join(result_dir, "local_map")
        os.makedirs(local_map_dir, exist_ok=True)
        
        # 定义最终输出视频路径
        output_video_name = f"{optimized_dirname}_video.mp4"
        final_output_video_path = os.path.join(output_video_dir, output_video_name)
        
        # 定义最终可视化地图路径
        visualization_name = f"{optimized_dirname}_map.jpg"
        final_visualization_path = os.path.join(local_map_dir, visualization_name)
        
        dirs = self._create_output_directories(video_output_dir)
        
        # 创建colmap目录
        colmap_dir = os.path.join(video_output_dir, "colmap")
        os.makedirs(colmap_dir, exist_ok=True)
        intrinsic_file = os.path.join(colmap_dir, "intrinsic.txt")

        # 提取帧并获取第一帧的尺寸
        self.logger.info(f"从第 {start_second} 秒开始提取 {video_path} 的帧到 {dirs['frames']} (前 {max_seconds} 秒，每秒 {frames_per_second} 帧)...")
        # 提取帧并获取帧索引作为时间戳
        frame_indices = self._extract_limited_frames(
            video_path, 
            dirs['frames'], 
            max_seconds,
            frames_per_second,
            start_second
        )
        
        # 获取第一帧图像尺寸（所有帧尺寸相同）
        self._get_first_frame_dimensions(dirs['frames'])
        
        # 将焦距信息写入intrinsic.txt（包含新增的cx和cy）
        self._write_intrinsic_file(intrinsic_file)
        
        # 创建dpvo目录和camera.txt文件
        self._create_dpvo_calibration_file(video_output_dir)
        
        # 运行DPVO算法生成轨迹
        if need_run_dpvo:
            self.logger.info("Running DPVO to generate camera trajectory...")
            calib_file = os.path.join(video_output_dir, "dpvo", "calib", "camera.txt")

            # 在运行DPVO前清理GPU内存
            torch.cuda.empty_cache()
            import gc
            gc.collect()

            trajectory_file = self.run_dpvo(
                imagedir=dirs['frames'],
                calib_file=calib_file,
                output_dir=video_output_dir,
                name=optimized_dirname,
                config="DPVO/config/default.yaml",
                stride=1,
                skip=0,
                timeit=False,
                viz=False,
                plot=True,
                save_trajectory=True,
                need_save_ply=True,
                save_colmap=False
            )
            
            # 使用生成的轨迹文件
            self.pose_file = trajectory_file
            self.poses = self._parse_pose_file(trajectory_file)
            self.logger.info(f"Loaded {len(self.poses)} poses from DPVO trajectory")
        
        # 在DPVO之后初始化模型
        self.logger.info("Initializing models after DPVO...")
        self.initialize_models(
            self.sam2_checkpoint,
            self.model_cfg,
            self.grounding_dino_config,
            self.grounding_dino_checkpoint,
            self.metric3d_config,
            self.metric3d_ckpt
        )
        
        # 初始化Metric3D - 传入深度数据目录
        self._initialize_metric3d(dirs['frames'], dirs['depth_data'])
        
        frame_names = self._get_sorted_frame_names(dirs['frames'])
        step = 20
        
        self.logger.info(f"Will process {len(frame_names)} frames (approx {max_seconds} seconds)")
        
        inference_state = self.video_predictor.init_state(
            video_path=dirs['frames'], 
            offload_video_to_cpu=True, 
            async_loading_frames=True
        )
        
        sam2_masks = MaskDictionaryModel()
        PROMPT_TYPE_FOR_VIDEO = "mask"
        objects_count = 0
        
        total_frames = len(frame_names)
        for start_frame_idx in range(0, total_frames, step):
            self._process_frame_batch(
                start_frame_idx, step, total_frames, frame_names, 
                dirs, text_prompt, inference_state, sam2_masks, 
                objects_count, PROMPT_TYPE_FOR_VIDEO
            )
        
        # 运行Metric3D深度估计
        self.logger.info("Computing depth maps using Metric3D...")
        do_scalecano_test_with_custom_data_my_designed(
            self.metric3d_model, 
            self.metric3d_cfg,
            video_output_dir,
            self.intrinsic_fx_fy,
            self.test_data,
            self.logger,
            self.distributed,
            self.local_rank,
            self.batch_size,
        )
        
        # 在运行Metric3D后添加尺度对齐
        if need_run_dpvo and self.dpvo_point_cloud is not None and self.poses:
            self.logger.info("Running scale alignment with Metric3D point clouds...")
            global_scale = self.align_scale_with_metric3d(
                dirs['frames'],
                dirs['depth_data'],
                self.dpvo_point_cloud,
                self.poses
            )
            
            self.logger.info(f"Applying global scale factor: {global_scale:.4f}")
            self._apply_scale_to_poses(global_scale)
            self._apply_scale_to_point_cloud(global_scale)
            self._save_scaled_trajectory(trajectory_file, global_scale)
        
        self.logger.info("Rendering final output with positions...")
        # 在渲染帧时传递帧索引
        object_positions = self._draw_masks_with_position(
            dirs['frames'], 
            dirs['mask_data'], 
            dirs['json_data'],
            dirs['depth_data'],
            dirs['result'],
            dirs['mask_photo'],
            frame_indices  # 传递帧索引列表
        )
        
        self.logger.info("Creating output video...")
        # 修改：使用新的命名约定和目录
        create_video_from_images(dirs['result'], final_output_video_path, frame_rate=30)
        self.logger.info(f"处理完成！输出视频已保存至：{final_output_video_path}")
        
        # 生成可视化图片（只考虑max_distance米以内的物体）
        if self.poses:
            # 确保 object_positions 是字典类型
            if not isinstance(object_positions, dict):
                self.logger.info(f"Warning: object_positions is not a dictionary, type: {type(object_positions)}")
            else:
                # 修改：传递最终路径给generate_visualization
                self.generate_visualization(
                    final_visualization_path, 
                    object_positions, 
                    self_logger,
                    max_distance
                )
        
        self._cleanup_temp_files()
        
        return final_output_video_path  # 返回新路径

    def _apply_scale_to_poses(self, scale: float):
        """应用尺度因子到位姿"""
        if not self.poses:
            return
            
        self.logger.info("Applying scale to poses...")
        for pose in self.poses:
            pose['position'] = [p * scale for p in pose['position']]
    
    def _apply_scale_to_point_cloud(self, scale: float):
        """应用尺度因子到点云"""
        if self.dpvo_point_cloud is None:
            return
            
        self.logger.info("Applying scale to point cloud...")
        self.dpvo_point_cloud *= scale
    
    def _save_scaled_trajectory(self, trajectory_file: str, scale: float):
        """保存尺度修正后的轨迹文件，并保存全局尺度因子"""
        if not self.poses:
            return
            
        # 获取轨迹文件所在目录
        trajectory_dir = os.path.dirname(trajectory_file)
        
        # 保存修正后的轨迹
        scaled_trajectory_file = os.path.join(trajectory_dir, "trajectory_scaled.txt")
        
        with open(scaled_trajectory_file, 'w') as f:
            for pose in self.poses:
                # TUM格式: timestamp tx ty tz qx qy qz qw
                line = f"{pose['timestamp']} {pose['position'][0]} {pose['position'][1]} {pose['position'][2]} "
                line += f"{pose['rotation'][0]} {pose['rotation'][1]} {pose['rotation'][2]} {pose['rotation'][3]}\n"
                f.write(line)
        
        self.logger.info(f"Scaled trajectory saved to: {scaled_trajectory_file}")
        
        # 保存全局尺度因子到单独的文件
        parent_dir = os.path.dirname(trajectory_dir) # 获取上级目录
        global_scale_file = os.path.join(parent_dir, "global_scale.txt")
        with open(global_scale_file, 'w') as f:
            f.write(f"{scale:.6f}")
        
        self.logger.info(f"Global scale factor saved to: {global_scale_file}")
    
    def _create_output_directories(self, video_output_dir: str) -> Dict[str, str]:
        """创建所有必要的输出目录"""
        dirs = {
            'frames': os.path.join(video_output_dir, "frames"),
            'mask_data': os.path.join(video_output_dir, "mask_data"),
            'json_data': os.path.join(video_output_dir, "json_data"),
            'depth_data': os.path.join(video_output_dir, "depth_data"),
            'result': os.path.join(video_output_dir, "result"),
            'mask_photo': os.path.join(video_output_dir, "mask_photo"),
            'depth_photo': os.path.join(video_output_dir, "depth_photo"),
        }
        
        for path in dirs.values():
            CommonUtils.creat_dirs(path)
            
        return dirs
    
    def _get_sorted_frame_names(self, frames_dir: str) -> List[str]:
        """获取排序后的帧文件名列表"""
        frame_names = [
            p for p in os.listdir(frames_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        return frame_names
    
    def _process_frame_batch(
        self,
        start_frame_idx: int,
        step: int,
        total_frames: int,
        frame_names: List[str],
        dirs: Dict[str, str],
        text_prompt: str,
        inference_state: Any,
        sam2_masks: MaskDictionaryModel,
        objects_count: int,
        PROMPT_TYPE_FOR_VIDEO: str
    ):
        """处理一批帧"""
        self.logger.info(f"Processing frame batch {start_frame_idx + 1} to {min(start_frame_idx + step, total_frames)}/{total_frames}")
        
        if start_frame_idx >= total_frames:
            return
            
        img_path = os.path.join(dirs['frames'], frame_names[start_frame_idx])
        image_base_name = frame_names[start_frame_idx].split(".")[0]
        mask_dict = MaskDictionaryModel(promote_type=PROMPT_TYPE_FOR_VIDEO, mask_name=f"mask_{image_base_name}.npy")
        
        # 确保文本提示以点结尾（Grounding DINO要求）
        if not text_prompt.endswith('.'):
            text_prompt += '.'
        
        # 使用本地Grounding DINO模型进行检测 - 在FP32模式下
        image_source, image = load_image(img_path)  # 加载图像
        
        # 确保使用FP32
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=0.25,
                text_threshold=0.25,
                device=self.device
            )
        
        # 处理检测结果
        self.image_predictor.set_image(image_source)
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        
        # 修改：为每个检测到的物体创建唯一复合ID（类别_计数器）
        unique_labels = []
        for i, label in enumerate(labels):
            # 检查类别是否已存在于计数器中
            if label not in self.class_counter:
                self.class_counter[label] = 0
            
            # 递增类别计数器
            self.class_counter[label] += 1
            
            # 创建唯一复合ID
            unique_id = f"{label}_{self.class_counter[label]}"
            unique_labels.append(unique_id)
        labels = unique_labels
        
        if input_boxes.shape[0] != 0:
            masks, scores, logits = self.image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            if masks.ndim == 2:
                masks = masks[None]
            elif masks.ndim == 4:
                masks = masks.squeeze(1)
            
            if mask_dict.promote_type == "mask":
                mask_dict.add_new_frame_annotation(
                    mask_list=torch.tensor(masks).to(self.device),
                    box_list=torch.tensor(input_boxes),
                    label_list=labels  # 使用复合ID
                )
            else:
                raise NotImplementedError("SAM 2视频预测器仅支持mask提示")
            
            objects_count = mask_dict.update_masks(
                tracking_annotation_dict=sam2_masks,
                iou_threshold=0.90,
            )
            self.logger.info(f"Objects updated with unique IDs: {', '.join(labels)}")
        else:
            self.logger.info(f"在帧 {frame_names[start_frame_idx]} 中未检测到对象，跳过合并")
            mask_dict = sam2_masks
        
        if len(mask_dict.labels) == 0:
            mask_dict.save_empty_mask_and_json(
                dirs['mask_data'],
                dirs['json_data'],
                image_name_list=frame_names[start_frame_idx:start_frame_idx+step]
            )
            self.logger.info(f"在起始帧 {start_frame_idx} 中未检测到对象，跳过该批次")
            return
        
        self.video_predictor.reset_state(inference_state)
        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )
        
        self._propagate_and_save_masks(
            inference_state,
            start_frame_idx,
            step,
            total_frames,
            frame_names,
            dirs,
            mask_dict,
            sam2_masks
        )
    
    def _propagate_and_save_masks(
        self,
        inference_state: Any,
        start_frame_idx: int,
        step: int,
        total_frames: int,
        frame_names: List[str],
        dirs: Dict[str, str],
        mask_dict: MaskDictionaryModel,
        sam2_masks: MaskDictionaryModel
    ):
        """传播mask并保存结果"""
        video_segments = {}
        propagate_frames = min(total_frames - start_frame_idx, step)
        
        # 传播mask
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
            inference_state,
            max_frame_num_to_track=propagate_frames,
            start_frame_idx=start_frame_idx
        ):
            # 如果帧索引超出范围，跳出
            if out_frame_idx >= total_frames:
                break
                
            frame_masks = self._create_frame_masks(
                out_frame_idx, 
                out_obj_ids, 
                out_mask_logits, 
                frame_names, 
                mask_dict
            )
            
            video_segments[out_frame_idx] = frame_masks
            sam2_masks = copy.deepcopy(frame_masks)
        
        # 保存追踪结果
        for frame_idx, frame_masks_info in video_segments.items():
            self._save_mask_and_json(
                frame_masks_info, 
                dirs['mask_data'], 
                dirs['json_data']
            )
    
    def _create_frame_masks(
        self,
        frame_idx: int,
        obj_ids: List[int],
        mask_logits: torch.Tensor,
        frame_names: List[str],
        mask_dict: MaskDictionaryModel
    ) -> MaskDictionaryModel:
        """为帧创建mask字典"""
        frame_masks = MaskDictionaryModel()
        image_base_name = frame_names[frame_idx].split(".")[0]
        
        for i, obj_id in enumerate(obj_ids):
            out_mask = (mask_logits[i] > 0.0)
            # 从原始mask_dict中获取类别名称
            class_name = mask_dict.get_target_class_name(obj_id)
            object_info = ObjectInfo(
                instance_id=obj_id,
                mask=out_mask[0],
                class_name=class_name  # 保留类别名称
            )
            object_info.update_box()
            frame_masks.labels[obj_id] = object_info
        
        frame_masks.mask_name = f"mask_{image_base_name}.npy"
        frame_masks.mask_height = mask_logits.shape[-2]
        frame_masks.mask_width = mask_logits.shape[-1]
        
        return frame_masks
    
    def _save_mask_and_json(
        self,
        frame_masks_info: MaskDictionaryModel,
        mask_data_dir: str,
        json_data_dir: str
    ):
        """保存mask和JSON数据"""
        # 创建mask图像
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in frame_masks_info.labels.items():
            mask_img[obj_info.mask == True] = int(obj_id.split('_')[1])  # 使用原始ID部分作为mask值
        
        # 保存mask
        mask_img = mask_img.numpy().astype(np.uint16)
        mask_path = os.path.join(mask_data_dir, frame_masks_info.mask_name)
        np.save(mask_path, mask_img)
        
        # 保存JSON
        json_data = frame_masks_info.to_dict()
        # 添加类别到JSON数据
        for obj_id, obj_info in json_data['labels'].items():
            obj_info['class_name'] = frame_masks_info.labels[obj_id].class_name
        json_name = frame_masks_info.mask_name.replace(".npy", ".json")
        json_path = os.path.join(json_data_dir, json_name)
        with open(json_path, "w") as f:
            json.dump(json_data, f)

    def _extract_limited_frames(
        self, 
        video_path: str, 
        output_dir: str, 
        max_seconds: int,
        frames_per_second: float = 10.0,
        start_second: int = 0
    ):
        """
        提取指定秒数的视频帧，使用帧采样
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            max_seconds: 最大处理秒数
            frames_per_second: 每秒提取的帧数（默认为10帧/秒）
            start_second: 从视频的第几秒开始处理
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件 {video_path}")
        
        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算起始帧
        start_frame = int(start_second * fps)
        
        # 计算需要提取的帧数
        frames_to_extract = min(int(fps * max_seconds), total_frames - start_frame)
        
        # 计算帧间隔（确保至少1帧）
        if frames_per_second <= 0:
            frames_per_second = 0.1  # 最小0.1帧/秒
        frame_interval = max(1, int(fps / frames_per_second))
        
        # 创建输出目录
        CommonUtils.creat_dirs(output_dir)
        
        # 计算需要的数字位数
        num_digits = len(str(math.ceil(frames_to_extract / frame_interval)))
        
        # 提取帧和时间戳（使用帧索引作为时间戳）
        saved_frame_count = 0
        frame_timestamps = []  # 存储帧索引作为时间戳
        
        # 设置起始位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, start_frame + frames_to_extract):
            # 如果已经处理完所有帧，跳出循环
            if frame_idx >= total_frames:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            # 每隔frame_interval帧保存一次
            if (frame_idx - start_frame) % frame_interval == 0:
                # 使用帧索引作为时间戳
                frame_timestamps.append(saved_frame_count)
                
                # 使用固定长度的数字命名帧文件
                frame_file = os.path.join(output_dir, f"{saved_frame_count:0{num_digits}d}.jpg")
                cv2.imwrite(frame_file, frame)
                saved_frame_count += 1
        
        cap.release()
        self.logger.info(f"从视频 {video_path} 的第 {start_second} 秒开始提取了 {saved_frame_count} 帧 (约 {max_seconds} 秒), 采样率: {saved_frame_count / max_seconds:.2f}帧/秒")
        return frame_timestamps
    
    def _find_closest_pose(self, frame_index: int) -> Optional[Dict]:
        """找到最接近给定帧索引的位姿"""
        if not self.poses:
            return None
            
        # 确保位姿列表按帧索引排序
        sorted_poses = sorted(self.poses, key=lambda p: p['timestamp'])
        
        # 如果帧索引超出范围，返回最后一个位姿
        if frame_index >= len(sorted_poses):
            return sorted_poses[-1] if sorted_poses else None
            
        # 直接返回对应帧索引的位姿
        return sorted_poses[frame_index]
    
    def _compute_depth_maps_with_metric3d(
        self, 
        frames_dir: str, 
        depth_data_dir: str, 
        depth_photo_dir: str
    ):
        """使用Metric3D的正确接口计算深度图"""
        # 确保目录存在
        os.makedirs(depth_data_dir, exist_ok=True)
        os.makedirs(depth_photo_dir, exist_ok=True)
        
        # 处理Metric3D输出的深度文件
        self._process_depth_outputs(
            temp_dir=self.temp_depth_dir,
            frames_dir=frames_dir,
            data_dir=depth_data_dir,
            photo_dir=depth_photo_dir
        )

    def _generate_output_dirname(self, basename: str) -> str:
            """生成输出目录名"""
            first_word_match = re.search(r'^([^\W_]+)', basename)
            first_word = first_word_match.group(1) if first_word_match else 'output'
            
            # 匹配末尾连续数字（任意长度）
            last_digits_match = re.search(r'(\d+)$', basename)
            if last_digits_match:
                return f"{first_word}_{last_digits_match.group(1)}"
            else:
                return first_word
    
    def _process_depth_outputs(
        self, 
        temp_dir: str, 
        frames_dir: str, 
        data_dir: str, 
        photo_dir: str
    ):
        """
        处理Metric3D生成的深度输出：
        1. 移动深度数据到指定目录
        2. 生成可视化深度图
        3. 处理点云文件
        """
        # 获取原始帧名称列表
        frame_bases = [
            os.path.splitext(f)[0] for f in os.listdir(frames_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # 在临时目录中查找深度文件
        depth_files = [f for f in os.listdir(temp_dir) if f.endswith('.npy')]
        ply_files = [f for f in os.listdir(temp_dir) if f.endswith('.ply')]
        
        # 处理每个深度文件
        for depth_file in depth_files:
            # 解析原始帧名称
            match = re.match(r'depth_(.+)\.npy', depth_file)
            if not match:
                continue
                
            raw_base = match.group(1)
            frame_base = os.path.splitext(raw_base)[0]
            
            # 检查是否在帧列表中
            if frame_base not in frame_bases:
                continue
                
            depth_path = os.path.join(temp_dir, depth_file)
            depth_data = np.load(depth_path)
            
            # 保存深度数据
            data_path = os.path.join(data_dir, f"depth_{frame_base}.npy")
            np.save(data_path, depth_data)
            
            # 创建可视化图像
            self._create_depth_visualization(depth_data, frame_base, photo_dir)
            
            self.logger.info(f"Processed depth data for frame: {frame_base}")
        
        # 处理点云文件
        for ply_file in ply_files:
            match = re.match(r'pointcloud_(.+)\.ply', ply_file)
            if not match:
                continue
                
            frame_base = match.group(1)
            if frame_base not in frame_bases:
                continue
                
            ply_path = os.path.join(temp_dir, ply_file)
            data_path = os.path.join(data_dir, f"pointcloud_{frame_base}.ply")
            
            # 移动点云文件到输出目录
            os.rename(ply_path, data_path)
            self.logger.info(f"Moved point cloud file for frame: {frame_base}")
            
    def _create_depth_visualization(self, depth_data: np.ndarray, frame_base: str, photo_dir: str):
        """创建深度图可视化"""
        # 归一化深度值到0-255范围
        depth_min = depth_data.min()
        depth_max = depth_data.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth_data - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = depth_data
            
        depth_img = (depth_normalized * 255).astype(np.uint8)
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(photo_dir, f"depth_{frame_base}.jpg"), depth_img)

    def _get_ply_point_data(self, ply_path: str) -> Tuple[np.ndarray, List[int]]:
        """读取点云数据并构建快速索引"""
        from plyfile import PlyData
        
        point_data = PlyData.read(ply_path)
        points = []
        indices = []
        
        # 假设点云数据包含顶点列表
        vertex = point_data['vertex']
        for i, (x, y, z) in enumerate(zip(vertex['x'], vertex['y'], vertex['z'])):
            points.append([x, y, z])
            # 生成基于像素坐标的索引（假设点云顺序与像素位置匹配）
            indices.append(i)
        
        return np.array(points), indices

    def _find_nearest_point(
        self, 
        points: np.ndarray, 
        u: int, 
        v: int, 
        width: int
    ) -> Optional[np.ndarray]:
        """根据像素位置找到最近的点云点"""
        # 计算点云索引（假设点云顺序与像素位置匹配）
        idx = int(v * width + u)
        
        # 如果索引超出范围，返回None
        if idx >= len(points):
            return None
        
        return points[idx]

    def _calculate_absolute_distance(self, point: np.ndarray) -> Optional[float]:
        """计算点到相机的欧几里得距离"""
        if point is None:
            return None
        return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
    
    def _draw_masks_with_position(
            self, 
            frames_dir: str, 
            mask_data_dir: str, 
            json_data_dir: str,
            depth_data_dir: str,
            result_dir: str,
            mask_photo_dir: str,
            frame_indices: List[int]  # 使用帧索引列表
        ) -> Dict[int, Dict[str, Any]]:
            frame_names = self._get_sorted_frame_names(frames_dir)
            object_positions = {}  # 存储物体ID到位置信息的映射
            
            # 缓存点云数据
            point_cloud_cache = {}
            
            for frame_idx, frame_name in enumerate(tqdm(frame_names, desc="Rendering")):
                # 获取当前帧的索引
                frame_index = frame_indices[frame_idx] if frame_idx < len(frame_indices) else frame_idx
                
                # 渲染当前帧，获取位置信息
                frame_object_positions = self._render_frame(
                    frame_name, 
                    frames_dir, 
                    mask_data_dir, 
                    json_data_dir,
                    depth_data_dir,
                    result_dir,
                    mask_photo_dir,
                    point_cloud_cache,
                    frame_index  # 传递帧索引
                )
                
                # 更新全局物体位置
                for obj_id, position_info in frame_object_positions.items():
                    if obj_id not in object_positions:
                        # 初始化新物体的位置列表 - 添加original_id
                        object_positions[obj_id] = {
                            'class_name': position_info['class_name'],
                            'original_id': position_info['original_id'],
                            'positions': []  # 存储所有位置信息（包含距离）
                        }
                    
                    # 添加新位置（包含距离信息）
                    object_positions[obj_id]['positions'].append({
                        'camera_position': position_info['camera_position'],
                        'world_position': position_info['world_position'],
                        'distance': position_info['distance']
                    })
            
            return object_positions
    
    def _render_frame(
        self,
        frame_name: str,
        frames_dir: str,
        mask_data_dir: str,
        json_data_dir: str,
        depth_data_dir: str,
        result_dir: str,
        mask_photo_dir: str,
        point_cloud_cache: Dict[str, Any],
        frame_index: int  # 使用帧索引而不是时间戳
    ) -> Dict[str, Dict[str, Any]]:
        """渲染单个帧，返回该帧中物体的位置信息（按物体ID组织）"""
        frame_path = os.path.join(frames_dir, frame_name)
        img = cv2.imread(frame_path)
        img_h, img_w = img.shape[:2]
        
        # 获取帧的基本名称（不带扩展名）
        frame_base = os.path.splitext(frame_name)[0]
        
        # 加载点云数据
        current_point_cloud = self._load_point_cloud(
            frame_base, depth_data_dir, point_cloud_cache
        )
        
        # 加载mask数据
        mask_img = self._load_mask(frame_base, mask_data_dir)
        if mask_img is None:
            cv2.imwrite(os.path.join(result_dir, frame_name), img)
            return {}
        
        # 保存mask可视化图像
        self._save_mask_visualization(
            mask_img, frame_base, mask_photo_dir
        )
        
        # 加载json数据
        objects_data = self._load_json_data(frame_base, json_data_dir)
        if objects_data is None:
            cv2.imwrite(os.path.join(result_dir, frame_name), img)
            return {}
        
        # 使用帧索引查找位姿
        pose = self._find_closest_pose(frame_index)  # 获取当前帧的位姿
        
        # 存储当前帧的物体位置（按物体ID）
        frame_object_positions = {}
        
        # 为每个对象绘制边界框和位置
        for obj_id_str, obj_info in objects_data["labels"].items():
            # 确保obj_info包含类别名称
            if 'class_name' not in obj_info:
                self.logger.warning(f"Missing class_name for object {obj_id_str} in frame {frame_name}")
                continue
                
            position_info = self._render_object(
                obj_id_str,  # 使用复合ID
                obj_info, 
                mask_img, 
                img, 
                current_point_cloud, 
                img_w, 
                objects_data,
                pose  # 传递位姿信息
            )
            
            # 如果位置信息有效，添加到结果
            if position_info and position_info.get('camera_position') is not None:
                frame_object_positions[obj_id_str] = position_info
        
        # 保存结果图像
        cv2.imwrite(os.path.join(result_dir, frame_name), img)
        
        return frame_object_positions
    
    def _render_object(
        self,
        obj_id: str,  # 修改为字符串类型（复合ID）
        obj_info: Dict[str, Any],
        mask_img: np.ndarray,
        img: np.ndarray,
        current_point_cloud: Optional[Tuple[np.ndarray, List[int]]],
        img_w: int,
        objects_data: Dict[str, Any],
        pose: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        """渲染单个对象，返回位置信息（包含相机坐标系和世界坐标系）"""
        # 解析复合ID获取类别和原始ID
        parts = obj_id.split('_')
        if len(parts) < 2:
            self.logger.warning(f"Invalid object ID format: {obj_id}")
            return None
            
        class_name = parts[0]
        original_id = parts[1]
        
        # 将原始ID转换为整数用于mask处理
        try:
            obj_id_int = int(original_id)
        except ValueError:
            self.logger.warning(f"Invalid object ID integer part: {original_id}")
            return None
            
        mask = (mask_img == obj_id_int).astype(np.uint8)
        
        # 获取对象的边界框
        x_min = obj_info["x1"]
        y_min = obj_info["y1"]
        x_max = obj_info["x2"]
        y_max = obj_info["y2"]
        
        # 计算边界框中心点
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # 绘制边界框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # 获取当前类别所有物体的mask
        class_mask = np.zeros_like(mask)

        # 使用传入的objects_data获取所有同类对象
        for other_id, other_info in objects_data["labels"].items():
            # 检查other_info是否有class_name属性
            if 'class_name' not in other_info:
                continue
                
            if other_info["class_name"] == class_name and other_id != obj_id:
                try:
                    other_id_int = int(other_id.split('_')[1])
                except (IndexError, ValueError):
                    continue
                    
                other_mask = (mask_img == other_id_int).astype(np.uint8)
                class_mask = np.bitwise_or(class_mask, other_mask)
        
        # 计算当前物体的非重叠区域（独占区）
        exclusive_mask = np.bitwise_and(mask, np.bitwise_not(class_mask))
        
        # 找出所有非重叠区域的像素坐标
        exclusive_pixels = np.argwhere(exclusive_mask > 0)
        
        # 使用连通区域分析找到大面积区域
        use_exclusive = False
        coords = None
        
        # 优先使用独占区域，如果没有则使用整个mask
        target_mask = exclusive_mask if exclusive_pixels.size > 0 else mask
        
        if np.any(target_mask):
            # 进行连通区域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(target_mask, connectivity=8)
            
            if num_labels > 1:  # 至少有一个连通区域（除了背景）
                # 找到面积最大的连通区域（跳过背景，索引0）
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_area_idx = np.argmax(areas) + 1
                
                # 获取最大连通区域的mask
                largest_component = (labels == max_area_idx).astype(np.uint8)
                
                # 计算该区域的中心点
                M = cv2.moments(largest_component)
                if M["m00"] > 0:
                    region_center_x = int(M["m10"] / M["m00"])
                    region_center_y = int(M["m01"] / M["m00"])
                    
                    # 在这个区域内找到距离区域中心最近的点
                    region_pixels = np.argwhere(largest_component > 0)
                    if region_pixels.size > 0:
                        # 转换为(x, y)坐标
                        region_coords = region_pixels[:, [1, 0]]
                        
                        # 计算每个点到区域中心的距离
                        distances = np.sqrt((region_coords[:, 0] - region_center_x)**2 + 
                                        (region_coords[:, 1] - region_center_y)**2)
                        
                        # 找到距离最小的点
                        min_idx = np.argmin(distances)
                        closest_point_in_region = region_coords[min_idx]
                        
                        # 使用这个点作为代表点
                        coords = np.array([closest_point_in_region])
                        use_exclusive = exclusive_pixels.size > 0
        
        # 如果没有找到合适的点，回退到原始方法
        if coords is None:
            if exclusive_pixels.size > 0:
                # 直接获取 (x, y) 坐标数组
                coords = exclusive_pixels[:, [1, 0]]  # 交换列顺序 -> (x, y)
                use_exclusive = True
            else:
                # 使用整个mask
                y_coords, x_coords = np.where(mask > 0)
                coords = np.column_stack((x_coords, y_coords))  # 组合为 (x, y) 对
                if exclusive_pixels.size > 0:
                    self.logger.warning(f"警告：物体{obj_id}非重叠区域不足({exclusive_pixels.shape[0]}点)，使用整个mask ({len(x_coords)}点)")
                else:
                    self.logger.warning(f"警告：物体{obj_id}无独占区域，使用整个mask ({len(x_coords)}点)")
        
        # 计算距离
        distance_3d = None
        distance_2d = None
        camera_position = None
        world_position = None
        
        if coords.size > 0 and current_point_cloud:
            # 我们已经通过连通区域分析找到了最佳点，直接使用
            if len(coords) == 1:
                u, v = coords[0]
            else:
                # 回退到找距离图像中心最近的点
                min_distance = float('inf')
                closest_point = None
                
                for coord in coords:
                    u, v = coord[0], coord[1]  # x,y
                    distance_to_center = math.sqrt((u - center_x)**2 + (v - center_y)**2)
                    
                    if distance_to_center < min_distance:
                        min_distance = distance_to_center
                        closest_point = (u, v)
                
                u, v = closest_point
            
            # 寻找最近点
            point = self._find_nearest_point(current_point_cloud[0], u, v, img_w)
            
            if point is not None:
                # 在结果图像上绘制这个点（小红点）
                cv2.circle(img, (u, v), 3, (0, 0, 255), -1)
                
                # 计算3D距离
                distance_3d = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
                
                # 使用这个点作为物体位置
                camera_position = point
                
                # 计算世界坐标系位置（如果位姿可用）
                if pose is not None:
                    try:
                        world_position = self._transform_point_to_world(
                            np.array(camera_position), 
                            pose
                        ).tolist()
                        
                        # 计算2D平面距离（仅考虑XZ平面）
                        if world_position is not None:
                            # 相机在XZ平面的位置
                            camera_x = pose['position'][0]
                            camera_z = pose['position'][2]
                            
                            # 物体在XZ平面的位置
                            object_x = world_position[0]
                            object_z = world_position[2]
                            
                            # 计算2D欧几里得距离
                            distance_2d = math.sqrt(
                                (object_x - camera_x) ** 2 + 
                                (object_z - camera_z) ** 2
                            )
                    except Exception as e:
                        self.logger.info(f"坐标转换错误: {e}")
                        world_position = None
        
        # 在边界框上方绘制位置信息
        position_text = f"{class_name} {original_id}"
        if distance_2d is not None:
            position_text += f": {distance_2d:.1f}m"
        elif distance_3d is not None:
            position_text += f": {distance_3d:.1f}m"
            
        cv2.putText(img, position_text, (x_min, max(15, y_min - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 返回位置信息字典
        return {
            'camera_position': camera_position,
            'world_position': world_position,
            'class_name': class_name,
            'original_id': original_id,  # 确保返回原始ID
            'composite_id': obj_id,  # 添加复合ID
            'distance': distance_2d if distance_2d is not None else distance_3d,
            'distance_2d': distance_2d,
            'distance_3d': distance_3d
        }

    def _transform_point_to_world(self, point: np.ndarray, pose: Dict) -> np.ndarray:
        # 获取旋转矩阵（从世界系到相机系）
        R = self._quaternion_to_rotation_matrix(pose['rotation']) # 这得到的是 R_{world->camera}
        # 获取平移向量（相机在世界系中的位置）
        t = np.array(pose['position'])
        # 正确的变换: P_world = R^T * P_camera + t
        transformed_point = R.T @ point + t
        return transformed_point
    
    def _quaternion_to_rotation_matrix(self, q: List[float]) -> np.ndarray:
        """将四元数转换为旋转矩阵"""
        # 四元数格式为 [qx, qy, qz, qw]
        qx, qy, qz, qw = q
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,       2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,         1 - 2*qx**2 - 2*qz**2,   2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,       1 - 2*qx**2 - 2*qy**2]
        ])
        return R

    def _load_point_cloud(
        self,
        frame_base: str,
        depth_data_dir: str,
        point_cloud_cache: Dict[str, Any]
    ) -> Optional[Tuple[np.ndarray, List[int]]]:
        """加载点云数据"""
        ply_name = f"pointcloud_{frame_base}.ply"
        ply_path = os.path.join(depth_data_dir, ply_name)
        
        if not os.path.exists(ply_path):
            return None
            
        self.logger.info(f"Loading point cloud data from {ply_path}")
        
        # 从缓存获取或加载点云数据
        if ply_path in point_cloud_cache:
            return point_cloud_cache[ply_path]
        else:
            points, indices = self._get_ply_point_data(ply_path)
            point_cloud_cache[ply_path] = (points, indices)
            return (points, indices)
    
    def _load_mask(self, frame_base: str, mask_data_dir: str) -> Optional[np.ndarray]:
        """加载mask数据"""
        mask_name = f"mask_{frame_base}.npy"
        mask_path = os.path.join(mask_data_dir, mask_name)
        
        if not os.path.exists(mask_path):
            return None
            
        return np.load(mask_path)
    
    def _save_mask_visualization(self, mask_img: np.ndarray, frame_base: str, mask_photo_dir: str):
        """保存mask可视化图像"""
        mask_vis = np.zeros((mask_img.shape[0], mask_img.shape[1], 3), dtype=np.uint8)
        unique_ids = np.unique(mask_img)
        center_points = []  # 存储所有中心点
        
        for obj_id in unique_ids:
            if obj_id == 0:  # 跳过背景
                continue
                
            color = np.random.randint(0, 255, size=3, dtype=np.uint8)
            mask_vis[mask_img == obj_id] = color
            
            # 计算对象中心位置
            y_coords, x_coords = np.where(mask_img == obj_id)
            if len(y_coords) > 0 and len(x_coords) > 0:
                center_x = int(np.mean(x_coords))
                center_y = int(np.mean(y_coords))
                center_points.append((center_x, center_y))
        
        # 在mask图像上绘制所有中心点
        for center_x, center_y in center_points:
            cv2.circle(mask_vis, (center_x, center_y), 3, (0, 0, 255), -1)
        
        cv2.imwrite(os.path.join(mask_photo_dir, f"mask_{frame_base}.jpg"), mask_vis)
    
    def _load_json_data(self, frame_base: str, json_data_dir: str) -> Optional[Dict]:
        """加载JSON数据"""
        json_name = f"mask_{frame_base}.json"
        json_path = os.path.join(json_data_dir, json_name)
        
        if not os.path.exists(json_path):
            return None
            
        with open(json_path, 'r') as f:
            return json.load(f)

    def _cleanup_temp_files(self):
        """清理临时文件 - 现在只需要清理COLMAP临时文件"""
        # 清理COLMAP临时目录
        if hasattr(self, 'temp_colmap_dir') and self.temp_colmap_dir and os.path.exists(self.temp_colmap_dir):
            try:
                shutil.rmtree(self.temp_colmap_dir, ignore_errors=True)
                self.logger.info(f"Cleaned up COLMAP temporary directory: {self.temp_colmap_dir}")
            except Exception as e:
                self.logger.info(f"Error cleaning up COLMAP directory {self.temp_colmap_dir}: {e}")
        
        # 清理COLMAP输出目录
        if hasattr(self, 'temp_colmap_output') and self.temp_colmap_output and os.path.exists(self.temp_colmap_output):
            try:
                shutil.rmtree(self.temp_colmap_output, ignore_errors=True)
                self.logger.info(f"Cleaned up COLMAP output directory: {self.temp_colmap_output}")
            except Exception as e:
                self.logger.info(f"Error cleaning up COLMAP output directory {self.temp_colmap_output}: {e}")

def process_single_video(config, output_dir, pipeline_params):
    """
    在独立进程中处理单个视频
    """
    video_path = config['video_path']
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # 创建视频输出目录
    optimized_dirname = CombinedPipeline()._generate_output_dirname(video_basename)
    video_output_dir = os.path.join(output_dir, optimized_dirname)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # 配置视频专用日志
    video_log_file = os.path.join(video_output_dir, f"{optimized_dirname}_logger.log")
    video_logger = logging.getLogger(optimized_dirname)
    video_logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    for handler in video_logger.handlers[:]:
        video_logger.removeHandler(handler)
    
    # 添加文件处理器和流处理器
    file_handler = logging.FileHandler(video_log_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    video_logger.addHandler(file_handler)
    video_logger.addHandler(stream_handler)
    
    video_logger.info(f"\n{'='*80}")
    video_logger.info(f"开始处理视频: {video_path}")
    video_logger.info(f"参数配置:")
    video_logger.info(f"  最大处理秒数: {config['max_seconds']}")
    video_logger.info(f"  帧采样率: {config['frames_per_second']} fps")
    video_logger.info(f"  COLMAP间隔: {config['colmap_interval']} 秒")
    video_logger.info(f"  最大距离阈值: {config['max_distance']} 米")
    video_logger.info(f"  检测物体: {config['text_prompt']}")
    
    # 如果有提供内参焦距，打印出来
    if 'intrinsic_fx_fy' in config and config['intrinsic_fx_fy'] is not None:
        video_logger.info(f"  使用提供的焦距: fx={config['intrinsic_fx_fy'][0]}, fy={config['intrinsic_fx_fy'][1]}")
    else:
        video_logger.info(f"  使用COLMAP计算焦距")
    
    video_logger.info(f"{'='*80}")
    
    try:
        # 为每个视频创建独立的管道实例
        pipeline = CombinedPipeline()
        # 设置管道实例的日志记录器
        pipeline.logger = video_logger
        
        # 设置模型路径
        pipeline.set_model_paths(**pipeline_params)
        
        # 提取内参焦距参数（如果存在）
        intrinsic_fx_fy = config.get('intrinsic_fx_fy', None)
        
        output_video = pipeline.process_video(
            video_path=config['video_path'],
            text_prompt=config['text_prompt'],
            output_dir=output_dir,
            start_second=config["start_second"],
            max_seconds=config['max_seconds'],
            frames_per_second=config['frames_per_second'],
            colmap_interval=config['colmap_interval'],
            need_run_dpvo=True,
            max_distance=config['max_distance'],
            intrinsic_fx_fy=intrinsic_fx_fy,  # 传递内参焦距参数
            self_logger=video_logger,
        )
        video_logger.info(f"视频处理完成: {output_video}")
        return output_video
    except Exception as e:
        video_logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
        traceback.print_exc(file=sys.stdout)
        return None
    finally:
        # 清理日志处理器
        for handler in video_logger.handlers[:]:
            handler.close()
            video_logger.removeHandler(handler)

def process_single_video_wrapper(index, config, output_dir, pipeline_params, results_list):
    """
    包装函数，用于将结果存储到共享列表中
    """
    try:
        result = process_single_video(config, output_dir, pipeline_params)
        results_list[index] = result
    except Exception as e:
        results_list[index] = None
        print(f"处理视频 {config['video_path']} 时出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    start_time = time.time()

    # 输出路径
    OUTPUT_DIR = "./outputs/output_1"

    # 视频配置列表 - 每个视频可以有自己的参数
    # 参数使用指南：
    #   1. video_path: (必填) 输入视频的完整路径
    #   2. max_seconds: 处理视频的最大时长（秒），设为0表示处理整个视频
    #   3. frames_per_second: 帧采样率（每秒处理的帧数），值越高处理时间越长但结果更精确
    #   4. colmap_interval: COLMAP计算焦距时的帧间隔（秒），值越小结果越精确但计算时间越长
    #   5. max_distance: 最大物体检测距离（米），超过此距离的物体将不显示在可视化地图中
    #   6. text_prompt: (必填) Grounding DINO的检测提示文本，以点结尾，多个物体用空格分隔
    #   7. intrinsic_fx_fy: [可选] 已知的内参焦距 [fx, fy]，如果提供则跳过COLMAP计算。若不提供则设为None
    #   
    #   使用建议：
    #   - 对于短视频（<30秒），可以设置max_seconds=0处理整个视频
    #   - 对于快速测试，可以降低frames_per_second值（如0.5-1帧/秒）
    #   - 如果已知相机焦距，提供intrinsic_fx_fy可以节省计算时间
    #   - text_prompt应尽可能具体，避免检测到不需要的物体
    #   - 根据场景调整max_distance，室内场景建议5-10米，室外场景建议10-20米
    #   
    #   输出说明：
    #   处理完成后，每个视频将在OUTPUT_DIR下生成一个目录，包含：
    #   - result_dir/output_video/{优化名}_video.mp4: 最终输出视频
    #   - result_dir/local_map/{优化名}_map.jpg: 可视化地图
    #   - 其他中间结果和日志文件
    VIDEO_CONFIGS = [
        # {
        #     "video_path": "./dataset/videos/Waikiki Beach Walk - Spring Break in Honolulu, Hawaii [TC82CUk7NWI]_0000.mp4",
        #     "start_second": 0,
        #     "max_seconds": 120,
        #     "frames_per_second": 1,
        #     "colmap_interval": 0.2,
        #     "max_distance": 8,
        #     "text_prompt": "traffic light. fire hydrant. stop sign. parking meter. bench. backpack. umbrella. tie. suitcase. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush. bike.",
        #     "intrinsic_fx_fy": [751.0000, 640.0000]
        # },
        # {
        #     "video_path": "./dataset/videos/Waikiki Beach Walk - Spring Break in Honolulu, Hawaii [TC82CUk7NWI]_0001.mp4",
        #     "start_second": 0,
        #     "max_seconds": 120,
        #     "frames_per_second": 1,
        #     "colmap_interval": 0.2,
        #     "max_distance": 8,
        #     "text_prompt": "traffic light. fire hydrant. stop sign. parking meter. bench. backpack. umbrella. tie. suitcase. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush. bike.",
        #     "intrinsic_fx_fy": [751.0000, 640.0000]
        # },
        # {
        #     "video_path": "./dataset/videos/Waikiki Beach Walk - Spring Break in Honolulu, Hawaii [TC82CUk7NWI]_0002.mp4",
        #     "start_second": 0,
        #     "max_seconds": 120,
        #     "frames_per_second": 1,
        #     "colmap_interval": 0.2,
        #     "max_distance": 8,
        #     "text_prompt": "traffic light. fire hydrant. stop sign. parking meter. bench. backpack. umbrella. tie. suitcase. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush. bike.",
        #     "intrinsic_fx_fy": [751.0000, 640.0000]
        # },
        {
            "video_path": "./dataset/videos/Waikiki Beach Walk - Spring Break in Honolulu, Hawaii [TC82CUk7NWI]_0000.mp4",
            "start_second": 0,
            "max_seconds": 10,
            "frames_per_second": 1,
            "colmap_interval": 0.2,
            "max_distance": 8,
            "text_prompt": "traffic light. fire hydrant. stop sign. parking meter. bench. backpack. umbrella. tie. suitcase. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush. bike.",
            "intrinsic_fx_fy": [751.0000, 640.0000]
        },
        {
            "video_path": "./dataset/videos/Waikiki Beach Walk - Spring Break in Honolulu, Hawaii [TC82CUk7NWI]_0001.mp4",
            "start_second": 0,
            "max_seconds": 10,
            "frames_per_second": 1,
            "colmap_interval": 0.2,
            "max_distance": 8,
            "text_prompt": "traffic light. fire hydrant. stop sign. parking meter. bench. backpack. umbrella. tie. suitcase. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush. bike.",
            "intrinsic_fx_fy": [751.0000, 640.0000]
        },
        {
            "video_path": "./dataset/videos/Waikiki Beach Walk - Spring Break in Honolulu, Hawaii [TC82CUk7NWI]_0002.mp4",
            "start_second": 0,
            "max_seconds": 10,
            "frames_per_second": 1,
            "colmap_interval": 0.2,
            "max_distance": 8,
            "text_prompt": "traffic light. fire hydrant. stop sign. parking meter. bench. backpack. umbrella. tie. suitcase. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush. bike.",
            "intrinsic_fx_fy": [751.0000, 640.0000]
        },
    ]
    
    # 管道参数
    PIPELINE_PARAMS = {
        "sam2_checkpoint": "./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
        "model_cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "grounding_dino_config": "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "grounding_dino_checkpoint": "Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
        "metric3d_config": "Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py",
        "metric3d_ckpt": "Metric3D/weight/metric_depth_vit_large_800k.pth"
    }

    # 创建主日志文件
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main_log_file = os.path.join(OUTPUT_DIR, "main_process.log")
    
    # 配置主日志记录
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger()
    
    logger.info(f"开始处理 {len(VIDEO_CONFIGS)} 个视频")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"主日志文件: {main_log_file}")

    # 创建共享结果列表
    manager = multiprocessing.Manager()
    results = manager.list([None] * len(VIDEO_CONFIGS))
    
    processes = []
    for idx, config in enumerate(VIDEO_CONFIGS):
        # 创建并启动进程
        p = multiprocessing.Process(
            target=process_single_video_wrapper,
            args=(idx, config, OUTPUT_DIR, PIPELINE_PARAMS, results)
        )
        p.daemon = False  # 关键修改：设置为非守护进程
        p.start()
        processes.append(p)
        logger.info(f"启动视频处理进程 {idx+1}/{len(VIDEO_CONFIGS)}: {config['video_path']}")
        
        # 控制并发进程数
        if len(processes) >= min(4, len(VIDEO_CONFIGS)):  # 限制最大并发数
            # 等待一个进程完成
            for p in processes:
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
                    break
            time.sleep(1)  # 避免忙等待
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 将共享列表转换为普通列表
    results = list(results)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"\n{'='*80}")
    logger.info(f"所有视频处理完成!")
    logger.info(f"处理了 {len(VIDEO_CONFIGS)} 个视频")
    logger.info(f"输出视频列表:")
    for i, video in enumerate(results):
        if video:
            logger.info(f"  {i+1}. {video}")
        else:
            logger.info(f"  {i+1}. 处理失败")
    logger.info(f"总处理时间: {total_time:.2f} 秒")
    logger.info(f"{'='*80}")
    
    # 保存总结报告
    summary_file = os.path.join(OUTPUT_DIR, "processing_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"视频处理总结报告\n")
        f.write(f"=================\n\n")
        f.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
        f.write(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
        f.write(f"总处理时间: {total_time:.2f} 秒\n")
        f.write(f"处理视频数量: {len(VIDEO_CONFIGS)}\n")
        f.write(f"成功处理: {sum(1 for r in results if r is not None)}\n")
        f.write(f"失败处理: {sum(1 for r in results if r is None)}\n\n")
        
        f.write("视频处理详情:\n")
        f.write("-------------\n")
        for i, (config, result) in enumerate(zip(VIDEO_CONFIGS, results)):
            f.write(f"视频 {i+1}:\n")
            f.write(f"  路径: {config['video_path']}\n")
            f.write(f"  最大秒数: {config['max_seconds']}\n")
            f.write(f"  帧率: {config['frames_per_second']} fps\n")
            f.write(f"  状态: {'成功' if result else '失败'}\n")
            if result:
                f.write(f"  输出视频: {result}\n")
            f.write("\n")
    
    logger.info(f"处理总结已保存至: {summary_file}")