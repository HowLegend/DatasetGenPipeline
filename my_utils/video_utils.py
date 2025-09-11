import cv2
import os
from typing import List, Optional
import math
from tqdm import tqdm

def create_video_from_images(image_dir: str, output_video_path: str, frame_rate: int = 30):
    """
    将图像目录中的图像序列合成为视频
    
    Args:
        image_dir: 包含图像的目录路径
        output_video_path: 输出视频文件路径
        frame_rate: 输出视频的帧率
    """
    # 获取目录中所有图像文件
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()  # 按文件名排序
    
    if not image_files:
        print(f"警告: 目录 {image_dir} 中没有图像文件")
        return
    
    # 读取第一张图像以获取尺寸
    first_image_path = os.path.join(image_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"无法读取第一张图像: {first_image_path}")
        return
    
    height, width, _ = frame.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    
    if not video_writer.isOpened():
        print(f"无法创建视频文件: {output_video_path}")
        return
    
    # 逐帧写入视频
    for image_file in tqdm(image_files, desc="创建视频"):
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is not None:
            video_writer.write(frame)
    
    # 释放资源
    video_writer.release()
    print(f"视频已保存至: {output_video_path}")


