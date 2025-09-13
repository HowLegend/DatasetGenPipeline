_base_=[
       '../_base_/models/encoder_decoder/dino_vit_large_reg.dpt_raft.py',
       '../_base_/datasets/_data_base_.py',
       '../_base_/default_runtime.py',
       ]

import numpy as np
model=dict(
    decode_head=dict(
        type='RAFTDepthNormalDPT5',
        iters=8,
        n_downsample=2,
        detach=False,
    )
)


max_value = 200
# configs of the canonical space
data_basic=dict(
    canonical_space = dict(
        # img_size=(540, 960),    # 代码原来就已经注释了
        focal_length=1000.0,
    ),
    depth_range=(0, 1),  # 模型​​输出的深度值范围
    depth_normalize=(0.1, max_value), # ​​深度归一化参数
    crop_size = (616, 1064),  # 图像预处理尺寸  能被28整除，确保ViT的patch操作无余数（28×28分块）
     clip_depth_range=(0.1, 200), # ​​有效深度截断范围:
                                  # <0.1米的深度设为0.1米
                                  # >200米的深度设为200米
    vit_size=(616,1064) # Vision Transformer的​​输入尺寸​​
) 

batchsize_per_gpu = 1
thread_per_gpu = 1
