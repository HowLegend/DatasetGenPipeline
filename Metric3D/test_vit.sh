python mono/tools/test_scale_cano.py \
    'mono/configs/HourglassDecoder/vit.raft5.large.py' \
    --load-from ./weight/metric_depth_vit_large_800k.pth \
    --test_data_path ./data/wild_demo \
    --launcher None