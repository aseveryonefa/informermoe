#!/bin/bash
config_file=./config/vamoe.yaml
config='vamoe'
run_num='1'


NAME='Informer'

LOG_DIR="./logs/${NAME}/"
# 确保目录存在
mkdir -p -- "$LOG_DIR"

export PYTHONIOENCODING=utf-8


# 2. 定位上一次最好的模型文件路径
BEST_CKPT="${LOG_DIR}${config}/${run_num}/training_checkpoints/best_ckpt.tar"

# 3. 判断模型文件是否存在
# 如果存在，则从该模型恢复训练；如果不存在，checkpoint 为空（从头开始）
if [ -f "$BEST_CKPT" ]; then
    echo "Found best checkpoint: $BEST_CKPT, resuming training..."
    checkpoint="$BEST_CKPT"
else
    echo "No existing checkpoint found at $BEST_CKPT, starting fresh..."
    checkpoint=""
fi

# 4. 启动训练
# 日志会重定向到 train.log，覆盖上一次的日志
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 train.py \
#             --enable_amp \
#             --yaml_config=$config_file \
#             --config=$config \
#             --run_num=$run_num \
#             --exp_dir=$LOG_DIR \
#             --checkpoint="$checkpoint" \
#             > ${LOG_DIR}train.log 2>&1

CUDA_VISIBLE_DEVICES=0 python train.py \
            --enable_amp \
            --yaml_config=$config_file \
            --config=$config \
            --run_num=$run_num \
            --exp_dir=$LOG_DIR \
            --checkpoint="$checkpoint" \
            > ${LOG_DIR}train.log 2>&1


# #双卡运行
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py \
#             --enable_amp \
#             --yaml_config=$config_file \
#             --config=$config \
#             --run_num=$run_num \
#             --exp_dir=$LOG_DIR \
#             --checkpoint="$checkpoint" \
#             > ${LOG_DIR}train.log 2>&1
