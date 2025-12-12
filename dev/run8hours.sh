#!/bin/bash

# 8小时 MacBook 训练脚本
# 目标：在 8 小时内训练尽可能大的模型
# 模型配置：d8 (8层, 512维, ~95M 参数)

# set -e  # 遇到错误就停止

# 环境设置
export OMP_NUM_THREADS=4
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # 允许使用更多内存
mkdir -p $NANOCHAT_BASE_DIR

# 安装依赖
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate

# wandb 设置
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Rust 环境（用于 tokenizer）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 清空报告
python -m nanochat.report reset

echo "=========================================="
echo "步骤 1/6: 下载训练数据 (约 5GB)"
echo "=========================================="
# 下载 20 个 shard 的数据（支持更长时间训练）
python -m nanochat.dataset -n 20

echo "=========================================="
echo "步骤 2/6: 训练 Tokenizer"
echo "=========================================="
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

echo "=========================================="
echo "步骤 3/6: Base 预训练 (约 3-4 小时)"
echo "=========================================="
# d8 模型：8层, 512维, ~95M 参数
# 20000 步，每步约 80ms，总计约 27 分钟
# 但由于 grad_accum 和评估，实际约 3-4 小时
python -m scripts.base_train \
    --depth=8 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --eval_every=1000 \
    --eval_tokens=8192 \
    --core_metric_every=-1 \
    --sample_every=2000 \
    --num_iterations=20000

python -m scripts.base_loss --device_batch_size=1 --split_tokens=8192

echo "=========================================="
echo "步骤 4/6: Mid Training (约 2-3 小时)"
echo "=========================================="
python -m scripts.mid_train \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --eval_every=500 \
    --eval_tokens=8192 \
    --total_batch_size=512 \
    --num_iterations=10000

# Mid 模型评估
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=50

echo "=========================================="
echo "步骤 5/6: SFT 训练 (约 1 小时)"
echo "=========================================="
# SFT 使用较短序列避免 OOM
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_iterations=3000 \
    --eval_steps=10 \
    --eval_metrics_max_problems=32

# SFT 模型评估
python -m scripts.chat_eval --source=sft --max-new-tokens=128 --max-problems=50

echo "=========================================="
echo "步骤 6/6: 生成报告"
echo "=========================================="
python -m nanochat.report generate

echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "你可以用以下命令与模型对话："
echo "  python -m scripts.chat_cli"
echo "或者启动 Web 界面："
echo "  python -m scripts.chat_web"

