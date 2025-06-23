#!/bin/bash

# --- 设置模型缓存目录 ---
export HF_HOME="/workspace/.cache"
export HF_DATASETS_CACHE="/workspace/.cache/datasets"

# --- 模型和服务器配置 ---
MODEL_NAME="/THUDM/GLM-4-9B-0414"
# 客户端将通过这个名称来调用模型
SERVER_MODEL_NAME="THUDM/GLM-4-9B-0414"
# 端口号
PORT=8000

echo "正在启动 vLLM OpenAI API 服务器..."
echo "模型: $MODEL_NAME"
echo "服务名称: $SERVER_MODEL_NAME"
echo "端口: $PORT"

# --- 启动 vLLM 服务器 ---
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --served-model-name "$SERVER_MODEL_NAME" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port "$PORT" \
    --trust-remote-code