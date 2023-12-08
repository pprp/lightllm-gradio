#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python -m lightllm.server.api_server \
                            --model_dir /data2/dongpeijie/workspace/lightllm/finetune_chatbotv4 \
                            --host 0.0.0.0                 \
                            --port 8080                    \
                            --tp 1                     \
                            --max_total_token_num 12000