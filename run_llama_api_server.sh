#!/bin/bash
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m lightllm.server.api_server \
                            --model_dir /data2/dongpeijie/workspace/vllm/merged_lora_chatbotv2 \
                            --host 0.0.0.0                 \
                            --port 8080                    \
                            --tp 4                     \
                            --max_total_token_num 12000