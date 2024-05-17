#!/bin/bash
set -e
pushd $CI_PROJECT_DIR/.cache/
if [ ! -f "llama-2-7b-chat.Q2_K.gguf" ]; then
    wget -q https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf?download=true -O llama-2-7b-chat.Q2_K.gguf
fi
mv $CI_PROJECT_DIR/.cache/llama-2-7b-chat.Q2_K.gguf $CI_PROJECT_DIR/test/ci/Assets/Models
popd
