#!/bin/bash
set -e
pushd $CI_PROJECT_DIR/.cache/
# huggingfaceディレクトリが存在しなければ作成
if [ ! -d "huggingface" ]; then
    mkdir -p huggingface
fi
# huggingfaceディレクトリに移動
cd huggingface

if [ ! -f "llama-2-7b-chat.Q2_K.gguf" ]; then
    wget -q https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf?download=true -O llama-2-7b-chat.Q2_K.gguf
    cd ../
fi
mv huggingface/llama-2-7b-chat.Q2_K.gguf Assets/Models
popd
