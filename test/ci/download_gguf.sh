#!/bin/bash
set -e
pushd "${GITHUB_WORKSPACE}/.cache/"
if [ ! -f "llama-2-7b-chat.Q2_K.gguf" ]; then
    wget -q https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q2_K.gguf?download=true -O llama-2-7b-chat.Q2_K.gguf
fi
mv ${GITHUB_WORKSPACE}/.cache/llama-2-7b-chat.Q2_K.gguf ${GITHUB_WORKSPACE}/test/ci/Assets/Models/
popd