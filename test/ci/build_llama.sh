#!/bin/bash
set -e
pushd $CI_PROJECT_DIR/.cache/
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp
git checkout 75cd4c7
make libllama.so
mv libllama.so /$HOME/.dotnet/shared/Microsoft.NETCore.App/5.0.17/
popd
