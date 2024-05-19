#!/bin/bash
set -e
pushd "${GITHUB_WORKSPACE}/.cache"
if [ ! -d ".dotnet" ]; then
    mkdir -p .dotnet
    cd .dotnet
    wget https://download.visualstudio.microsoft.com/download/pr/904da7d0-ff02-49db-bd6b-5ea615cbdfc5/966690e36643662dcc65e3ca2423041e/dotnet-sdk-5.0.408-linux-x64.tar.gz
    tar -zxvf dotnet-sdk-5.0.408-linux-x64.tar.gz
    cd ../
fi
cp -r .dotnet $HOME
popd
