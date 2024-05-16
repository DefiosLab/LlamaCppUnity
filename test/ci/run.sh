set -e 
pushd test/ci/
export PATH=$HOME/.dotnet:$PATH
cp ../CSLlamaSampleProject/Assets/Scripts/LlamaSample.cs Program.cs
cp -r ../CSLlamaSampleProject/Assets/Scripts/test .
cp -r ../../Packages/Runtime .
mv ../CSLlamaSampleProject/Assets/Models/llama-2-7b-chat.ggml.q2_K.gguf Assets/Models/
ls 
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
dotnet build 
dotnet run
echo "exit code:$?"
popd
