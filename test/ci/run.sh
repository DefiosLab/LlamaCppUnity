set -e 
pushd test/ci/
export PATH=$HOME/.dotnet:$PATH
cp ../TestLlamaCppUnity/Assets/Scripts/LlamaTest.cs Program.cs
cp -r ../../Packages/Runtime .
mv ../TestLlamaCppUnity/Assets/Models/llama-2-7b-chat.q2_K.gguf Assets/Models/
ls 
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
dotnet build 
dotnet run
echo "exit code:$?"
popd
