set -e 
pushd test/ci/
export PATH=$HOME/.dotnet:$PATH
cp -r ../../Packages/Runtime .
ls 
export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
dotnet build 
dotnet run
echo "exit code:$?"
popd