@echo off
setlocal
set "FP_ARCH=75"
if "%~1"=="90" set "FP_ARCH=90"
if not "%~1"=="" if not "%~1"=="75" if not "%~1"=="90" goto usage
set "FP_VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
set "FP_NVCC=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe"
if not exist "%FP_VCVARS%" (
  echo Visual Studio 2022 Build Tools were not found at the configured location.
  exit /b 1
)
if not exist "%FP_NVCC%" (
  echo CUDA 12.4 was not found at the configured location.
  exit /b 1
)
call "%FP_VCVARS%" x64 -vcvars_ver=14.38
if errorlevel 1 exit /b %errorlevel%
"%FP_NVCC%" -ccbin "%VCToolsInstallDir%bin\Hostx64\x64" -std=c++17 -O2 -arch=sm_%FP_ARCH% "%~dp0probe_hardened.cu" -o "%~dp0probe_hardened_sm%FP_ARCH%.exe"
if errorlevel 1 exit /b %errorlevel%
echo Built "%~dp0probe_hardened_sm%FP_ARCH%.exe". No GPU measurement was run.
exit /b 0
:usage
echo Usage: build_windows.cmd [75 or 90]
exit /b 1
