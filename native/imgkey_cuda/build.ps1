[CmdletBinding()]
param(
    [string]$CudaPath = $env:CUDA_PATH,
    [string]$VsInstallPath = "",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $OutDir = Join-Path $SourceDir "build"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Resolve-CudaRoot {
    param([string]$Preferred)
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($Preferred)) {
        $candidates += $Preferred
    }
    $candidates += "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    $cudaParent = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $cudaParent) {
        $candidates += Get-ChildItem -LiteralPath $cudaParent -Directory | Sort-Object Name -Descending | ForEach-Object { $_.FullName }
    }
    foreach ($candidate in $candidates | Select-Object -Unique) {
        $nvcc = Join-Path $candidate "bin\nvcc.exe"
        if (Test-Path -LiteralPath $nvcc) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    $nvccCommand = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvccCommand) {
        return (Resolve-Path -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $nvccCommand.Source)) ".")).Path
    }
    throw "nvcc.exe not found. Install the NVIDIA CUDA Toolkit or set CUDA_PATH."
}

function Resolve-VsDevCmd {
    param([string]$PreferredInstallPath)
    $installCandidates = @()
    if (-not [string]::IsNullOrWhiteSpace($PreferredInstallPath)) {
        $installCandidates += $PreferredInstallPath
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $found = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($found)) {
            $installCandidates += $found.Trim()
        }
    }
    $installCandidates += "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
    $installCandidates += "C:\Program Files\Microsoft Visual Studio\2022\BuildTools"
    $installCandidates += "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"

    foreach ($install in $installCandidates | Select-Object -Unique) {
        $vsDevCmd = Join-Path $install "Common7\Tools\VsDevCmd.bat"
        if (Test-Path -LiteralPath $vsDevCmd) {
            return (Resolve-Path -LiteralPath $vsDevCmd).Path
        }
        $vcvars64 = Join-Path $install "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path -LiteralPath $vcvars64) {
            return (Resolve-Path -LiteralPath $vcvars64).Path
        }
    }
    throw "Visual Studio Build Tools with MSVC x64 tools not found. Install VS Build Tools or pass -VsInstallPath."
}

function Get-NvccGpuArchList {
    param([string]$Nvcc)
    $output = & $Nvcc --list-gpu-arch 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "nvcc --list-gpu-arch failed: $output"
    }
    return @($output | ForEach-Object { "$($_)".Trim() } | Where-Object { $_ })
}

$CudaRoot = Resolve-CudaRoot -Preferred $CudaPath
$Nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$VsDevCmd = Resolve-VsDevCmd -PreferredInstallPath $VsInstallPath
$ArchList = Get-NvccGpuArchList -Nvcc $Nvcc

if ($ArchList -contains "compute_120") {
    $Gencode = @("-gencode=arch=compute_120,code=sm_120", "-gencode=arch=compute_120,code=compute_120")
    $ArchNote = "native compute_120/sm_120"
} elseif ($ArchList -contains "compute_90") {
    $Gencode = @("-gencode=arch=compute_90,code=sm_90", "-gencode=arch=compute_90,code=compute_90")
    $ArchNote = "sm_90 plus compute_90 PTX forward-JIT for newer GPUs"
} else {
    throw "nvcc does not support compute_120 or compute_90; cannot build the ImgKey CUDA DLL."
}

$Source = Join-Path $SourceDir "imgkey_cuda.cu"
$OutputDll = Join-Path $OutDir "imgkey_cuda.dll"
$LogPath = Join-Path $OutDir "build.log"
$GencodeText = ($Gencode -join " ")

$cmdParts = @(
    "call `"$VsDevCmd`" -arch=amd64 -host_arch=amd64",
    "echo ImgKey CUDA DLL build",
    "echo CUDA root: `"$CudaRoot`"",
    "echo VS env: `"$VsDevCmd`"",
    "echo Arch strategy: $ArchNote",
    "cl",
    "`"$Nvcc`" --version",
    "`"$Nvcc`" -shared -std=c++17 -cudart static $GencodeText -Xcompiler `"/MD /O2`" -o `"$OutputDll`" `"$Source`""
)
$cmd = $cmdParts -join " && "

Write-Host "Building $OutputDll"
Write-Host "Logging to $LogPath"
& cmd.exe /d /s /c $cmd 2>&1 | Tee-Object -FilePath $LogPath
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    throw "ImgKey CUDA DLL build failed with exit code $exitCode. See $LogPath"
}

Write-Host "Built $OutputDll"
