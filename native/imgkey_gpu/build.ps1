param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root "build"
$Shader = Join-Path $Root "imgkey_gpu_color.hlsl"
$GeneratedHeader = Join-Path $BuildDir "imgkey_gpu_shaders.h"
$IdentityBlob = Join-Path $BuildDir "imgkey_identity.cso"
$ColorBlob = Join-Path $BuildDir "imgkey_color_tile.cso"
$FullColorBlob = Join-Path $BuildDir "imgkey_full_color_tile.cso"
$ResponseFile = Join-Path $BuildDir "cl.rsp"
$DllOut = Join-Path $BuildDir "imgkey_gpu.dll"

if ($Clean -and (Test-Path -LiteralPath $BuildDir)) {
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null

function Find-VsWhere {
    $pf86 = ${env:ProgramFiles(x86)}
    if (-not $pf86) { $pf86 = "C:\Program Files (x86)" }
    $candidates = @(
        (Join-Path $pf86 "Microsoft Visual Studio\Installer\vswhere.exe")
    )
    foreach ($dir in ($env:PATH -split [IO.Path]::PathSeparator)) {
        if ($dir) { $candidates += (Join-Path $dir "vswhere.exe") }
    }
    foreach ($path in $candidates) {
        if (Test-Path -LiteralPath $path) { return $path }
    }
    return $null
}

function Find-VsRoot {
    $roots = @()
    $vswhere = Find-VsWhere
    if ($vswhere) {
        $found = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        foreach ($line in $found) {
            if ($line -and (Test-Path -LiteralPath $line)) { $roots += $line }
        }
    }
    $roots += @(
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
        "C:\Program Files\Microsoft Visual Studio\2022\Community",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional",
        "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
    )
    foreach ($root in $roots) {
        $vcvars = Join-Path $root "VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path -LiteralPath $vcvars) { return $root }
    }
    return $null
}

function Find-SdkRoots {
    $roots = @()
    if ($env:WindowsSdkDir) { $roots += $env:WindowsSdkDir }
    if ($env:WindowsSDKDir) { $roots += $env:WindowsSDKDir }
    $roots += @(
        "C:\Program Files (x86)\Windows Kits\10",
        "C:\Program Files\Windows Kits\10",
        "C:\Program Files (x86)\Windows Kits\11",
        "C:\Program Files\Windows Kits\11"
    )
    $out = @()
    foreach ($root in $roots) {
        if ($root -and (Test-Path -LiteralPath $root)) { $out += (Resolve-Path -LiteralPath $root).Path }
    }
    return $out | Select-Object -Unique
}

function Find-ToolInSdk([string]$Name) {
    $pathTool = Get-Command $Name -ErrorAction SilentlyContinue
    if ($pathTool) { return $pathTool.Source }
    $candidates = @()
    foreach ($root in Find-SdkRoots) {
        $candidates += Get-ChildItem -Path (Join-Path $root "bin") -Filter $Name -Recurse -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
    }
    if ($candidates.Count -gt 0) { return ($candidates | Sort-Object | Select-Object -Last 1) }
    return $null
}

function Invoke-ShaderCompile([string]$Entry, [string]$OutFile) {
    $dxc = Find-ToolInSdk "dxc.exe"
    if ($dxc) {
        & $dxc -nologo -T cs_6_0 -E $Entry -Fo $OutFile $Shader
        if ($LASTEXITCODE -eq 0) { return }
        & $dxc -nologo -T cs_5_0 -E $Entry -Fo $OutFile $Shader
        if ($LASTEXITCODE -eq 0) { return }
        throw "DXC failed compiling $Entry"
    }
    $fxc = Find-ToolInSdk "fxc.exe"
    if ($fxc) {
        & $fxc /nologo /T cs_5_0 /E $Entry /Fo $OutFile $Shader
        if ($LASTEXITCODE -eq 0) { return }
        throw "FXC failed compiling $Entry"
    }
    throw "Neither dxc.exe nor fxc.exe was found. Install Windows SDK shader tools to build imgkey_gpu.dll."
}

function Write-CArray([string]$Path, [string]$Name, [byte[]]$Bytes, [bool]$Append = $false) {
    $lines = New-Object System.Collections.Generic.List[string]
    if (-not $Append) {
        $lines.Add("#pragma once")
        $lines.Add("")
    }
    $lines.Add("static const unsigned char $Name[] = {")
    for ($i = 0; $i -lt $Bytes.Length; $i += 12) {
        $end = [Math]::Min($i + 11, $Bytes.Length - 1)
        $chunk = for ($j = $i; $j -le $end; $j++) { "0x{0:x2}" -f $Bytes[$j] }
        $suffix = if ($end -lt $Bytes.Length - 1) { "," } else { "" }
        $lines.Add("    " + ($chunk -join ", ") + $suffix)
    }
    $lines.Add("};")
    $lines.Add("")
    if ($Append) {
        Add-Content -LiteralPath $Path -Value $lines -Encoding ASCII
    } else {
        Set-Content -LiteralPath $Path -Value $lines -Encoding ASCII
    }
}

Invoke-ShaderCompile "ImgKeyIdentityCS" $IdentityBlob
Invoke-ShaderCompile "ImgKeyColorTileCS" $ColorBlob
Invoke-ShaderCompile "ImgKeyFullColorTileCS" $FullColorBlob
Write-CArray $GeneratedHeader "g_imgkey_identity_cs" ([IO.File]::ReadAllBytes($IdentityBlob)) $false
Write-CArray $GeneratedHeader "g_imgkey_color_tile_cs" ([IO.File]::ReadAllBytes($ColorBlob)) $true
Write-CArray $GeneratedHeader "g_imgkey_full_color_tile_cs" ([IO.File]::ReadAllBytes($FullColorBlob)) $true

$vsRoot = Find-VsRoot
$vcvars = if ($vsRoot) { Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat" } else { $null }
$cl = Get-Command "cl.exe" -ErrorAction SilentlyContinue
if (-not $vcvars -and -not $cl) {
    throw "MSVC Build Tools were not found. Install Visual Studio Build Tools with C++ x64 tools."
}

$rsp = @(
    "/nologo",
    "/EHsc",
    "/std:c++17",
    "/O2",
    "/LD",
    "/Fo`"$BuildDir\\`"",
    "/DUNICODE",
    "/D_UNICODE",
    "/I`"$Root`"",
    "/I`"$BuildDir`"",
    "`"$(Join-Path $Root 'imgkey_gpu.cpp')`""
)
Set-Content -LiteralPath $ResponseFile -Value $rsp -Encoding ASCII

if ($vcvars) {
    $cmd = "call `"$vcvars`" >nul && cl.exe @`"$ResponseFile`" /link /NOLOGO /DLL /OUT:`"$DllOut`" /IMPLIB:`"$(Join-Path $BuildDir 'imgkey_gpu.lib')`" d3d12.lib dxgi.lib dxguid.lib"
    & cmd.exe /d /s /c $cmd
} else {
    & $cl.Source "@$ResponseFile" /link /NOLOGO /DLL "/OUT:$DllOut" "/IMPLIB:$(Join-Path $BuildDir 'imgkey_gpu.lib')" d3d12.lib dxgi.lib dxguid.lib
}
if ($LASTEXITCODE -ne 0) {
    throw "MSVC build failed with exit code $LASTEXITCODE"
}

$dumpbin = Get-Command "dumpbin.exe" -ErrorAction SilentlyContinue
if ($dumpbin) {
    & $dumpbin.Source /dependents $DllOut | Set-Content -LiteralPath (Join-Path $BuildDir "dependents.txt") -Encoding UTF8
}

"Built $DllOut"
