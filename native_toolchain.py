from __future__ import annotations

import glob
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


TOOLCHAIN_SCHEMA_VERSION = 1
ONE_EXE_PREFERRED_MAX_MB = 150
ONE_EXE_HARD_STOP_MB = 250


def _command(args: list[str], *, timeout_seconds: float = 4.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout_seconds)
    except FileNotFoundError as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "", "error": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"command timed out after {timeout_seconds:.1f}s",
        }
    except Exception as exc:  # pragma: no cover - defensive OS boundary
        return {"ok": False, "returncode": None, "stdout": "", "stderr": "", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "error": None if completed.returncode == 0 else (completed.stderr.strip() or completed.stdout.strip()),
    }


def _find_tool_path(name: str, extra_dirs: list[Path] | None = None) -> str | None:
    path = shutil.which(name)
    if path:
        return path
    for directory in extra_dirs or []:
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    return None


def _tool(name: str, version_args: list[str] | None = None, *, extra_dirs: list[Path] | None = None) -> dict[str, Any]:
    path = _find_tool_path(name, extra_dirs)
    result: dict[str, Any] = {"name": name, "available": path is not None, "path": path, "version": None, "error": None}
    if path and version_args:
        version = _command([path, *version_args], timeout_seconds=4.0)
        if version["ok"] or version["stdout"] or version["stderr"]:
            text = (version["stdout"] or version["stderr"]).strip().splitlines()
            result["version"] = text[0].strip() if text else None
        if not version["ok"] and not result["version"]:
            result["error"] = version.get("error")
    return result


def _vswhere_path() -> Path | None:
    candidates = []
    program_files_x86 = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    candidates.append(Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if raw:
            candidates.append(Path(raw) / "vswhere.exe")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _visual_studio_installations() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("VSINSTALLDIR", "VCINSTALLDIR"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    vswhere = _vswhere_path()
    if vswhere is not None:
        result = _command(
            [
                str(vswhere),
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            timeout_seconds=5.0,
        )
        if result["stdout"]:
            roots.extend(Path(line.strip()) for line in result["stdout"].splitlines() if line.strip())
    roots.extend(
        [
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"),
            Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional"),
            Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"),
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            resolved = root.expanduser()
        key = str(resolved).casefold()
        if key not in seen and resolved.exists():
            seen.add(key)
            unique.append(resolved)
    return unique


def _probe_msvc() -> dict[str, Any]:
    roots = _visual_studio_installations()
    tool_dirs_paths: list[Path] = []
    for root in roots:
        tool_dirs_paths.extend(Path(path) for path in glob.glob(str(root / "VC" / "Tools" / "MSVC" / "*" / "bin" / "Hostx64" / "x64")))
    cl = _tool("cl.exe", extra_dirs=tool_dirs_paths)
    link = _tool("link.exe", extra_dirs=tool_dirs_paths)
    available = bool(roots) or bool(cl["available"] and link["available"])
    return {
        "available": available,
        "vswhere": str(_vswhere_path()) if _vswhere_path() is not None else None,
        "installations": [str(path) for path in roots],
        "tool_dirs": sorted(str(path) for path in tool_dirs_paths),
        "cl": cl,
        "link": link,
        "message": "MSVC Build Tools found." if available else "MSVC Build Tools were not found on PATH or in standard install locations.",
    }


def _sdk_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("WindowsSdkDir", "WindowsSDKDir"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    roots.extend(
        [
            Path(r"C:\Program Files (x86)\Windows Kits\10"),
            Path(r"C:\Program Files\Windows Kits\10"),
            Path(r"C:\Program Files (x86)\Windows Kits\11"),
            Path(r"C:\Program Files\Windows Kits\11"),
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            resolved = root.expanduser()
        key = str(resolved).casefold()
        if key not in seen and resolved.exists():
            seen.add(key)
            unique.append(resolved)
    return unique


def _latest_existing(patterns: list[Path]) -> str | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(path) for path in glob.glob(str(pattern)))
    files = [path for path in matches if path.is_file()]
    if not files:
        return None
    return str(sorted(files, key=lambda path: str(path))[-1])


def _probe_windows_sdk() -> dict[str, Any]:
    roots = _sdk_roots()
    include_patterns: list[Path] = []
    lib_patterns: list[Path] = []
    for root in roots:
        include_patterns.extend([root / "Include" / "*" / "um" / "d3d12.h", root / "Include" / "*" / "um" / "dxgi1_6.h", root / "Include" / "*" / "shared" / "dxgi1_6.h"])
        lib_patterns.extend([root / "Lib" / "*" / "um" / "x64" / "d3d12.lib", root / "Lib" / "*" / "um" / "x64" / "dxgi.lib"])
    d3d12_header = _latest_existing([root / "Include" / "*" / "um" / "d3d12.h" for root in roots])
    dxgi_header = _latest_existing([root / "Include" / "*" / "um" / "dxgi1_6.h" for root in roots] + [root / "Include" / "*" / "shared" / "dxgi1_6.h" for root in roots])
    d3d12_lib = _latest_existing([root / "Lib" / "*" / "um" / "x64" / "d3d12.lib" for root in roots])
    dxgi_lib = _latest_existing([root / "Lib" / "*" / "um" / "x64" / "dxgi.lib" for root in roots])
    available = all((d3d12_header, dxgi_header, d3d12_lib, dxgi_lib))
    return {
        "available": available,
        "roots": [str(root) for root in roots],
        "directx_headers": {"d3d12_h": d3d12_header, "dxgi1_6_h": dxgi_header},
        "directx_libs": {"d3d12_lib": d3d12_lib, "dxgi_lib": dxgi_lib},
        "message": "Windows SDK DirectX headers/libs found." if available else "Windows SDK DirectX headers/libs are incomplete or missing.",
    }


def _probe_shader_compilers() -> dict[str, Any]:
    sdk_bin_dirs: list[Path] = []
    for root in _sdk_roots():
        sdk_bin_dirs.extend(Path(path) for path in glob.glob(str(root / "bin" / "*" / "x64")))
        sdk_bin_dirs.extend(Path(path) for path in glob.glob(str(root / "bin" / "x64")))
    dxc = _tool("dxc.exe", ["--help"], extra_dirs=sdk_bin_dirs)
    fxc = _tool("fxc.exe", ["/?"], extra_dirs=sdk_bin_dirs)
    return {
        "available": bool(dxc["available"] or fxc["available"]),
        "dxc": dxc,
        "fxc": fxc,
        "build_time_policy": "Use DXC/FXC at build time only; packaged EXEs must not depend on shader compiler binaries unless explicitly approved later.",
    }


def _probe_dependency_audit() -> dict[str, Any]:
    vs_tool_dirs: list[Path] = []
    for root in _visual_studio_installations():
        vs_tool_dirs.extend(Path(path) for path in glob.glob(str(root / "VC" / "Tools" / "MSVC" / "*" / "bin" / "Hostx64" / "x64")))
        ide = root / "Common7" / "IDE" / "VC" / "VCPackages"
        if ide.exists():
            vs_tool_dirs.append(ide)
    dumpbin = _tool("dumpbin.exe", ["/?"], extra_dirs=vs_tool_dirs)
    llvm_objdump = _tool("llvm-objdump.exe", ["--version"])
    return {
        "available": bool(dumpbin["available"] or llvm_objdump["available"]),
        "dumpbin": dumpbin,
        "llvm_objdump": llvm_objdump,
        "message": "Native dependency audit tool found." if dumpbin["available"] or llvm_objdump["available"] else "dumpbin/llvm-objdump not found; native dependency audits need a developer prompt or LLVM tools.",
    }


def _probe_vulkan(*, enabled: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": enabled,
        "available": False,
        "vulkan_sdk": os.environ.get("VULKAN_SDK"),
        "sdk_roots": [],
        "headers": None,
        "import_lib": None,
        "loader_dll": None,
        "strategy": "Do not ship the Vulkan SDK. Build against headers/import lib only when present, compile SPIR-V at build time, and runtime-load the installed Vulkan loader/driver.",
        "shader_policy": "Use DXC -spirv at build time only; packaged apps must not depend on DXC, shader compiler binaries, or Vulkan SDK files.",
        "validation_layers_policy": "Validation layers are development-only diagnostics and must never be packaged.",
        "message": "Vulkan toolchain probe disabled.",
    }
    if not enabled:
        return result
    sdk_roots: list[Path] = []
    for env_name in ("VULKAN_SDK", "VK_SDK_PATH"):
        value = os.environ.get(env_name)
        if value:
            sdk_roots.append(Path(value))
    sdk_roots.extend(Path(path) for path in glob.glob(r"C:\VulkanSDK\*"))
    program_files = os.environ.get("ProgramFiles") or r"C:\Program Files"
    program_files_x86 = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    sdk_roots.extend(Path(path) for path in glob.glob(str(Path(program_files) / "VulkanSDK" / "*")))
    sdk_roots.extend(Path(path) for path in glob.glob(str(Path(program_files_x86) / "VulkanSDK" / "*")))
    unique_sdk_roots: list[Path] = []
    seen: set[str] = set()
    for root in sdk_roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            resolved = root.expanduser()
        key = str(resolved).casefold()
        if key not in seen and resolved.exists():
            seen.add(key)
            unique_sdk_roots.append(resolved)
    header_candidates = []
    lib_candidates = []
    for sdk in unique_sdk_roots:
        header_candidates.append(sdk / "Include" / "vulkan" / "vulkan.h")
        lib_candidates.append(sdk / "Lib" / "vulkan-1.lib")
        lib_candidates.append(sdk / "Lib" / "vulkan" / "vulkan-1.lib")
    loader_candidates = []
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    loader_candidates.append(system_root / "System32" / "vulkan-1.dll")
    loader_candidates.append(system_root / "SysWOW64" / "vulkan-1.dll")
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if raw:
            loader_candidates.append(Path(raw) / "vulkan-1.dll")
    header = next((path for path in header_candidates if path.is_file()), None)
    import_lib = next((path for path in lib_candidates if path.is_file()), None)
    loader = next((path for path in loader_candidates if path.is_file()), None)
    result.update(
        {
            "sdk_roots": [str(path) for path in unique_sdk_roots],
            "headers": str(header) if header else None,
            "import_lib": str(import_lib) if import_lib else None,
            "loader_dll": str(loader) if loader else None,
            "available": bool(header and import_lib and loader),
            "message": "Vulkan headers/import lib/loader found." if header and import_lib and loader else "Vulkan headers/import lib/loader are incomplete or missing.",
        }
    )
    return result


def _one_exe_decision(components: dict[str, Any]) -> dict[str, Any]:
    del components
    return {
        "status": "deferred",
        "approved": False,
        "preferred_max_mb": ONE_EXE_PREFERRED_MAX_MB,
        "hard_stop_mb": ONE_EXE_HARD_STOP_MB,
        "reason": "D3D12 MVP binaries may be built locally, but release one-EXE merge still needs size measurements, dependency audit, sanitized-PATH EXE fallback evidence, and explicit approval.",
        "policy": "Keep the current lightweight ImgKey.exe plus optional ImgKey-GPU.exe policy until backend evidence satisfies the size/dependency/fallback gates.",
        "spec_changes_required_now": False,
    }


def probe_native_toolchain(*, vulkan_enabled: bool | None = None) -> dict[str, Any]:
    if vulkan_enabled is None:
        raw_vulkan_enabled = os.environ.get("IMGKEY_ENABLE_VULKAN_PROBE")
        vulkan_enabled = True if raw_vulkan_enabled is None else str(raw_vulkan_enabled).strip().lower() in {"1", "true", "yes", "on"}
    components = {
        "msvc": _probe_msvc(),
        "windows_sdk": _probe_windows_sdk(),
        "shader_compilers": _probe_shader_compilers(),
        "vulkan": _probe_vulkan(enabled=bool(vulkan_enabled)),
        "dependency_audit": _probe_dependency_audit(),
    }
    required_ready = bool(components["msvc"]["available"] and components["windows_sdk"]["available"] and components["shader_compilers"]["available"] and components["dependency_audit"]["available"])
    status = "ready" if required_ready else "incomplete"
    report = {
        "schema_version": TOOLCHAIN_SCHEMA_VERSION,
        "probe": "imgkey_native_gpu_toolchain",
        "status": status,
        "platform": sys.platform,
        "components": components,
        "vulkan_gate": {
            "status": "ready" if components["vulkan"].get("available") else ("disabled" if not components["vulkan"].get("enabled") else "blocked"),
            "reason": None if components["vulkan"].get("available") else ("vulkan_probe_disabled" if not components["vulkan"].get("enabled") else "vulkan_toolchain_incomplete"),
            "message": components["vulkan"].get("message"),
        },
        "packaging_decision": _one_exe_decision(components),
    }
    report["message"] = "Native D3D12 build/audit toolchain appears ready." if status == "ready" else "Native D3D12 build/audit toolchain is incomplete on this machine."
    return report
