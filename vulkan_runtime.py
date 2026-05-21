from __future__ import annotations

import copy
import ctypes
import os
from pathlib import Path
import struct
import sys
from typing import Any


VULKAN_RUNTIME_SCHEMA_VERSION = 1
VULKAN_LOADER_NAME = "vulkan-1.dll" if sys.platform == "win32" else "libvulkan.so.1"

_VULKAN_RUNTIME_PROBE_CACHE: dict[str, dict[str, Any]] = {}

VK_SUCCESS = 0
VK_QUEUE_GRAPHICS_BIT = 0x00000001
VK_QUEUE_COMPUTE_BIT = 0x00000002
VK_QUEUE_TRANSFER_BIT = 0x00000004
VK_QUEUE_SPARSE_BINDING_BIT = 0x00000008
VK_QUEUE_PROTECTED_BIT = 0x00000010
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_API_VERSION_1_0 = 1 << 22


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
    ]


class VkQueueFamilyProperties(ctypes.Structure):
    _fields_ = [
        ("queueFlags", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("timestampValidBits", ctypes.c_uint32),
        ("minImageTransferGranularityWidth", ctypes.c_uint32),
        ("minImageTransferGranularityHeight", ctypes.c_uint32),
        ("minImageTransferGranularityDepth", ctypes.c_uint32),
    ]


def _version_tuple(version: int | None) -> list[int] | None:
    if version is None:
        return None
    raw = int(version)
    return [(raw >> 22) & 0x3FF, (raw >> 12) & 0x3FF, raw & 0xFFF]


def _candidate_loader_paths(loader_path: str | os.PathLike[str] | None = None) -> list[Path | str]:
    if loader_path is not None:
        return [Path(loader_path).expanduser()]
    candidates: list[Path | str] = []
    env_loader = os.environ.get("IMGKEY_VULKAN_LOADER")
    if env_loader:
        candidates.append(Path(env_loader).expanduser())
    if sys.platform == "win32":
        system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        candidates.append(system_root / "System32" / VULKAN_LOADER_NAME)
        candidates.append(system_root / "SysWOW64" / VULKAN_LOADER_NAME)
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if raw:
            candidates.append(Path(raw) / VULKAN_LOADER_NAME)
    candidates.append(VULKAN_LOADER_NAME)
    unique: list[Path | str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _load_vulkan_loader(loader_path: str | os.PathLike[str] | None = None) -> tuple[ctypes.CDLL, str, list[str]]:
    checked: list[str] = []
    errors: list[str] = []
    for candidate in _candidate_loader_paths(loader_path):
        candidate_text = str(candidate)
        if isinstance(candidate, Path):
            if not candidate.is_file():
                checked.append(candidate_text)
                continue
            load_arg = str(candidate)
        else:
            load_arg = candidate_text
        try:
            return ctypes.CDLL(load_arg), candidate_text, checked
        except OSError as exc:
            errors.append(f"{candidate_text}: {exc}")
    detail = "; ".join(errors) if errors else "checked " + ", ".join(checked)
    raise OSError(f"{VULKAN_LOADER_NAME} could not be loaded ({detail})")


def _queue_flags(flags: int) -> list[str]:
    out: list[str] = []
    if flags & VK_QUEUE_GRAPHICS_BIT:
        out.append("graphics")
    if flags & VK_QUEUE_COMPUTE_BIT:
        out.append("compute")
    if flags & VK_QUEUE_TRANSFER_BIT:
        out.append("transfer")
    if flags & VK_QUEUE_SPARSE_BINDING_BIT:
        out.append("sparse_binding")
    if flags & VK_QUEUE_PROTECTED_BIT:
        out.append("protected")
    return out


def _result_name(result: int) -> str:
    names = {
        0: "VK_SUCCESS",
        1: "VK_NOT_READY",
        2: "VK_TIMEOUT",
        3: "VK_EVENT_SET",
        4: "VK_EVENT_RESET",
        5: "VK_INCOMPLETE",
        -1: "VK_ERROR_OUT_OF_HOST_MEMORY",
        -2: "VK_ERROR_OUT_OF_DEVICE_MEMORY",
        -3: "VK_ERROR_INITIALIZATION_FAILED",
        -4: "VK_ERROR_DEVICE_LOST",
        -5: "VK_ERROR_MEMORY_MAP_FAILED",
        -6: "VK_ERROR_LAYER_NOT_PRESENT",
        -7: "VK_ERROR_EXTENSION_NOT_PRESENT",
        -8: "VK_ERROR_FEATURE_NOT_PRESENT",
        -9: "VK_ERROR_INCOMPATIBLE_DRIVER",
    }
    return names.get(int(result), f"VkResult({int(result)})")


def _device_type_name(device_type: int) -> str:
    return {
        0: "other",
        1: "integrated_gpu",
        2: "discrete_gpu",
        3: "virtual_gpu",
        4: "cpu",
    }.get(int(device_type), f"unknown_{int(device_type)}")


def _remember_vulkan_probe(cache_key: str, result: dict[str, Any]) -> dict[str, Any]:
    _VULKAN_RUNTIME_PROBE_CACHE[cache_key] = copy.deepcopy(result)
    return copy.deepcopy(result)


def _vendor_name(vendor_id: int) -> str | None:
    return {
        0x10DE: "NVIDIA",
        0x1002: "AMD",
        0x1022: "AMD",
        0x8086: "Intel",
        0x13B5: "Arm",
        0x5143: "Qualcomm",
        0x1010: "ImgTec",
        0x106B: "Apple",
        0x1414: "Microsoft",
    }.get(int(vendor_id))


def _physical_device_properties(vk: ctypes.CDLL, device: ctypes.c_void_p) -> dict[str, Any]:
    vk.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    vk.vkGetPhysicalDeviceProperties.restype = None
    props = ctypes.create_string_buffer(4096)
    vk.vkGetPhysicalDeviceProperties(device, ctypes.byref(props))
    raw = props.raw
    api_version, driver_version, vendor_id, device_id, device_type = struct.unpack_from("<IIIII", raw, 0)
    name_bytes = raw[20 : 20 + 256].split(b"\x00", 1)[0]
    device_name = name_bytes.decode("utf-8", errors="replace") if name_bytes else None
    return {
        "name": device_name,
        "api_version": int(api_version),
        "api_version_tuple": _version_tuple(api_version),
        "driver_version": int(driver_version),
        "vendor_id": int(vendor_id),
        "vendor_name": _vendor_name(vendor_id),
        "device_id": int(device_id),
        "device_type": _device_type_name(device_type),
    }


def _queue_family_properties(vk: ctypes.CDLL, device: ctypes.c_void_p) -> list[dict[str, Any]]:
    vk.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(VkQueueFamilyProperties)]
    vk.vkGetPhysicalDeviceQueueFamilyProperties.restype = None
    count = ctypes.c_uint32(0)
    vk.vkGetPhysicalDeviceQueueFamilyProperties(device, ctypes.byref(count), None)
    if count.value <= 0:
        return []
    array_type = VkQueueFamilyProperties * int(count.value)
    queues = array_type()
    vk.vkGetPhysicalDeviceQueueFamilyProperties(device, ctypes.byref(count), queues)
    out: list[dict[str, Any]] = []
    for index in range(int(count.value)):
        item = queues[index]
        out.append(
            {
                "index": index,
                "queue_count": int(item.queueCount),
                "flags_mask": int(item.queueFlags),
                "flags": _queue_flags(int(item.queueFlags)),
                "compute": bool(item.queueCount and (int(item.queueFlags) & VK_QUEUE_COMPUTE_BIT)),
            }
        )
    return out


def probe_vulkan_runtime(loader_path: str | os.PathLike[str] | None = None, *, refresh: bool = False) -> dict[str, Any]:
    """Runtime-load the Vulkan loader and enumerate compute-capable devices.

    This probe intentionally uses ctypes and the installed Vulkan loader only. It
    does not require Vulkan SDK headers/import libraries and does not request
    validation layers, because validation layers are development-only and must
    never become packaged app dependencies.
    """

    cache_key = "<default>" if loader_path is None else str(Path(loader_path).expanduser())
    if not refresh and cache_key in _VULKAN_RUNTIME_PROBE_CACHE:
        return copy.deepcopy(_VULKAN_RUNTIME_PROBE_CACHE[cache_key])

    result: dict[str, Any] = {
        "schema_version": VULKAN_RUNTIME_SCHEMA_VERSION,
        "probe": "imgkey_vulkan_runtime",
        "status": "unavailable",
        "available": False,
        "reason": "not_run",
        "message": "Vulkan runtime probe has not run.",
        "loader_dll": None,
        "loader_checked": [],
        "api_version": None,
        "api_version_tuple": None,
        "device_count": 0,
        "compute_device_count": 0,
        "devices": [],
        "validation_layers": {
            "requested": False,
            "policy": "Validation layers are development-only diagnostics and are never packaged with ImgKey.",
        },
    }
    instance = ctypes.c_void_p()
    vk: ctypes.CDLL | None = None
    try:
        vk, loader, checked = _load_vulkan_loader(loader_path)
        result["loader_dll"] = loader
        result["loader_checked"] = checked
    except Exception as exc:
        result.update(
            {
                "status": "unavailable",
                "reason": "vulkan_loader_unavailable",
                "message": f"Vulkan loader is unavailable: {type(exc).__name__}: {exc}. CPU/D3D12 fallback remains active.",
            }
        )
        return _remember_vulkan_probe(cache_key, result)

    try:
        api_version = ctypes.c_uint32(VK_API_VERSION_1_0)
        if hasattr(vk, "vkEnumerateInstanceVersion"):
            vk.vkEnumerateInstanceVersion.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
            vk.vkEnumerateInstanceVersion.restype = ctypes.c_int32
            enum_version = int(vk.vkEnumerateInstanceVersion(ctypes.byref(api_version)))
            if enum_version != VK_SUCCESS:
                api_version = ctypes.c_uint32(VK_API_VERSION_1_0)
        result["api_version"] = int(api_version.value)
        result["api_version_tuple"] = _version_tuple(int(api_version.value))

        app_name = b"ImgKey Vulkan Probe"
        engine_name = b"ImgKey"
        app_info = VkApplicationInfo(
            VK_STRUCTURE_TYPE_APPLICATION_INFO,
            None,
            app_name,
            1,
            engine_name,
            1,
            VK_API_VERSION_1_0,
        )
        create_info = VkInstanceCreateInfo(
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            None,
            0,
            ctypes.pointer(app_info),
            0,
            None,
            0,
            None,
        )
        vk.vkCreateInstance.argtypes = [ctypes.POINTER(VkInstanceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        vk.vkCreateInstance.restype = ctypes.c_int32
        create_status = int(vk.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance)))
        if create_status != VK_SUCCESS or not instance.value:
            result.update(
                {
                    "status": "unavailable",
                    "reason": "vulkan_instance_unavailable",
                    "message": f"Vulkan loader loaded, but vkCreateInstance failed with {_result_name(create_status)}. CPU/D3D12 fallback remains active.",
                }
            )
            return _remember_vulkan_probe(cache_key, result)

        vk.vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_void_p)]
        vk.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
        count = ctypes.c_uint32(0)
        enum_status = int(vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None))
        if enum_status != VK_SUCCESS:
            result.update(
                {
                    "status": "unavailable",
                    "reason": "vulkan_device_enumeration_failed",
                    "message": f"Vulkan device enumeration failed with {_result_name(enum_status)}. CPU/D3D12 fallback remains active.",
                }
            )
            return _remember_vulkan_probe(cache_key, result)
        if count.value <= 0:
            result.update(
                {
                    "status": "unavailable",
                    "reason": "vulkan_no_devices",
                    "message": "Vulkan loader is present, but no Vulkan physical devices were reported. CPU/D3D12 fallback remains active.",
                }
            )
            return _remember_vulkan_probe(cache_key, result)

        device_array_type = ctypes.c_void_p * int(count.value)
        device_handles = device_array_type()
        enum_status = int(vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), device_handles))
        if enum_status != VK_SUCCESS:
            result.update(
                {
                    "status": "unavailable",
                    "reason": "vulkan_device_enumeration_failed",
                    "message": f"Vulkan physical device retrieval failed with {_result_name(enum_status)}. CPU/D3D12 fallback remains active.",
                }
            )
            return _remember_vulkan_probe(cache_key, result)

        devices: list[dict[str, Any]] = []
        compute_count = 0
        for index in range(int(count.value)):
            handle = ctypes.c_void_p(device_handles[index])
            properties = _physical_device_properties(vk, handle)
            queues = _queue_family_properties(vk, handle)
            compute_indices = [q["index"] for q in queues if q.get("compute")]
            if compute_indices:
                compute_count += 1
            properties.update(
                {
                    "index": index,
                    "queue_families": queues,
                    "compute_queue_families": compute_indices,
                    "supports_compute": bool(compute_indices),
                }
            )
            devices.append(properties)

        result["devices"] = devices
        result["device_count"] = len(devices)
        result["compute_device_count"] = compute_count
        if compute_count <= 0:
            result.update(
                {
                    "status": "unavailable",
                    "reason": "vulkan_no_compute_queue",
                    "message": "Vulkan devices were found, but none expose a compute queue. CPU/D3D12 fallback remains active.",
                }
            )
            return _remember_vulkan_probe(cache_key, result)
        result.update(
            {
                "status": "available",
                "available": True,
                "reason": None,
                "message": f"Vulkan runtime loader and {compute_count} compute-capable physical device(s) are available.",
            }
        )
        return _remember_vulkan_probe(cache_key, result)
    except Exception as exc:  # pragma: no cover - OS/driver boundary
        result.update(
            {
                "status": "unavailable",
                "available": False,
                "reason": "vulkan_runtime_probe_failed",
                "message": f"Vulkan runtime probe failed: {type(exc).__name__}: {exc}. CPU/D3D12 fallback remains active.",
            }
        )
        return _remember_vulkan_probe(cache_key, result)
    finally:
        if vk is not None and instance.value and hasattr(vk, "vkDestroyInstance"):
            try:
                vk.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                vk.vkDestroyInstance.restype = None
                vk.vkDestroyInstance(instance, None)
            except Exception:
                pass
