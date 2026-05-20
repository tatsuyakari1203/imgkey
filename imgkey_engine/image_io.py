from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def read_image_rgb(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        image = ImageOps.exif_transpose(Image.open(path))
    except Exception as exc:
        raise ValueError(f"Cannot read image: {path}") from exc

    has_alpha = image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3].copy()
    original_alpha = rgba[:, :, 3].astype(np.float32) / 255.0 if has_alpha else None
    return rgb, original_alpha


def write_png_rgba(path: str | Path, rgba: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    ok, encoded = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError(f"Cannot encode PNG: {path}")
    encoded.tofile(str(path))


def read_grayscale_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a manual keep/remove/imported matte as uint8 grayscale.

    If ``shape`` is supplied, the mask is resized with nearest-neighbor
    interpolation so brush/import tools can pass it directly to the engine.
    """

    try:
        mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Cannot read mask: {path}") from exc
    if shape is not None and mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def read_imported_matte_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read an imported foreground-protection matte as uint8 grayscale."""

    return read_grayscale_mask(path, shape)


def read_alpha_hint_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Backward-compatible alias for imported matte loading."""

    return read_imported_matte_mask(path, shape)


def write_grayscale_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    Image.fromarray(np.clip(mask, 0, 255).astype(np.uint8), mode="L").save(path)


def resize_for_preview(rgb: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return rgb.copy(), 1.0
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    out = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale


def checkerboard_composite(rgba: np.ndarray, cell: int = 18) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    h, w = rgba.shape[:2]
    yy, xx = np.indices((h, w))
    board = ((xx // cell + yy // cell) % 2).astype(np.float32)
    bg = (0.78 + board[:, :, None] * 0.14).astype(np.float32)
    out = rgb * a + bg * (1.0 - a)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
