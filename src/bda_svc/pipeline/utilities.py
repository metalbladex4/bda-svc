"""A collection of model utility functions."""

from pathlib import Path

import yaml
from PIL import Image, ImageDraw

CONFIG_PATH = Path(__file__).parent / "config.yaml"
DOCTRINE_PATH = Path(__file__).parent / "doctrine.yaml"


def load_yaml(path: Path) -> dict:
    """Load file from YAML.

    Args:
        path: Path to YAML file.

    Returns:
        A dictionary with loaded YAML.
    """
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def format_detection_doctrine(categories: list[str]) -> str:
    """Format detection guidance doctrine for selected categories.

    Args:
        categories: Doctrinal BDA target categories.

    Returns:
        Detection guidance block for the prompt.
    """
    output = []
    doctrine = load_yaml(DOCTRINE_PATH)

    for category in categories:
        section = doctrine.get(category)
        if not isinstance(section, dict):
            continue
        section_entry = section.get("detection_guidance")
        if section_entry is None:
            continue
        output.append(f"{category}:")
        output.append(str(section_entry).strip())

    return "\n\n".join(output)


def format_pda_doctrine(category: str) -> str:
    """Format PDA doctrine for a selected target category.

    Args:
        category: A doctrinal BDA target category.

    Returns:
        A doctrinal PDA string with definitions and considerations.
    """
    doctrine = load_yaml(DOCTRINE_PATH)
    section = doctrine.get(category)

    if not isinstance(section, dict):
        return "NO TARGET DOCTRINE AVAILABLE."

    output = []
    title = category.replace("_", " ").upper()
    for key in ["physical_damage_definitions", "physical_damage_considerations"]:
        section_entry = section.get(key)
        if section_entry is None:
            continue
        section_title = key.replace("_", " ").upper()
        output.append(f"{title} {section_title}")
        output.append(str(section_entry).strip())

    return "\n".join(output)


def bbox_to_pixels(
    original_image: Image.Image,
    model_image: Image.Image,
    bbox: list[int | float],
    bbox_convention: str,
) -> tuple[int, int, int, int] | None:
    """Convert a model bbox into pixel-space coordinates.

    Args:
        original_image: Original source image.
        model_image: Image that was actually sent to the model.
        bbox: Raw bbox returned by the model.
        bbox_convention: Bounding box ordering and scale convention.

    Returns:
        Pixel-space bounding box, or `None` if invalid.
    """
    # Fail safe in invalid
    if len(bbox) != 4:
        return None

    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    if bbox_convention.startswith("xyxy"):
        xmin, ymin, xmax, ymax = values
    elif bbox_convention.startswith("yxyx"):
        ymin, xmin, ymax, xmax = values
    else:
        return None

    if bbox_convention.endswith("_1"):
        max_x = max_y = 1.0
    elif bbox_convention.endswith("_1000"):
        max_x = max_y = 1000.0
    elif bbox_convention.endswith("_pixels"):
        max_x = float(model_image.width)
        max_y = float(model_image.height)
    else:
        return None

    if not (0 <= xmin <= max_x and 0 <= ymin <= max_y):
        return None
    if not (0 <= xmax <= max_x and 0 <= ymax <= max_y):
        return None
    if xmin >= xmax or ymin >= ymax:
        return None

    # Scale pixel-space coordinates
    xmin_px = min(
        max(int(round((xmin / max_x) * original_image.width)), 0),
        original_image.width,
    )
    ymin_px = min(
        max(int(round((ymin / max_y) * original_image.height)), 0),
        original_image.height,
    )
    xmax_px = min(
        max(int(round((xmax / max_x) * original_image.width)), 0),
        original_image.width,
    )
    ymax_px = min(
        max(int(round((ymax / max_y) * original_image.height)), 0),
        original_image.height,
    )

    return xmin_px, ymin_px, xmax_px, ymax_px


def expand_box(
    image: Image.Image,
    box: tuple[int, int, int, int],
    buffer_ratio: float,
    min_size: int = 32,
) -> tuple[int, int, int, int]:
    """Expand a detection box with buffered padding and image-bound clamping.

    Args:
        image: Source image.
        box: Bounding box in integer pixel coordinates.
        buffer_ratio: Fractional padding applied to width and height.
        min_size: Minimum width and height of the returned box.

    Returns:
        Expanded box clamped to the image bounds.
    """
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    height = ymax - ymin

    # Apply buffer
    pad_x = int(round(width * buffer_ratio))
    pad_y = int(round(height * buffer_ratio))

    left = max(0, xmin - pad_x)
    top = max(0, ymin - pad_y)
    right = min(image.width, xmax + pad_x)
    bottom = min(image.height, ymax + pad_y)

    # Enforce minimum size
    cur_w = right - left
    cur_h = bottom - top

    if cur_w < min_size:
        extra = min_size - cur_w
        left -= extra // 2
        right += extra - extra // 2
    if cur_h < min_size:
        extra = min_size - cur_h
        top -= extra // 2
        bottom += extra - extra // 2

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image.width:
        shift = right - image.width
        left -= shift
        right = image.width
    if bottom > image.height:
        shift = bottom - image.height
        top -= shift
        bottom = image.height

    # Final safety clamp
    left = max(0, left)
    top = max(0, top)
    right = min(image.width, right)
    bottom = min(image.height, bottom)

    return left, top, right, bottom


def crop_with_buffer(
    image: Image.Image,
    box: tuple[int, int, int, int],
    buffer_ratio: float,
    min_size: int = 32,
) -> Image.Image:
    """Crop a detection box with a small padding buffer.

    Args:
        image: Source image.
        box: Bounding box in integer pixel coordinates.
        buffer_ratio: Fractional padding applied to width and height.
        min_size: Minimum width and height of the returned crop.

    Returns:
        Cropped image region clamped to the image bounds.
    """
    return image.crop(expand_box(image, box, buffer_ratio, min_size=min_size))


def resize_for_vlm(image: Image.Image, max_side: int) -> Image.Image:
    """Resize image for VLM inference while preserving aspect ratio.

    Args:
        image: Source image.
        max_side: Maximum allowed width or height in pixels.

    Returns:
        Original image if already within bounds, otherwise resized copy.
    """
    if max(image.size) <= max_side:
        return image

    resized = image.copy()
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def draw_box_overlay(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Draw an outline box on a scene image.

    Args:
        image: Source scene image.
        box: Bounding box in integer pixel coordinates.

    Returns:
        Scene image copy with an outlined target box.
    """
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)

    xmin, ymin, xmax, ymax = box
    line_width = max(2, int(round(min(image.width, image.height) * 0.003)))
    draw.rectangle(
        (xmin, ymin, xmax, ymax),
        outline=(0, 255, 0),
        width=line_width,
    )

    return overlay
