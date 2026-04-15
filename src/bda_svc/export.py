"""Export utilities."""

import datetime
import json
import uuid
from pathlib import Path

from PIL import Image

from bda_svc import constants
from bda_svc.pipeline.utilities import crop_with_buffer, draw_box_overlay


def build_report(
    bda: dict, image_path: str | Path, model_name: str, inference_time: float
) -> dict:
    """Build report IAW JSON schema.

    Args:
        bda: BDA analysis dictionary.
        image_path: Path of the original image.
        model_name: Model name metadata.
        inference_time: Inference time metadata.
    """
    image_path = Path(image_path)

    return {
        "metadata": {
            "model_name": model_name,
            "image_id": str(uuid.uuid4()),
            "image_filename": image_path.name,
            "date_created": datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "location": {"crs": "", "coordinates": ""},
            "report_type": "PDA",
            "analyst": "bda-svc",
            "inference_time": f"{inference_time:.2f}",
        },
        "physical_damage": bda.get("physical_damage", {}),
        "summary": bda.get("summary", ""),
    }


def save_json(
    bda: dict,
    image_path: str | Path,
    output_path: str | Path | None,
    model_name: str,
    inference_time: float,
    *,
    timestamp: str | None = None,
) -> Path:
    """Save BDA as a JSON file.

    Args:
        bda: BDA analysis dictionary.
        image_path: Path of the original image.
        output_path: Path of output folder. Uses default if None/empty.
        model_name: Model name metadata.
        inference_time: Inference time metadata.
        timestamp: Optional output timestamp reused across debug artifacts.

    Returns:
        Path to the written JSON report.
    """
    image_path = Path(image_path)
    output_path = Path(output_path or constants.DEFAULT_OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    report = build_report(bda, image_path, model_name, inference_time)

    timestamp = timestamp or datetime.datetime.now(datetime.UTC).strftime(
        "%Y-%m-%d_%H%M%SZ"
    )
    json_path = output_path / f"{image_path.stem}_{timestamp}.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[*] Exported: {json_path}")
    return json_path


def save_debug_images(
    bda: dict,
    image_path: str | Path,
    output_path: str | Path | None,
    *,
    timestamp: str,
    crop_buffer_ratio: float,
) -> Path | None:
    """Save temporary debug overlay/crop images for each detected target.

    This export path is intentionally marked as temporary to support prompt
    iteration. Remove it once prompt tuning is finalized.

    Args:
        bda: BDA analysis dictionary.
        image_path: Path to the original image.
        output_path: Base output folder.
        timestamp: Shared timestamp used for sibling JSON/debug artifacts.
        crop_buffer_ratio: Padding ratio applied to saved debug crops.

    Returns:
        Path to the debug directory, or `None` if no target debug images were
        written.
    """
    image_path = Path(image_path)
    output_path = Path(output_path or constants.DEFAULT_OUTPUT_PATH)
    targets = bda.get("physical_damage", {})

    debug_dir = output_path / f"{image_path.stem}_{timestamp}_debug"
    wrote_any = False

    with Image.open(image_path).convert("RGB") as image:
        for target_id, target in targets.items():
            bbox = target.get("bounding_box")
            if not isinstance(bbox, list | tuple) or len(bbox) != 4:
                continue

            try:
                box = tuple(int(value) for value in bbox)
            except (TypeError, ValueError):
                continue

            xmin, ymin, xmax, ymax = box
            if xmin >= xmax or ymin >= ymax:
                continue

            debug_dir.mkdir(parents=True, exist_ok=True)
            overlay = draw_box_overlay(image, box)
            overlay_path = debug_dir / f"{target_id}_overlay.jpg"
            overlay.save(overlay_path, quality=95)

            crop = crop_with_buffer(image, box, crop_buffer_ratio)
            crop_path = debug_dir / f"{target_id}_crop.jpg"
            crop.save(crop_path, quality=95)
            wrote_any = True

    if wrote_any:
        return debug_dir

    return None


def save_debug_payloads(
    bda: dict,
    image_path: str | Path,
    output_path: str | Path | None,
    *,
    timestamp: str,
) -> Path | None:
    """Save temporary pipeline debug payloads alongside visual debug artifacts.

    Args:
        bda: BDA analysis dictionary.
        image_path: Path to the original image.
        output_path: Base output folder.
        timestamp: Shared timestamp used for sibling JSON/debug artifacts.

    Returns:
        Path to the debug directory, or `None` if there was no debug payload to
        export.
    """
    image_path = Path(image_path)
    output_path = Path(output_path or constants.DEFAULT_OUTPUT_PATH)
    debug_payload = bda.get("_debug")
    if not isinstance(debug_payload, dict) or not debug_payload:
        return None

    debug_dir = output_path / f"{image_path.stem}_{timestamp}_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / "pipeline_debug.json"
    debug_path.write_text(json.dumps(debug_payload, indent=4), encoding="utf-8")
    return debug_dir


def save_outputs(
    bda: dict,
    image_path: str | Path,
    output_path: str | Path | None,
    model_name: str,
    *,
    inference_time: float = 0.0,
    debug_export_images: bool = False,
    crop_buffer_ratio: float = 0.0,
) -> Path:
    """Save the main JSON report and optional temporary debug exports.

    Args:
        bda: BDA analysis dictionary.
        image_path: Path of the original image.
        output_path: Path of output folder. Uses default if None/empty.
        model_name: Model name metadata.
        inference_time: Inference time metadata for the main JSON report.
        debug_export_images: Whether to save temporary overlay/crop artifacts
            and any available pipeline debug payloads.
        crop_buffer_ratio: Padding ratio applied to saved debug crops.

    Returns:
        Path to the written JSON report.
    """
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d_%H%M%SZ")
    json_path = save_json(
        bda,
        image_path,
        output_path,
        model_name,
        inference_time,
        timestamp=timestamp,
    )

    if debug_export_images:
        debug_dir = save_debug_payloads(
            bda,
            image_path,
            output_path,
            timestamp=timestamp,
        )
        image_debug_dir = save_debug_images(
            bda,
            image_path,
            output_path,
            timestamp=timestamp,
            crop_buffer_ratio=crop_buffer_ratio,
        )
        if debug_dir or image_debug_dir:
            print(
                f"[*] Exported temporary debug artifacts: "
                f"{debug_dir or image_debug_dir} "
                "(remove after prompt tuning is finalized)"
            )

    return json_path
