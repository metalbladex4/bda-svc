"""Export test suite."""

import copy
import json
import re
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from bda_svc import export


@pytest.fixture
def metadata_std(tmp_path):
    """Metadata fixture for export tests."""
    return {
        "image_path": tmp_path / "image42.png",
        "model_name": "qwen3",
        "inference_time": 3.14,
    }


def get_bda_template():
    """Return a deep copy of the BDA report template."""
    return copy.deepcopy(
        {
            "metadata": {
                "model_name": "",
                "image_id": "",
                "image_filename": "",
                "date_created": "",
                "location": {"crs": "", "coordinates": ""},
                "report_type": "PDA",
                "analyst": "bda-svc",
                "inference_time": "",
            },
            "physical_damage": {"target_0": {}},
            "summary": "---TEST SUMMARY---",
        }
    )


# ----------------------------------------------------------------------
# Test: Export Folder Validation (build_report)
# ----------------------------------------------------------------------


def _test_bda(bda_std, bda_to_test):
    """Test contents of one BDA against another BDA."""
    assert isinstance(bda_to_test["metadata"], dict)
    assert bda_to_test["metadata"]["model_name"] == bda_std["metadata"]["model_name"]

    # Validate that `image_id` is a valid UUID4 value
    regex = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )
    uuid4_valid = re.search(regex, bda_to_test["metadata"]["image_id"])

    assert uuid4_valid is not None

    assert (
        bda_to_test["metadata"]["image_filename"]
        == bda_std["metadata"]["image_path"].name
    )
    assert datetime.fromisoformat(bda_to_test["metadata"]["date_created"])
    assert isinstance(bda_to_test["metadata"]["location"], dict)
    assert bda_to_test["metadata"]["report_type"] == "PDA"
    assert bda_to_test["metadata"]["analyst"] == "bda-svc"
    assert (
        bda_to_test["metadata"]["inference_time"]
        == f"{bda_std['metadata']['inference_time']:.2f}"
    )
    assert bda_to_test["physical_damage"] == bda_std["physical_damage"]
    assert bda_to_test["summary"] == bda_std["summary"]


def test_build_report(metadata_std):
    """Test if our partial BDA gets converted to a full BDA."""
    # Build the BDA to compare to
    bda_std = get_bda_template()
    bda_std["metadata"].update(metadata_std)

    bda_to_test = export.build_report(bda_std, **metadata_std)

    _test_bda(bda_std, bda_to_test)


# ----------------------------------------------------------------------
# Test: Export Folder Validation (save_json)
# ----------------------------------------------------------------------


def test_save_json(tmp_path, metadata_std):
    """Test if valid JSON file created."""
    bda_std = get_bda_template()
    bda_std["metadata"].update(metadata_std)

    json_path = export.save_json(bda_std, output_path=tmp_path, **metadata_std)

    assert isinstance(json_path, Path)

    with json_path.open("r", encoding="utf-8") as file:
        bda_to_test = json.load(file)

        _test_bda(bda_std, bda_to_test)


def make_bda() -> dict:
    """Return a minimal BDA payload for save_outputs tests."""
    return {
        "summary": "1 military equipment (1 destroyed) is visible in the scene.",
        "physical_damage": {
            "target_0": {
                "target_type": "military_equipment",
                "damage_category": "DESTROYED",
                "confidence_level": "CONFIRMED",
                "brief_supporting_logic": "active fire; heavy smoke",
                "bounding_box": [10, 10, 30, 30],
            }
        },
    }


def test_save_outputs_without_debug_images_writes_json_only(tmp_path) -> None:
    """Normal export should write JSON without creating debug artifacts."""
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    json_path = export.save_outputs(
        make_bda(),
        image_path,
        tmp_path / "out",
        "detection=model;assessment=model",
    )

    assert json_path.exists()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["physical_damage"]["target_0"]["damage_category"] == "DESTROYED"

    debug_dirs = list((tmp_path / "out").glob("*_debug"))
    assert debug_dirs == []


def test_save_outputs_with_debug_images_writes_overlay_and_crop(tmp_path) -> None:
    """Temporary debug export should write per-target overlay and crop images."""
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    json_path = export.save_outputs(
        make_bda(),
        image_path,
        tmp_path / "out",
        "detection=model;assessment=model",
        debug_export_images=True,
        crop_buffer_ratio=0.25,
    )

    assert json_path.exists()

    debug_dirs = list((tmp_path / "out").glob("*_debug"))
    assert len(debug_dirs) == 1

    debug_dir = debug_dirs[0]
    overlay_path = debug_dir / "target_0_overlay.jpg"
    crop_path = debug_dir / "target_0_crop.jpg"
    assert overlay_path.exists()
    assert crop_path.exists()

    overlay = Image.open(overlay_path)
    crop = Image.open(crop_path)
    assert overlay.size == (100, 100)
    assert crop.size == (32, 32)


def test_save_outputs_skips_invalid_debug_bbox(tmp_path) -> None:
    """Invalid placeholder boxes should not produce debug images."""
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (100, 100), color="white").save(image_path)

    bda = {
        "summary": "",
        "physical_damage": {
            "target_0": {
                "target_type": "object_not_found",
                "damage_category": "NOT APPLICABLE",
                "confidence_level": "CONFIRMED",
                "brief_supporting_logic": "No visible targets in image.",
                "bounding_box": [0, 0, 0, 0],
            }
        },
    }

    export.save_outputs(
        bda,
        image_path,
        tmp_path / "out",
        "detection=model;assessment=model",
        debug_export_images=True,
        crop_buffer_ratio=0.25,
    )

    debug_dirs = list((tmp_path / "out").glob("*_debug"))
    assert debug_dirs == []
