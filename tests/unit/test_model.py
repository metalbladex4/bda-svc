"""Pipeline model tests."""

import json
from pathlib import Path

import pytest
from PIL import Image

from bda_svc.pipeline import model as pipeline_model
from bda_svc.pipeline.interfaces import Detection

# ----------------------------------------------------------------------
# Test Setup: Create fake backend
# ----------------------------------------------------------------------


class FakeVLM:
    """Small fake VLM backend that returns scripted responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Store scripted responses and record each call."""
        self.responses = responses or []
        self.calls = []

    def generate(
        self,
        image: Image.Image | list[Image.Image],
        prompt: str,
        system_prompt: str | None = None,
        format_schema: dict | None = None,
        temperature: float | None = None,
    ) -> str:
        """Return next scripted response and record the call."""
        if isinstance(image, list):
            image_size = [img.size for img in image]
        else:
            image_size = image.size
        self.calls.append(
            {
                "image_size": image_size,
                "prompt": prompt,
                "system_prompt": system_prompt,
                "format_schema": format_schema,
                "temperature": temperature,
            }
        )
        return self.responses.pop(0)


# ----------------------------------------------------------------------
# Test Setup: Monkeypatch Helpers
# ----------------------------------------------------------------------


def patch_backends(
    monkeypatch: pytest.MonkeyPatch,
    detection_vlm: FakeVLM,
    assessment_vlm: FakeVLM,
    model_names: list[str],
) -> None:
    """Replace `OllamaVLM` with `FakeVLM` for pipeline tests."""
    fake_instances = [detection_vlm, assessment_vlm]

    def fake_ollama_vlm(model: str) -> FakeVLM:
        """Record the model name and return the next fake backend."""
        model_names.append(model)
        return fake_instances.pop(0)

    monkeypatch.setattr(pipeline_model, "OllamaVLM", fake_ollama_vlm)


def patch_config_overrides(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict,
) -> None:
    """Patch `config.yaml` for a test while keeping `doctrine.yaml`."""
    original_load_yaml = pipeline_model.load_yaml
    config = original_load_yaml(pipeline_model.CONFIG_PATH)

    for section, values in overrides.items():
        config[section].update(values)

    def fake_load_yaml(path):
        if path == pipeline_model.CONFIG_PATH:
            return config
        return original_load_yaml(path)

    monkeypatch.setattr(pipeline_model, "load_yaml", fake_load_yaml)


# ----------------------------------------------------------------------
# Initialization tests
# ----------------------------------------------------------------------


def test_init_uses_config_models_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use config model names when environment variables are absent."""
    # Set VLM models in fake config
    patch_config_overrides(
        monkeypatch,
        {
            "detection_vlm": {"model": "config-detection-model"},
            "assessment_vlm": {"model": "config-assessment-model"},
        },
    )
    # Remove environment variables
    monkeypatch.delenv("BDA_DETECTION_MODEL", raising=False)
    monkeypatch.delenv("BDA_ASSESSMENT_MODEL", raising=False)

    model_names = []
    patch_backends(
        monkeypatch,
        detection_vlm=FakeVLM(),
        assessment_vlm=FakeVLM(),
        model_names=model_names,
    )
    pipeline_model.BDAPipeline()
    assert model_names == ["config-detection-model", "config-assessment-model"]


def test_init_uses_env_model_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use environment model names when provided instead of config."""
    # Set VLM models in fake config
    patch_config_overrides(
        monkeypatch,
        {
            "detection_vlm": {"model": "config-detection-model"},
            "assessment_vlm": {"model": "config-assessment-model"},
        },
    )
    # Set environment variables
    monkeypatch.setenv("BDA_DETECTION_MODEL", "env-detection-model")
    monkeypatch.setenv("BDA_ASSESSMENT_MODEL", "env-assessment-model")

    model_names = []
    patch_backends(
        monkeypatch,
        detection_vlm=FakeVLM(),
        assessment_vlm=FakeVLM(),
        model_names=model_names,
    )
    pipeline_model.BDAPipeline()
    assert model_names == ["env-detection-model", "env-assessment-model"]


# ----------------------------------------------------------------------
# Object detection tests
# ----------------------------------------------------------------------


def test_detect_objects_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return a detected target with image coordinates and a crop."""
    # Ensure bbox_convention is xyxy_1000 for tests instead of relying on config.yaml
    patch_config_overrides(
        monkeypatch,
        {"detection_vlm": {"bbox_convention": "xyxy_1000"}},
    )

    # Create a fake detection VLM
    detection_vlm = FakeVLM(
        [
            json.dumps(
                {
                    "detections": [
                        {"target_type": "buildings", "bbox": [100, 100, 500, 500]}
                    ]
                }
            )
        ]
    )
    patch_backends(
        monkeypatch,
        detection_vlm=detection_vlm,
        assessment_vlm=FakeVLM(),
        model_names=[],
    )

    pipeline = pipeline_model.BDAPipeline()
    detections = pipeline.detect_objects(Image.new("RGB", (4000, 2000)))

    # Happy path should return a detection
    assert len(detections) == 1
    assert detections[0].label == "buildings"
    # Detections should come back in original image pixels
    assert detections[0].bbox == (400, 200, 2000, 1000)
    # A buffered crop is attached to the detection
    assert detections[0].crop is not None


def test_detect_objects_fail_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return no detections when the model output is invalid."""
    # Create a fake detection VLM
    detection_vlm = FakeVLM([json.dumps([{"target_type": "buildings"}])])
    patch_backends(
        monkeypatch,
        detection_vlm=detection_vlm,
        assessment_vlm=FakeVLM(),
        model_names=[],
    )

    pipeline = pipeline_model.BDAPipeline()
    detections = pipeline.detect_objects(Image.new("RGB", (4000, 2000)))

    # A bad detection should return empty list instead of an exception
    assert detections == []
    assert (
        pipeline.last_debug_info["detection"]["raw_response"]
        == '[{"target_type": "buildings"}]'
    )
    assert "validation_error" in pipeline.last_debug_info["detection"]


def test_detect_objects_skips_bad_detections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip invalid detections while keeping valid ones."""
    # Ensure bbox_convention is xyxy_1000 for tests instead of relying on config.yaml
    patch_config_overrides(
        monkeypatch,
        {"detection_vlm": {"bbox_convention": "xyxy_1000"}},
    )

    # Create a fake detection VLM
    detection_vlm = FakeVLM(
        [
            json.dumps(
                {
                    "detections": [
                        # target_type is not doctrinal -> skip
                        {"target_type": "fake_category", "bbox": [100, 100, 500, 500]},
                        # bounding box is not valid -> skip
                        {"target_type": "buildings", "bbox": [-1, -1, -1, -1]},
                        {"target_type": "buildings", "bbox": [100, 100, 500, 500]},
                    ]
                }
            )
        ]
    )
    patch_backends(
        monkeypatch,
        detection_vlm=detection_vlm,
        assessment_vlm=FakeVLM(),
        model_names=[],
    )

    pipeline = pipeline_model.BDAPipeline()
    detections = pipeline.detect_objects(Image.new("RGB", (4000, 2000)))

    # There should only be one valid detection
    assert len(detections) == 1
    assert detections[0].label == "buildings"
    assert detections[0].bbox == (400, 200, 2000, 1000)
    assert detections[0].crop is not None


def test_detect_objects_refines_bbox_inside_roi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use a second pass inside an ROI when refinement is enabled."""
    patch_config_overrides(
        monkeypatch,
        {
            "detection_vlm": {
                "bbox_convention": "xyxy_1000",
                "max_image_size": 4096,
                "refinement_enabled": True,
                "refinement_roi_buffer_ratio": 0.25,
            }
        },
    )

    detection_vlm = FakeVLM(
        [
            json.dumps(
                {
                    "detections": [
                        {"target_type": "buildings", "bbox": [100, 100, 500, 500]}
                    ]
                }
            ),
            json.dumps(
                {
                    "detections": [
                        {"target_type": "buildings", "bbox": [200, 200, 700, 700]}
                    ]
                }
            ),
        ]
    )
    patch_backends(
        monkeypatch,
        detection_vlm=detection_vlm,
        assessment_vlm=FakeVLM(),
        model_names=[],
    )

    pipeline = pipeline_model.BDAPipeline()
    detections = pipeline.detect_objects(Image.new("RGB", (4000, 2000)))

    assert len(detections) == 1
    assert detections[0].bbox == (480, 240, 1680, 840)
    refinement = pipeline.last_debug_info["detection"]["refinement"]
    assert refinement["enabled"] is True
    assert refinement["attempts"][0]["decision"] == "selected_refined_bbox"
    assert refinement["attempts"][0]["selected_bbox"] == [480, 240, 1680, 840]


# ----------------------------------------------------------------------
# Target assessment tests
# ----------------------------------------------------------------------


def test_assess_detection_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return a structured assessment for one detected target."""
    assessment_vlm = FakeVLM(
        [
            json.dumps(
                {
                    "damage_category": "destroyed",
                    "confidence_level": "confirmed",
                    "brief_supporting_logic": "roof collapse; burn damage",
                }
            )
        ]
    )
    patch_backends(
        monkeypatch,
        detection_vlm=FakeVLM(),
        assessment_vlm=assessment_vlm,
        model_names=[],
    )

    pipeline = pipeline_model.BDAPipeline()
    detection = Detection(
        label="buildings",
        bbox=(10, 20, 30, 40),
        crop=Image.new("RGB", (50, 50)),
    )

    result = pipeline.assess_detection(detection)

    assert result == {
        "target_type": "buildings",
        "damage_category": "DESTROYED",
        "confidence_level": "CONFIRMED",
        "brief_supporting_logic": "roof collapse; burn damage",
        "bounding_box": [10, 20, 30, 40],
    }


def test_assess_detection_fail_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return no assessment when the model output is invalid."""
    assessment_vlm = FakeVLM([json.dumps({"damage_category": "destroyed"})])
    patch_backends(
        monkeypatch,
        detection_vlm=FakeVLM(),
        assessment_vlm=assessment_vlm,
        model_names=[],
    )

    pipeline = pipeline_model.BDAPipeline()
    detection = Detection(
        label="buildings",
        bbox=(10, 20, 30, 40),
        crop=Image.new("RGB", (50, 50)),
    )

    result = pipeline.assess_detection(detection)

    # A bad assessment should return None instead of an exception
    assert result is None


# ----------------------------------------------------------------------
# Scene summary tests
# ----------------------------------------------------------------------


def test_summarize_scene_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return a scene summary based on the image and assessed targets."""
    assessment_vlm = FakeVLM(["Concise scene summary."])
    patch_backends(
        monkeypatch,
        detection_vlm=FakeVLM(),
        assessment_vlm=assessment_vlm,
        model_names=[],
    )

    pipeline = pipeline_model.BDAPipeline()
    targets = [
        {
            "target_type": "buildings",
            "damage_category": "DESTROYED",
            "confidence_level": "CONFIRMED",
            "brief_supporting_logic": "roof collapse; wall breach",
            "bounding_box": [10, 20, 30, 40],
        }
    ]

    summary = pipeline.summarize_scene(Image.new("RGB", (4000, 2000)), targets)

    # Happy path should return scene summary
    assert summary == "Concise scene summary."
    # Previous target assessments should be in the prompt
    assert json.dumps(targets, indent=2) in assessment_vlm.calls[0]["prompt"]


# ----------------------------------------------------------------------
# Overall pipeline tests
# ----------------------------------------------------------------------


def test_analyze_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Return a scene summary and assessed targets."""
    patch_config_overrides(
        monkeypatch,
        {"detection_vlm": {"bbox_convention": "xyxy_1000"}},
    )

    detection_vlm = FakeVLM(
        [
            json.dumps(
                {"detections": [{"target_type": "buildings", "bbox": [10, 20, 30, 40]}]}
            )
        ]
    )
    assessment_vlm = FakeVLM(
        [
            json.dumps(
                {
                    "damage_category": "destroyed",
                    "confidence_level": "confirmed",
                    "brief_supporting_logic": "roof collapse; wall breach",
                }
            ),
            "Scene summary grounded in target assessments.",
        ]
    )
    patch_backends(
        monkeypatch,
        detection_vlm=detection_vlm,
        assessment_vlm=assessment_vlm,
        model_names=[],
    )

    pipeline = pipeline_model.BDAPipeline()
    image_path = tmp_path / "tank.jpg"
    Image.new("RGB", (4000, 2000)).save(image_path)

    result = pipeline.analyze(image_path)

    # Happy path should return summary section
    assert result["summary"] == "Scene summary grounded in target assessments."
    # Physical damage section should be formatted IAW JSON schema
    assert result["physical_damage"]["target_0"] == {
        "target_type": "buildings",
        "damage_category": "DESTROYED",
        "confidence_level": "CONFIRMED",
        "brief_supporting_logic": "roof collapse; wall breach",
        "bounding_box": [40, 40, 120, 80],
    }
    assert result["_debug"]["detection"]["bbox_convention"] == "xyxy_1000"
    assert result["_debug"]["detection"]["kept_detections"][0]["pixel_bbox"] == [
        40,
        40,
        120,
        80,
    ]


def test_analyze_no_target_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return the no-target placeholder when detection finds nothing."""
    # Malformed detection output
    detection_vlm = FakeVLM(["Not valid targets"])
    assessment_vlm = FakeVLM(["No relevant targets were assessed in the scene."])
    patch_backends(
        monkeypatch,
        detection_vlm=detection_vlm,
        assessment_vlm=assessment_vlm,
        model_names=[],
    )

    pipeline = pipeline_model.BDAPipeline()
    image_path = tmp_path / "scene.jpg"
    Image.new("RGB", (4000, 2000)).save(image_path)

    result = pipeline.analyze(image_path)

    assert result["summary"] == "No relevant targets were assessed in the scene."
    # No targets should return physical damage section IAW JSON schema
    assert result["physical_damage"]["target_0"] == {
        "target_type": "object_not_found",
        "damage_category": "NOT APPLICABLE",
        "confidence_level": "CONFIRMED",
        "brief_supporting_logic": "No visible targets in image.",
        "bounding_box": [0, 0, 0, 0],
    }
    assert result["_debug"]["detection"]["raw_response"] == "Not valid targets"
    assert "validation_error" in result["_debug"]["detection"]
