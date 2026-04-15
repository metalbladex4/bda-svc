"""Pipeline 'config.yaml' and 'doctrine.yaml' tests."""

from bda_svc.pipeline.utilities import CONFIG_PATH, DOCTRINE_PATH, load_yaml

# ----------------------------------------------------------------------
# Test Setup: config.yaml
# ----------------------------------------------------------------------


EXPECTED_CONFIG_KEYS = {"detection_vlm", "assessment_vlm", "prompts"}
EXPECTED_DETECTION_VLM_KEYS = {
    "model",
    "bbox_convention",
    "temperature",
    "max_image_size",
    "crop_buffer_ratio",
    "refinement_enabled",
    "refinement_roi_buffer_ratio",
}
EXPECTED_ASSESSMENT_VLM_KEYS = {"model", "temperature", "max_image_size"}
EXPECTED_PROMPT_KEYS = {"system", "detect_objects", "assess_damage", "summarize_scene"}


def test_config_keys() -> None:
    """Config should contain expected top-level sections."""
    config = load_yaml(CONFIG_PATH)
    assert set(config.keys()) == EXPECTED_CONFIG_KEYS


def test_config_vlm_keys() -> None:
    """Config detection and assessment VLM keys should be exact."""
    config = load_yaml(CONFIG_PATH)
    assert set(config["detection_vlm"].keys()) == EXPECTED_DETECTION_VLM_KEYS
    assert set(config["assessment_vlm"].keys()) == EXPECTED_ASSESSMENT_VLM_KEYS


def test_config_detection_vlm_types() -> None:
    """Config detection VLM values should have the expected types."""
    config = load_yaml(CONFIG_PATH)
    detection = config["detection_vlm"]
    assert isinstance(detection["model"], str)
    assert isinstance(detection["bbox_convention"], str)
    assert isinstance(detection["temperature"], float)
    assert isinstance(detection["max_image_size"], int)
    assert isinstance(detection["crop_buffer_ratio"], float)
    assert isinstance(detection["refinement_enabled"], bool)
    assert isinstance(detection["refinement_roi_buffer_ratio"], float)


def test_config_assessment_vlm_types() -> None:
    """Config assessment VLM values should have the expected types."""
    config = load_yaml(CONFIG_PATH)
    assessment = config["assessment_vlm"]
    assert isinstance(assessment["model"], str)
    assert isinstance(assessment["temperature"], float)
    assert isinstance(assessment["max_image_size"], int)


def test_config_prompt_keys() -> None:
    """Config prompt keys should be exact."""
    config = load_yaml(CONFIG_PATH)
    prompts = config["prompts"]
    assert set(prompts.keys()) == EXPECTED_PROMPT_KEYS


def test_config_prompt_placeholders() -> None:
    """Config prompts should contain required placeholders."""
    config = load_yaml(CONFIG_PATH)
    prompts = config["prompts"]
    assert "{categories}" in prompts["detect_objects"]
    assert "{detection_guidance}" in prompts["detect_objects"]
    assert "{bbox_format}" in prompts["detect_objects"]
    assert "{bbox_scale}" in prompts["detect_objects"]
    assert "{target_type}" in prompts["assess_damage"]
    assert "{doctrine}" in prompts["assess_damage"]
    assert "{target_assessments}" in prompts["summarize_scene"]


# ----------------------------------------------------------------------
# Test Setup: doctrine.yaml
# ----------------------------------------------------------------------


EXPECTED_DOCTRINE_KEYS = {"buildings", "military_equipment"}
EXPECTED_DOCTRINE_ENTRY_KEYS = {
    "detection_guidance",
    "physical_damage_definitions",
    "physical_damage_considerations",
}


def test_doctrine_keys() -> None:
    """Doctrine should contain expected top-level sections."""
    doctrine = load_yaml(DOCTRINE_PATH)
    assert set(doctrine.keys()) == EXPECTED_DOCTRINE_KEYS


def test_doctrine_entry_keys() -> None:
    """Each doctrine category should contain the expected sections."""
    doctrine = load_yaml(DOCTRINE_PATH)
    for entry in doctrine.values():
        assert set(entry.keys()) == EXPECTED_DOCTRINE_ENTRY_KEYS
