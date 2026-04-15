"""Vision-Language Model BDA pipeline."""

import json
import os
from pathlib import Path

from json_repair import repair_json
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from bda_svc.pipeline.interfaces import Detection, OllamaVLM
from bda_svc.pipeline.utilities import (
    CONFIG_PATH,
    DOCTRINE_PATH,
    bbox_to_pixels,
    crop_with_buffer,
    draw_box_overlay,
    expand_box,
    format_detection_doctrine,
    format_pda_doctrine,
    load_yaml,
    resize_for_vlm,
)


class DetectionItem(BaseModel):
    """Structured detection item returned by detection model."""

    target_type: str
    bbox: list[float] = Field(min_length=4, max_length=4)


class DetectionResponse(BaseModel):
    """Structured detection response returned by detection model."""

    detections: list[DetectionItem]


class AssessmentResponse(BaseModel):
    """Structured assessment response returned by assessment model."""

    damage_category: str
    confidence_level: str
    brief_supporting_logic: str


class BDAPipeline:
    """BDA pipeline combining detection and damage assessment."""

    def __init__(self) -> None:
        """Initialize configuration, doctrine, prompts, and backends."""
        self.last_debug_info: dict[str, object] = {}

        # Load yamls
        self.config = load_yaml(CONFIG_PATH)
        self.doctrine = load_yaml(DOCTRINE_PATH)
        self.categories = list(self.doctrine.keys())

        # Load prompts
        prompts = self.config["prompts"]
        self.system_prompt = prompts["system"]
        self.detect_objects_prompt_template = prompts["detect_objects"]
        self.assess_damage_prompt_template = prompts["assess_damage"]
        self.summarize_scene_prompt_template = prompts["summarize_scene"]

        # Load detection backend
        detection_cfg = self.config["detection_vlm"]
        detection_model = os.environ.get("BDA_DETECTION_MODEL", detection_cfg["model"])
        self.detection_bbox_convention = detection_cfg["bbox_convention"]
        self.detection_temperature = float(detection_cfg["temperature"])
        self.detection_max_image_size = int(detection_cfg["max_image_size"])
        self.crop_buffer_ratio = float(detection_cfg["crop_buffer_ratio"])
        self.detection_refinement_enabled = bool(
            detection_cfg.get("refinement_enabled", False)
        )
        self.detection_refinement_roi_buffer_ratio = float(
            detection_cfg.get("refinement_roi_buffer_ratio", self.crop_buffer_ratio)
        )
        self.detection_vlm = OllamaVLM(model=detection_model)

        # Load assessment backend
        assessment_cfg = self.config["assessment_vlm"]
        assessment_model = os.environ.get(
            "BDA_ASSESSMENT_MODEL", assessment_cfg["model"]
        )
        self.assessment_temperature = float(assessment_cfg["temperature"])
        self.assessment_max_image_size = int(assessment_cfg["max_image_size"])
        self.assessment_vlm = OllamaVLM(model=assessment_model)

    def detect_objects(self, image: Image.Image) -> list[Detection]:
        """Produce detections for configured doctrinal categories.

        Args:
            image: PIL image to analyze.

        Returns:
            Detection records with crops attached.
        """
        # Get detections, bounding boxes are stored in pixel coordinates
        detections, debug_record = self._vlm_detections(image)

        if self.detection_refinement_enabled and detections:
            detections, refinement_debug = self._refine_detections(image, detections)
            debug_record["refinement"] = refinement_debug

        debug_record["final_detections"] = [
            {"target_type": det.label, "pixel_bbox": list(det.bbox)}
            for det in detections
        ]
        self.last_debug_info["detection"] = debug_record

        # Attach padded image crops to detections
        detections_with_crops = [
            Detection(
                label=det.label,
                bbox=det.bbox,
                crop=crop_with_buffer(image, det.bbox, self.crop_buffer_ratio),
            )
            for det in detections
        ]

        # Sort by label then left-to-right
        detections_with_crops.sort(key=lambda d: (d.label.lower(), d.bbox[0]))
        return detections_with_crops

    def _vlm_detections(
        self,
        image: Image.Image,
        *,
        categories: list[str] | None = None,
    ) -> tuple[list[Detection], dict[str, object]]:
        """Use the detection VLM to produce object detections.

        Args:
            image: PIL image to analyze.
            categories: Optional doctrinal categories to constrain detection.

        Returns:
            Parsed detections in raw pixel coordinates plus a debug record.
        """
        categories = categories or self.categories
        prompt = self.detect_objects_prompt_template

        # Format prompt with doctrinal categories
        category_text = ", ".join(categories)
        prompt = prompt.replace("{categories}", category_text)
        prompt = prompt.replace(
            "{detection_guidance}", format_detection_doctrine(categories)
        )

        # Format prompt with bbox format
        if self.detection_bbox_convention.startswith("xyxy"):
            bbox_format = "[xmin, ymin, xmax, ymax]"
        elif self.detection_bbox_convention.startswith("yxyx"):
            bbox_format = "[ymin, xmin, ymax, xmax]"
        else:
            raise ValueError(
                "Unsupported bounding box convention specified in config."
                " Supported formats start with 'xyxy' or 'yxyx'."
            )
        prompt = prompt.replace("{bbox_format}", bbox_format)

        # Format prompt with bbox scale
        if self.detection_bbox_convention.endswith("_1"):
            bbox_scale = "normalized coordinates from 0.0 to 1.0"
        elif self.detection_bbox_convention.endswith("_1000"):
            bbox_scale = "normalized coordinates from 0 to 1000"
        elif self.detection_bbox_convention.endswith("_pixels"):
            bbox_scale = "raw pixel coordinates relative to the image"
        else:
            raise ValueError(
                "Unsupported bounding box convention specified in config."
                " Supported formats end with '_1' or '_1000' or '_pixels'."
            )
        prompt = prompt.replace("{bbox_scale}", bbox_scale)

        # Get VLM response
        vlm_image = resize_for_vlm(image, self.detection_max_image_size)
        response = self.detection_vlm.generate(
            image=vlm_image,
            prompt=prompt,
            system_prompt=self.system_prompt,
            format_schema=DetectionResponse.model_json_schema(),
            temperature=self.detection_temperature,
        )

        debug_record: dict[str, object] = {
            "bbox_convention": self.detection_bbox_convention,
            "original_image_size": [image.width, image.height],
            "model_image_size": [vlm_image.width, vlm_image.height],
            "raw_response": response,
        }

        # Fail safely
        try:
            payload = repair_json(response)
            debug_record["repaired_response"] = payload
            payload = DetectionResponse.model_validate_json(payload)
        except ValidationError as exc:
            debug_record["validation_error"] = str(exc)
            return [], debug_record

        debug_record["parsed_detections"] = payload.model_dump()["detections"]

        # Return list of detections
        detections = []
        kept_detections = []
        rejected_detections = []
        for index, item in enumerate(payload.detections):
            # Validate target_type is doctrinal
            target_type = item.target_type.strip().lower()
            if target_type not in categories:
                rejected_detections.append(
                    {
                        "index": index,
                        "reason": "invalid_target_type",
                        "target_type": item.target_type,
                        "raw_bbox": item.bbox,
                    }
                )
                continue

            # Validate bounding box is valid
            pixel_box = bbox_to_pixels(
                image,
                vlm_image,
                item.bbox,
                bbox_convention=self.detection_bbox_convention,
            )
            if pixel_box is None:
                rejected_detections.append(
                    {
                        "index": index,
                        "reason": "invalid_bbox",
                        "target_type": target_type,
                        "raw_bbox": item.bbox,
                    }
                )
                continue

            detections.append(Detection(label=target_type, bbox=pixel_box))
            kept_detections.append(
                {
                    "index": index,
                    "target_type": target_type,
                    "raw_bbox": item.bbox,
                    "pixel_bbox": list(pixel_box),
                }
            )

        debug_record["kept_detections"] = kept_detections
        if rejected_detections:
            debug_record["rejected_detections"] = rejected_detections
        return detections, debug_record

    def _refine_detections(
        self,
        image: Image.Image,
        detections: list[Detection],
    ) -> tuple[list[Detection], dict[str, object]]:
        """Re-run detection inside expanded ROIs to refine coarse boxes.

        Args:
            image: Full-scene source image.
            detections: First-pass detections in scene pixel coordinates.

        Returns:
            Refined detections plus per-attempt debug data.
        """
        attempts: list[dict[str, object]] = []
        refined_detections: list[Detection] = []

        for index, detection in enumerate(detections):
            roi_box = expand_box(
                image,
                detection.bbox,
                self.detection_refinement_roi_buffer_ratio,
            )
            roi_image = image.crop(roi_box)
            roi_detections, roi_debug = self._vlm_detections(
                roi_image,
                categories=[detection.label],
            )

            attempt: dict[str, object] = {
                "index": index,
                "target_type": detection.label,
                "original_bbox": list(detection.bbox),
                "roi_bbox": list(roi_box),
                "roi_size": [roi_image.width, roi_image.height],
                "detection_debug": roi_debug,
            }

            if not roi_detections:
                attempt["decision"] = "kept_original_no_refined_detection"
                refined_detections.append(detection)
                attempts.append(attempt)
                continue

            translated_candidates = []
            for candidate in roi_detections:
                translated_box = self._translate_box(candidate.bbox, roi_box)
                translated_candidates.append(
                    {
                        "target_type": candidate.label,
                        "pixel_bbox": list(translated_box),
                        "iou_vs_original": self._bbox_iou(
                            translated_box,
                            detection.bbox,
                        ),
                        "area": self._bbox_area(translated_box),
                    }
                )

            attempt["translated_candidates"] = translated_candidates
            best_candidate = max(
                translated_candidates,
                key=lambda item: (item["iou_vs_original"], item["area"]),
            )

            if best_candidate["iou_vs_original"] <= 0.0:
                attempt["decision"] = "kept_original_no_overlap"
                refined_detections.append(detection)
            else:
                attempt["decision"] = "selected_refined_bbox"
                attempt["selected_bbox"] = best_candidate["pixel_bbox"]
                refined_detections.append(
                    Detection(
                        label=detection.label,
                        bbox=tuple(best_candidate["pixel_bbox"]),
                    )
                )

            attempts.append(attempt)

        return refined_detections, {
            "enabled": True,
            "roi_buffer_ratio": self.detection_refinement_roi_buffer_ratio,
            "attempts": attempts,
        }

    @staticmethod
    def _translate_box(
        box: tuple[int, int, int, int],
        roi_box: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Translate an ROI-local box back into scene coordinates."""
        left, top, _, _ = roi_box
        xmin, ymin, xmax, ymax = box
        return xmin + left, ymin + top, xmax + left, ymax + top

    @staticmethod
    def _bbox_area(box: tuple[int, int, int, int]) -> int:
        """Return bounding-box area."""
        xmin, ymin, xmax, ymax = box
        return max(0, xmax - xmin) * max(0, ymax - ymin)

    @classmethod
    def _bbox_iou(
        cls,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Return intersection-over-union between two boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_left = max(ax1, bx1)
        inter_top = max(ay1, by1)
        inter_right = min(ax2, bx2)
        inter_bottom = min(ay2, by2)

        inter_w = max(0, inter_right - inter_left)
        inter_h = max(0, inter_bottom - inter_top)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        union_area = cls._bbox_area(box_a) + cls._bbox_area(box_b) - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def assess_detection(
        self,
        detection: Detection,
        scene_image: Image.Image | None = None,
    ) -> dict | None:
        """Assess damage for a single detected object crop.

        Args:
            detection: Detection with populated `bbox` and `crop`.
            scene_image: Optional full-scene for additional context.

        Returns:
            Final target assessment record.
        """
        # Format prompt
        doctrine = format_pda_doctrine(detection.label)
        prompt = self.assess_damage_prompt_template
        prompt = prompt.replace("{target_type}", detection.label)
        prompt = prompt.replace("{doctrine}", doctrine)

        # Format image inputs
        if scene_image is None:
            image_input = resize_for_vlm(detection.crop, self.assessment_max_image_size)
        else:
            scene_with_overlay = draw_box_overlay(scene_image, detection.bbox)
            image_input = [
                resize_for_vlm(scene_with_overlay, self.assessment_max_image_size),
                resize_for_vlm(detection.crop, self.assessment_max_image_size),
            ]

        # Get VLM response
        response = self.assessment_vlm.generate(
            image=image_input,
            prompt=prompt,
            system_prompt=self.system_prompt,
            format_schema=AssessmentResponse.model_json_schema(),
            temperature=self.assessment_temperature,
        )

        # Fail safely
        try:
            payload = repair_json(response)
            payload = AssessmentResponse.model_validate_json(payload)
        except ValidationError:
            return None

        # Return structured output
        return {
            "target_type": detection.label,
            "damage_category": payload.damage_category.upper(),
            "confidence_level": payload.confidence_level.upper(),
            "brief_supporting_logic": payload.brief_supporting_logic,
            "bounding_box": list(detection.bbox),
        }

    def summarize_scene(self, scene_image: Image.Image, targets: list[dict]) -> str:
        """Summarize the scene using the image and assessed targets.

        Args:
            scene_image: Full-scene image.
            targets: Finalized per-target assessment payloads.

        Returns:
            Concise scene summary text.
        """
        target_assessments = json.dumps(targets, indent=2)
        prompt = self.summarize_scene_prompt_template.replace(
            "{target_assessments}", target_assessments
        )
        summary_image = resize_for_vlm(scene_image, self.assessment_max_image_size)
        response = self.assessment_vlm.generate(
            image=summary_image,
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=self.assessment_temperature,
        )
        return response.strip()

    def consolidate_results(
        self, targets: list[dict] | None, scene_summary: str
    ) -> dict:
        """Consolidate per-target results into the final output shape.

        Args:
            targets: Per-target assessment payloads.
            scene_summary: Scene-level summary.

        Returns:
            Final image-level result dictionary.
        """
        template = {"summary": scene_summary, "physical_damage": {}}

        if not targets:
            targets = [
                {
                    "target_type": "object_not_found",
                    "damage_category": "NOT APPLICABLE",
                    "confidence_level": "CONFIRMED",
                    "brief_supporting_logic": "No visible targets in image.",
                    "bounding_box": [0, 0, 0, 0],
                }
            ]

        for i, target in enumerate(targets):
            template["physical_damage"][f"target_{i}"] = target

        return template

    def analyze(self, image_path: str | Path) -> str:
        """Run the full BDA pipeline and return the final result payload.

        Args:
            image_path: Path to the input image.

        Returns:
            Final image-level BDA payload.
        """
        self.last_debug_info = {}
        with Image.open(Path(image_path)).convert("RGB") as image:
            detections = self.detect_objects(image)
            targets = [self.assess_detection(d, scene_image=image) for d in detections]
            targets = [t for t in targets if t is not None]
            scene_summary = self.summarize_scene(image, targets)
            result = self.consolidate_results(targets, scene_summary)
            if self.last_debug_info:
                result["_debug"] = self.last_debug_info
            return result
