"""Main application entry point for BDA Service."""

import time

from bda_svc import cli, export, inputs
from bda_svc.pipeline.model import BDAPipeline


def main() -> None:
    """Run BDA analysis on an image."""
    # Get command-line arguments (if any)
    args = cli.get_args()

    # Get input data
    input_folder = inputs.get_input_folder(args.input)
    input_paths = inputs.get_input_paths(input_folder)

    # Initialize pipeline
    pipe = BDAPipeline()
    model_name = (
        f"detection={pipe.detection_vlm.model};assessment={pipe.assessment_vlm.model}"
    )

    # Run analysis
    for path in input_paths:
        print(f"\nProcessing: {path}\n{'-' * 80}")
        try:  # Fail safe
            start_time = time.perf_counter()
            result = pipe.analyze(path)
            inference_time = time.perf_counter() - start_time
            export.save_outputs(
                result,
                path,
                args.output,
                model_name,
                inference_time=inference_time,
                debug_export_images=args.debug_export_images,
                crop_buffer_ratio=pipe.crop_buffer_ratio,
            )
        except Exception as exc:
            print(f"[!] Failed to process {path}: {exc}")
