# run_vision.py

from text_generation import TextGenerationAI
from vision_ai import VisionAI
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run AI Vision with Text Generation capabilities')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--model', type=str, default="keras_rnn_model", help='Model name to use')
    parser.add_argument('--mode', type=str, default="normal",
                        choices=["normal", "grayscale", "edges", "blur", "face_detection", "text_detection"],
                        help='Initial processing mode')
    parser.add_argument('--no-window', action='store_true', help='Disable visual window')
    parser.add_argument('--no-text', action='store_true', help='Disable text generation')
    parser.add_argument('--capture-only', action='store_true', help='Capture a single frame and exit')
    parser.add_argument('--output', type=str, help='Output filename for captured frame')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Check if model directory exists
    model_dir = f"models/{args.model}"
    model_path = f"{model_dir}/model.keras"

    # Initialize text generation AI
    text_ai = None
    if not args.no_text:
        print(f"Initializing Text Generation AI with model: {args.model}")
        text_ai = TextGenerationAI(model_name=args.model)

        if os.path.exists(model_path):
            if text_ai.load():
                print("Text generation model loaded successfully")
            else:
                print("Failed to load text generation model")
        else:
            print(f"Warning: Model not found at {model_path}")
            print("Text generation will be limited")

    # Initialize vision AI
    vision_ai = VisionAI(text_ai=text_ai, camera_index=args.camera)
    vision_ai.processing_mode = args.mode

    # Single capture mode
    if args.capture_only:
        output_file = args.output if args.output else f"ai_vision_capture.jpg"
        print(f"Capturing a single frame with mode: {args.mode}")
        frame = vision_ai.capture_and_save(output_file)
        if frame is not None:
            vision_ai.visualize_frame(frame)
            if text_ai and text_ai.model is not None:
                description = vision_ai.describe_scene(frame)
                print("\nAI Description:")
                print(description)
    else:
        # Interactive mode
        print("Starting vision system...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'm' - Change processing mode")
        print("  's' - Save current frame")
        print("  'd' - Generate description of current scene")

        vision_ai.run_vision_loop(
            show_window=not args.no_window,
            enable_text_generation=not args.no_text
        )