"""
DeepFake Framework - Main Entry Point
Example usage and CLI interface
"""

import argparse
from pathlib import Path
from deepfake import DeepFakeFramework


def main():
    parser = argparse.ArgumentParser(
        description="DeepFake Framework - Advanced Image & Voice Manipulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py interactive
  python main.py -i
  
  # Image processing
  python main.py image --input photo.jpg --type complete --output result.png
  
  # Face swapping
  python main.py swap --source face1.jpg --target face2.jpg --output swapped.png
  
  # Voice cloning
  python main.py voice --sample voice.wav --text "Hello" --output output.wav
  
For more help on a specific command:
  python main.py <command> --help
        """
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Launch interactive CLI menu (recommended for beginners)"
    )


    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Processing mode", metavar="<command>")
    
    # Original modes
    parser_img = subparsers.add_parser("image", help="Image processing modes")
    parser_img.add_argument("--input", "-i", type=str, required=True, help="Input image path")
    parser_img.add_argument("--output", "-o", type=str, help="Output image path")
    parser_img.add_argument("--type", "-t", type=str, 
                           choices=["color", "size", "remove", "complete"],
                           default="complete", help="Processing type")
    parser_img.add_argument("--config", "-c", type=str, help="Path to config.yaml")
    parser_img.add_argument("--device", "-d", type=str, choices=["cpu", "cuda"], default="cpu",
                           help="Device to use (cpu/cuda)")
    
    # Face swap modes
    parser_swap = subparsers.add_parser("swap", help="Face swapping")
    parser_swap.add_argument("--source", "-s", type=str, required=True,
                            help="Source image path (face to swap from)")
    parser_swap.add_argument("--target", "-t", type=str, required=True,
                            help="Target image/video path (face to swap to)")
    parser_swap.add_argument("--output", "-o", type=str, required=True,
                            help="Output path")
    parser_swap.add_argument("--blend", "-b", type=str, 
                            choices=["seamless", "linear", "feathered"],
                            default="seamless", help="Blending mode")
    parser_swap.add_argument("--source-face", type=int, default=0,
                            help="Source face index (if multiple faces)")
    parser_swap.add_argument("--target-face", type=int, default=0,
                            help="Target face index (if multiple faces)")
    
    # Voice cloning modes
    parser_voice = subparsers.add_parser("voice", help="Voice cloning and synthesis")
    parser_voice.add_argument("--sample", "-s", type=str, required=True,
                             help="Voice sample audio file path")
    parser_voice.add_argument("--text", "-t", type=str, required=True,
                             help="Text to synthesize")
    parser_voice.add_argument("--output", "-o", type=str, required=True,
                             help="Output audio file path")
    parser_voice.add_argument("--method", "-m", type=str,
                             choices=["auto", "coqui", "pyttsx3", "gtts"],
                             default="auto", help="TTS method")
    
    # For backwards compatibility
    parser.add_argument("--input", "-i", type=str, help="Input image path (legacy)")
    parser.add_argument("--output", "-o", type=str, help="Output path (legacy)")
    parser.add_argument("--mode", "-m", type=str, help="Processing mode (legacy)")
    parser.add_argument("--config", "-c", type=str, help="Path to config.yaml")
    parser.add_argument("--device", "-d", type=str, choices=["cpu", "cuda"], default="cpu",
                       help="Device to use (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Handle interactive mode
    if args.interactive or (not args.mode and not hasattr(args, 'type')):
        try:
            from deepfake.cli import main_interactive
            main_interactive()
            return
        except ImportError:
            print("Interactive mode requires rich and colorama.")
            print("Install with: pip install rich colorama")
            print("\nFalling back to command-line mode...")
            print("Use --help to see available commands.")
            return
    
    # Handle legacy mode
    if args.mode and not hasattr(args, 'type'):
        # Legacy single-argument mode
        legacy_args = type('obj', (object,), {
            'input': args.input,
            'output': args.output,
            'type': args.mode,
            'config': args.config,
            'device': args.device
        })()
        args = legacy_args
        mode = "image"
    else:
        mode = args.mode if hasattr(args, 'mode') and args.mode else "image"
    
    # Initialize framework
    print("Initializing DeepFake Framework...")
    config_path = getattr(args, 'config', None)
    device = getattr(args, 'device', 'cpu')
    
    framework = DeepFakeFramework(
        config_path=config_path,
        model={"device": device}
    )
    
    # Process based on mode
    if mode == "swap":
        # Face swapping mode
        print("\n=== Face Swapping ===")
        source = args.source
        target = args.target
        output = args.output
        
        # Check if target is video
        is_video = target.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))
        
        if is_video:
            print(f"Swapping face from image to video...")
            print(f"Source: {source}")
            print(f"Target Video: {target}")
            print(f"Output: {output}")
            
            metadata = framework.swap_face_to_video(
                source_image_path=source,
                target_video_path=target,
                output_video_path=output,
                source_face_idx=getattr(args, 'source_face', 0),
                blend_mode=getattr(args, 'blend', 'seamless')
            )
            
            print(f"\n✓ Video processing complete!")
            print(f"  Frames processed: {metadata['total_frames']}")
            print(f"  Faces detected: {metadata['faces_detected']}")
            print(f"  FPS: {metadata['fps']}")
        else:
            print(f"Swapping faces between images...")
            print(f"Source: {source}")
            print(f"Target: {target}")
            print(f"Output: {output}")
            
            metadata = framework.swap_faces_image(
                source_image_path=source,
                target_image_path=target,
                output_path=output,
                source_face_idx=getattr(args, 'source_face', 0),
                target_face_idx=getattr(args, 'target_face', 0),
                blend_mode=getattr(args, 'blend', 'seamless'),
                blend_factor=0.8
            )
            
            print(f"\n✓ Face swap complete!")
            print(f"  Source faces detected: {metadata['source_faces_detected']}")
            print(f"  Target faces detected: {metadata['target_faces_detected']}")
            print(f"  Blend mode: {metadata['blend_mode']}")
        
        print(f"\nSaved to: {output}")
    
    elif mode == "voice":
        # Voice cloning mode
        print("\n=== Voice Cloning ===")
        sample = args.sample
        text = args.text
        output = args.output
        
        print(f"Voice Sample: {sample}")
        print(f"Text: {text[:100]}...")
        print(f"Output: {output}")
        
        # Initialize voice cloner with specified method
        from deepfake.voice_cloner import VoiceCloner
        voice_cloner = VoiceCloner(method=getattr(args, 'method', 'auto'))
        
        # Clone voice
        result = voice_cloner.clone_voice_from_sample(
            voice_sample_path=sample,
            text=text,
            output_path=output
        )
        
        print(f"\n✓ Voice cloning complete!")
        print(f"  Method: {result['synthesis']['method']}")
        print(f"  Duration: {result['synthesis'].get('duration', 0):.2f}s")
        print(f"  Saved to: {output}")
        
    elif mode == "image":
        # Original image processing modes
        input_path = getattr(args, 'input', None)
        if not input_path:
            print("Error: --input is required for image processing mode")
            return
        
        processing_type = getattr(args, 'type', 'complete')
        
        if processing_type == "color":
            print("\n=== Clothes Color Detection ===")
            result = framework.get_clothes_color(input_path)
        print(f"Primary Color: {result['primary_color']['name']} ({result['primary_color']['hex']})")
        print(f"RGB: {result['primary_color']['rgb']}")
        print(f"\nAll Dominant Colors:")
        for color in result['dominant_colors']:
            print(f"  - {color['name']}: {color['hex']} ({color['percentage']}%)")
    
    elif args.mode == "size":
        print("\n=== Body Size Estimation ===")
        result = framework.get_body_size(args.input)
        if "error" not in result:
            measurements = result.get("measurements", {})
            print(f"Body Type: {measurements.get('body_type', 'Unknown')}")
            if "shoulder_width_cm" in measurements:
                print(f"Shoulder Width: {measurements['shoulder_width_cm']:.2f} cm")
            if "hip_width_cm" in measurements:
                print(f"Hip Width: {measurements['hip_width_cm']:.2f} cm")
            if "leg_length_cm" in measurements:
                print(f"Leg Length: {measurements['leg_length_cm']:.2f} cm")
            else:
                print(f"Error: {result['error']}")
        
        elif processing_type == "remove":
            print("\n=== Clothing Removal ===")
            output = getattr(args, 'output', None) or "output_removed.png"
            result = framework.remove_clothes(input_path, output_path=output)
            print(f"Processing complete! Saved to: {output}")
        
        elif processing_type == "complete":
            print("\n=== Complete Processing Pipeline ===")
            output = getattr(args, 'output', None) or "output_complete.png"
            result = framework.process_complete(input_path, output_path=output, return_metadata=True)
        
        print("\n=== Results ===")
        print(f"\nClothes Color:")
        if result["clothes_color"]["primary_color"]:
            pc = result["clothes_color"]["primary_color"]
            print(f"  Primary: {pc['name']} ({pc['hex']}) - {pc['percentage']}%")
        
        print(f"\nBody Size:")
        if "measurements" in result["body_size"]:
            m = result["body_size"]["measurements"]
            print(f"  Body Type: {m.get('body_type', 'Unknown')}")
            if "shoulder_width_cm" in m:
                print(f"  Shoulder Width: {m['shoulder_width_cm']:.2f} cm")
            if "hip_width_cm" in m:
                print(f"  Hip Width: {m['hip_width_cm']:.2f} cm")
            
            print(f"\nProcessed image saved to: {output}")
    
    if mode not in ["swap", "voice"]:
        print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()

