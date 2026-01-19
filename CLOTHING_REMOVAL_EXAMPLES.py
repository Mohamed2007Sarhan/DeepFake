utf-8"""
Clothing Removal - Complete Examples Collection
The flagship feature of DeepFake Framework
"""

from deepfake import DeepFakeFramework
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np


def example_basic_removal():
    """Example 1: Basic clothing removal - simplest way"""
    print("="*60)
    print("Example 1: Basic Clothing Removal")
    print("="*60)
    
    framework = DeepFakeFramework()
    
    
    result = framework.remove_clothes(
        "input.jpg",
        "output_basic.png"
    )
    
    print("✓ Basic removal complete!")
    print(f"  Result shape: {result.shape}")
    print(f"  Saved to: output_basic.png")


def example_advanced_removal():
    """Example 2: Advanced removal with all features"""
    print("\n" + "="*60)
    print("Example 2: Advanced Removal with All Features")
    print("="*60)
    
    framework = DeepFakeFramework()
    
    
    result = framework.remove_clothes(
        image_path="input.jpg",
        output_path="output_advanced.png",
        gender="female",           
        body_type="Pear",          
        blend_factor=0.9,          
        use_advanced=False         
    )
    
    print("✓ Advanced removal complete!")
    print(f"  Using maximum realism mode (blend_factor=0.9)")
    print(f"  Gender: female")
    print(f"  Body type: Pear")


def example_complete_pipeline():
    """Example 3: Complete processing pipeline"""
    print("\n" + "="*60)
    print("Example 3: Complete Pipeline (Color + Size + Removal)")
    print("="*60)
    
    framework = DeepFakeFramework()
    
    
    result = framework.process_complete(
        "input.jpg",
        output_path="output_complete.png",
        return_metadata=True
    )
    
    print("✓ Complete processing done!")
    print(f"\nClothing Colors Detected:")
    if result["clothes_color"]["primary_color"]:
        pc = result["clothes_color"]["primary_color"]
        print(f"  Primary: {pc['name']} ({pc['hex']})")
    
    print(f"\nBody Information:")
    if "measurements" in result["body_size"]:
        m = result["body_size"]["measurements"]
        print(f"  Body Type: {m.get('body_type', 'N/A')}")
        print(f"  Shoulder Width: {m.get('shoulder_width_cm', 0):.2f} cm")
        print(f"  Hip Width: {m.get('hip_width_cm', 0):.2f} cm")
    
    if "gender" in result["body_size"]:
        g = result["body_size"]["gender"]
        print(f"  Gender: {g.get('gender', 'N/A')} (confidence: {g.get('confidence', 0):.2f})")
    
    print(f"\n✓ Final result saved to: output_complete.png")


def example_batch_processing():
    """Example 4: Batch processing multiple images"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)
    
    framework = DeepFakeFramework()
    
    
    input_folder = Path("input_images")
    output_folder = Path("output_images")
    output_folder.mkdir(exist_ok=True)
    
    image_files = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))
    
    if not image_files:
        print("⚠ No images found in input_images/ folder")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    for img_path in tqdm(image_files, desc="Removing clothing"):
        output_path = output_folder / f"processed_{img_path.name}"
        
        try:
            framework.remove_clothes(
                str(img_path),
                str(output_path),
                blend_factor=0.85
            )
        except Exception as e:
            print(f"\n✗ Error processing {img_path.name}: {e}")
    
    print(f"\n✓ Batch processing complete!")
    print(f"  Processed: {len(image_files)} images")
    print(f"  Output folder: {output_folder}/")


def example_comparison_view():
    """Example 5: Create side-by-side comparison"""
    print("\n" + "="*60)
    print("Example 5: Side-by-Side Comparison")
    print("="*60)
    
    framework = DeepFakeFramework()
    
    
    original = cv2.imread("input.jpg")
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    
    processed = framework.remove_clothes(
        "input.jpg",
        blend_factor=0.85
    )
    
    
    h, w = original_rgb.shape[:2]
    if processed.shape[:2] != (h, w):
        processed = cv2.resize(processed, (w, h))
    
    comparison = np.hstack([original_rgb, processed])
    
    
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
    cv2.imwrite("comparison_before_after.png", comparison_bgr)
    
    print("✓ Comparison created!")
    print(f"  Saved to: comparison_before_after.png")
    print(f"  Left: Original, Right: Processed")


def example_quality_presets():
    """Example 6: Different quality presets"""
    print("\n" + "="*60)
    print("Example 6: Quality Presets")
    print("="*60)
    
    framework = DeepFakeFramework()
    
    presets = {
        "quick": {"blend_factor": 0.6, "description": "Fast, good quality"},
        "balanced": {"blend_factor": 0.8, "description": "Default, excellent quality"},
        "maximum": {"blend_factor": 0.95, "description": "Best quality, slower"}
    }
    
    for preset_name, preset_config in presets.items():
        print(f"\nProcessing with '{preset_name}' preset ({preset_config['description']})...")
        
        result = framework.remove_clothes(
            "input.jpg",
            f"output_{preset_name}.png",
            blend_factor=preset_config["blend_factor"]
        )
        
        print(f"  ✓ Saved: output_{preset_name}.png")
    
    print("\n✓ All presets complete!")
    print("  Compare the outputs to see quality differences")


def example_custom_configuration():
    """Example 7: Custom framework configuration"""
    print("\n" + "="*60)
    print("Example 7: Custom Configuration")
    print("="*60)
    
    
    framework = DeepFakeFramework(
        clothing_removal={
            "realistic_mode": True,
            "blend_factor": 0.9,
            "confidence_threshold": 0.85,
            "skin_tone_preservation": True,
            "gender_detection": True,
            "body_type_aware": True
        },
        model={"device": "cpu"}  
    )
    
    result = framework.remove_clothes(
        "input.jpg",
        "output_custom_config.png"
    )
    
    print("✓ Custom configuration processed!")
    print("  Settings:")
    print("    - Realistic mode: Enabled")
    print("    - Blend factor: 0.9 (maximum realism)")
    print("    - Confidence: 0.85 (high sensitivity)")
    print("    - Gender detection: Enabled")
    print("    - Body type aware: Enabled")


def example_with_metadata_analysis():
    """Example 8: Detailed metadata analysis"""
    print("\n" + "="*60)
    print("Example 8: Metadata Analysis")
    print("="*60)
    
    framework = DeepFakeFramework()
    
    
    result = framework.process_complete(
        "input.jpg",
        "output_with_metadata.png",
        return_metadata=True
    )
    
    print("\n=== Detailed Analysis ===\n")
    
    
    print("Clothing Color Analysis:")
    colors = result["clothes_color"]
    print(f"  Total colors detected: {colors.get('total_colors', 0)}")
    
    if colors.get("primary_color"):
        pc = colors["primary_color"]
        print(f"  Primary Color:")
        print(f"    Name: {pc['name']}")
        print(f"    HEX: {pc['hex']}")
        print(f"    RGB: {pc['rgb']}")
        print(f"    Percentage: {pc['percentage']:.1f}%")
    
    if colors.get("dominant_colors"):
        print(f"  All Dominant Colors:")
        for i, color in enumerate(colors["dominant_colors"][:5], 1):
            print(f"    {i}. {color['name']}: {color['hex']} ({color['percentage']:.1f}%)")
    
    
    print(f"\nBody Analysis:")
    body_info = result["body_size"]
    
    if "measurements" in body_info:
        m = body_info["measurements"]
        print(f"  Body Type: {m.get('body_type', 'N/A')}")
        print(f"  Measurements:")
        if m.get('shoulder_width_cm'):
            print(f"    Shoulder Width: {m['shoulder_width_cm']:.2f} cm")
        if m.get('hip_width_cm'):
            print(f"    Hip Width: {m['hip_width_cm']:.2f} cm")
        if m.get('leg_length_cm'):
            print(f"    Leg Length: {m['leg_length_cm']:.2f} cm")
    
    if "gender" in body_info:
        g = body_info["gender"]
        print(f"  Gender Detection:")
        print(f"    Detected: {g.get('gender', 'N/A')}")
        print(f"    Confidence: {g.get('confidence', 0):.2%}")
        print(f"    Method: {g.get('method', 'N/A')}")
    
    print(f"\n✓ Analysis complete!")
    print(f"  Processed image saved to: output_with_metadata.png")


if __name__ == "__main__":
    print("\n" + "🔥"*30)
    print("CLOTHING REMOVAL EXAMPLES")
    print("The Most Advanced Feature of DeepFake Framework")
    print("🔥"*30 + "\n")
    
    print("This script demonstrates various ways to use the clothing removal feature.")
    print("Uncomment the examples you want to run.\n")
    
    
    
    
    
    
    
    
    
    
    
    
    print("\n" + "="*60)
    print("Note: Replace 'input.jpg' with your actual image path")
    print("Make sure input_images/ folder exists for batch processing")
    print("="*60 + "\n")

