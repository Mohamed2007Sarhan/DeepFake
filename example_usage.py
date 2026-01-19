utf-8"""
Example Usage Scripts for DeepFake Framework
Demonstrates various use cases and capabilities
"""

from deepfake import DeepFakeFramework
import numpy as np


def example_basic_usage():
    """Basic usage example"""
    print("=" * 50)
    print("Basic Usage Example")
    print("=" * 50)
    
    
    framework = DeepFakeFramework()
    
    
    color_info = framework.get_clothes_color("input_image.jpg")
    print(f"\nClothes Color: {color_info['primary_color']['name']}")
    print(f"Hex Code: {color_info['primary_color']['hex']}")
    
    
    body_info = framework.get_body_size("input_image.jpg")
    if "measurements" in body_info:
        print(f"\nBody Type: {body_info['measurements']['body_type']}")
    
    
    result = framework.remove_clothes("input_image.jpg", output_path="output.png")
    print(f"\n✓ Processing complete! Result shape: {result.shape}")


def example_advanced_usage():
    """Advanced usage with custom configuration"""
    print("=" * 50)
    print("Advanced Usage Example")
    print("=" * 50)
    
    
    framework = DeepFakeFramework(
        color_detection={"algorithm": "kmeans", "num_colors": 10},
        body_size={"reference_height": 175.0, "use_mediapipe": True},
        clothing_removal={"blend_factor": 0.9, "confidence_threshold": 0.8},
        model={"device": "cpu"}  
    )
    
    
    results = framework.process_complete(
        "input_image.jpg",
        output_path="complete_output.png",
        return_metadata=True
    )
    
    
    print(f"\nPrimary Color: {results['clothes_color']['primary_color']['name']}")
    print(f"Body Type: {results['body_size']['measurements'].get('body_type', 'Unknown')}")
    print(f"Output saved to: complete_output.png")


def example_batch_processing():
    """Batch processing example"""
    print("=" * 50)
    print("Batch Processing Example")
    print("=" * 50)
    
    framework = DeepFakeFramework()
    
    
    image_paths = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg"
    ]
    
    
    results = framework.batch_process(
        image_paths,
        output_dir="outputs/",
        return_metadata=False
    )
    
    print(f"\n✓ Processed {len(results)} images")
    for i, result in enumerate(results):
        print(f"  Image {i+1}: {result['output_path']}")


def example_individual_components():
    """Using individual components"""
    print("=" * 50)
    print("Individual Components Example")
    print("=" * 50)
    
    from deepfake import ClothesColorDetector, BodySizeEstimator, ClothingRemover
    
    
    color_detector = ClothesColorDetector(algorithm="kmeans", num_colors=5)
    colors = color_detector.get_clothes_color("input_image.jpg")
    print(f"\nDetected Colors: {len(colors['dominant_colors'])}")
    
    
    body_estimator = BodySizeEstimator(use_mediapipe=True, reference_height=170.0)
    body_info = body_estimator.estimate_body_size("input_image.jpg")
    if "measurements" in body_info:
        m = body_info["measurements"]
        print(f"Shoulder Width: {m.get('shoulder_width_cm', 'N/A')} cm")
        print(f"Hip Width: {m.get('hip_width_cm', 'N/A')} cm")
    
    
    remover = ClothingRemover(device="cpu", confidence_threshold=0.7)
    result = remover.remove_clothing("input_image.jpg", output_path="removed.png")
    print(f"\n✓ Clothing removal complete! Shape: {result.shape}")


def example_color_analysis():
    """Detailed color analysis example"""
    print("=" * 50)
    print("Color Analysis Example")
    print("=" * 50)
    
    from deepfake import ClothesColorDetector
    import cv2
    
    detector = ClothesColorDetector(algorithm="kmeans", num_colors=10)
    
    
    image = cv2.imread("input_image.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    color_info = detector.analyze_color_distribution(image_rgb)
    
    print(f"\nMean RGB: {color_info.get('mean_rgb', 'N/A')}")
    print(f"Median RGB: {color_info.get('median_rgb', 'N/A')}")
    print(f"Color Variance: {color_info.get('color_variance', 'N/A'):.2f}")


if __name__ == "__main__":
    print("\n🔥 DeepFake Framework - Example Usage\n")
    
    
    
    
    
    
    
    
    
    print("\n⚠️  Note: Replace 'input_image.jpg' with actual image paths")
    print("Run examples by uncommenting the desired function call above\n")

