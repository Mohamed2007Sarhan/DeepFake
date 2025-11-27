"""
Model Downloader Script
Downloads the best recommended model for clothing removal
"""

import argparse
from deepfake.utils.model_downloader import ModelDownloader, setup_recommended_model


def main():
    parser = argparse.ArgumentParser(description="Download AI models for clothing removal")
    parser.add_argument("--model", "-m", type=str, help="Model ID to download (default: recommended)")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--auto", "-a", action="store_true", help="Auto-download recommended model")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    
    if args.list:
        print("\n" + "="*60)
        print("Available Models for Clothing Removal")
        print("="*60 + "\n")
        
        for model_id, model_info in downloader.list_available_models().items():
            exists = "✓" if downloader.check_model_exists(model_id) else "✗"
            recommended = " [RECOMMENDED]" if model_info.get("recommended") else ""
            
            print(f"{exists} {model_id}{recommended}")
            print(f"  Name: {model_info['name']}")
            print(f"  Description: {model_info['description']}")
            print(f"  Size: {model_info['size_mb']} MB")
            print(f"  File: {model_info['filename']}")
            print()
        
        return
    
    if args.auto or args.model is None:
        # Download recommended model
        print("\n" + "="*60)
        print("Downloading Recommended Model")
        print("="*60 + "\n")
        
        recommended = downloader.get_recommended_model()
        print(f"Model: {recommended['name']}")
        print(f"Description: {recommended['description']}")
        print(f"Size: {recommended['size_mb']} MB\n")
        
        model_path = downloader.download_model()
        
        if model_path:
            print(f"\n✓ Setup complete! Model ready at: {model_path}")
            print("\nYou can now use advanced clothing removal with better quality.")
        else:
            print("\n⚠ Model download failed or not available.")
            print("The framework will use traditional inpainting methods (which work great!).")
            print("\nTo download manually:")
            print(f"1. Visit: {recommended['url']}")
            print(f"2. Download and place in: models/")
            print(f"3. Name it: {recommended['filename']}")
    else:
        # Download specific model
        model_path = downloader.download_model(args.model)
        if model_path:
            print(f"\n✓ Model downloaded: {model_path}")


if __name__ == "__main__":
    main()

