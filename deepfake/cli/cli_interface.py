utf-8"""
Enhanced CLI Interface for DeepFake Framework
Modern command-line interface with colors, menus, and progress indicators
"""

import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
import argparse

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.layout import Layout
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

from .. import DeepFakeFramework
from ..utils.model_downloader import setup_recommended_model


class CLIInterface:
    """
    Enhanced CLI interface with rich formatting and interactive menus
    """
    
    def __init__(self):
        """Initialize CLI interface"""
        if RICH_AVAILABLE:
            self.console = Console()
            self.use_rich = True
        else:
            self.console = None
            self.use_rich = False
            if COLORAMA_AVAILABLE:
                self.use_colorama = True
            else:
                self.use_colorama = False
        
        self.framework: Optional[DeepFakeFramework] = None
    
    def print_header(self, title: str):
        """Print formatted header"""
        if self.use_rich:
            self.console.print(Panel.fit(
                f"[bold magenta]{title}[/bold magenta]",
                border_style="bright_blue"
            ))
        elif self.use_colorama:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}")
            print(f"{title:^60}")
            print(f"{'='*60}{Style.RESET_ALL}\n")
        else:
            print(f"\n{'='*60}")
            print(f"{title:^60}")
            print(f"{'='*60}\n")
    
    def print_success(self, message: str):
        """Print success message"""
        if self.use_rich:
            self.console.print(f"[bold green]✓[/bold green] {message}")
        elif self.use_colorama:
            print(f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}")
        else:
            print(f"✓ {message}")
    
    def print_error(self, message: str):
        """Print error message"""
        if self.use_rich:
            self.console.print(f"[bold red]✗ Error:[/bold red] {message}")
        elif self.use_colorama:
            print(f"{Fore.RED}✗ Error: {message}{Style.RESET_ALL}")
        else:
            print(f"✗ Error: {message}")
    
    def print_warning(self, message: str):
        """Print warning message"""
        if self.use_rich:
            self.console.print(f"[bold yellow]⚠ Warning:[/bold yellow] {message}")
        elif self.use_colorama:
            print(f"{Fore.YELLOW}⚠ Warning: {message}{Style.RESET_ALL}")
        else:
            print(f"⚠ Warning: {message}")
    
    def print_info(self, message: str):
        """Print info message"""
        if self.use_rich:
            self.console.print(f"[cyan]ℹ[/cyan] {message}")
        elif self.use_colorama:
            print(f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}")
        else:
            print(f"ℹ {message}")
    
    def create_progress(self) -> Progress:
        """Create progress bar"""
        if self.use_rich:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=self.console
            )
        return None
    
    def show_menu(self) -> str:
        """Show interactive main menu"""
        if self.use_rich:
            menu_table = Table(title="DeepFake Framework - Main Menu", show_header=True, header_style="bold magenta")
            menu_table.add_column("Option", style="cyan", width=10)
            menu_table.add_column("Description", style="white")
            
            menu_table.add_row("1", "⭐ Remove Clothing (FLAGSHIP FEATURE)")
            menu_table.add_row("2", "Image Processing (Color, Size, Complete)")
            menu_table.add_row("3", "Face Swapping (Image-to-Image)")
            menu_table.add_row("4", "Face Swapping (Image-to-Video)")
            menu_table.add_row("5", "Voice Cloning & Synthesis")
            menu_table.add_row("6", "Framework Information")
            menu_table.add_row("0", "Exit")
            
            self.console.print(menu_table)
            choice = Prompt.ask("\n[bold cyan]Select an option[/bold cyan]", choices=["0", "1", "2", "3", "4", "5", "6"])
        else:
            print("\n" + "="*60)
            print("DeepFake Framework - Main Menu")
            print("="*60)
            print("1. ⭐ Remove Clothing (FLAGSHIP FEATURE)")
            print("2. Image Processing (Color, Size, Complete)")
            print("3. Face Swapping (Image-to-Image)")
            print("4. Face Swapping (Image-to-Video)")
            print("5. Voice Cloning & Synthesis")
            print("6. Framework Information")
            print("0. Exit")
            print("="*60)
            choice = input("\nSelect an option: ").strip()
        
        return choice
    
    def show_image_processing_menu(self):
        """Show image processing submenu"""
        if self.use_rich:
            menu_table = Table(title="Image Processing Options", show_header=True)
            menu_table.add_column("Option", style="cyan")
            menu_table.add_column("Description", style="white")
            
            menu_table.add_row("1", "Complete Processing (Color + Size + Removal)")
            menu_table.add_row("2", "Color Detection Only")
            menu_table.add_row("3", "Body Size Estimation Only")
            menu_table.add_row("4", "Clothing Removal Only")
            menu_table.add_row("0", "Back to Main Menu")
            
            self.console.print(menu_table)
            choice = Prompt.ask("\n[bold cyan]Select option[/bold cyan]", choices=["0", "1", "2", "3", "4"])
        else:
            print("\nImage Processing Options:")
            print("1. Complete Processing (Color + Size + Removal)")
            print("2. Color Detection Only")
            print("3. Body Size Estimation Only")
            print("4. Clothing Removal Only")
            print("0. Back to Main Menu")
            choice = input("\nSelect option: ").strip()
        
        return choice
    
    def prompt_file(self, prompt: str, file_types: Optional[List[str]] = None) -> Optional[str]:
        """Prompt for file path"""
        if self.use_rich:
            path = Prompt.ask(f"[cyan]{prompt}[/cyan]")
        else:
            path = input(f"{prompt}: ").strip()
        
        if not path:
            return None
        
        path = path.strip('"\'')  
        
        if not Path(path).exists():
            self.print_error(f"File not found: {path}")
            return None
        
        return path
    
    def prompt_save_file(self, prompt: str, default: Optional[str] = None) -> Optional[str]:
        """Prompt for save file path"""
        if default:
            prompt_with_default = f"{prompt} [{default}]"
        else:
            prompt_with_default = prompt
        
        if self.use_rich:
            path = Prompt.ask(f"[cyan]{prompt_with_default}[/cyan]", default=default or "")
        else:
            path = input(f"{prompt_with_default}: ").strip()
        
        if not path and default:
            path = default
        
        return path
    
    def prompt_text(self, prompt: str, default: Optional[str] = None) -> str:
        """Prompt for text input"""
        if self.use_rich:
            text = Prompt.ask(f"[cyan]{prompt}[/cyan]", default=default or "")
        else:
            if default:
                user_input = input(f"{prompt} [{default}]: ").strip()
                text = user_input if user_input else default
            else:
                text = input(f"{prompt}: ").strip()
        
        return text
    
    def display_results_table(self, title: str, data: Dict[str, Any]):
        """Display results in a formatted table"""
        if self.use_rich:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            for key, value in data.items():
                if value is not None:
                    table.add_row(str(key), str(value))
            
            self.console.print(table)
        else:
            print(f"\n{title}:")
            print("-" * 60)
            for key, value in data.items():
                if value is not None:
                    print(f"{key:30}: {value}")
            print("-" * 60)
    
    def handle_image_processing(self):
        """Handle image processing menu"""
        choice = self.show_image_processing_menu()
        
        if choice == "0":
            return
        
        
        input_path = self.prompt_file("Enter input image path")
        if not input_path:
            return
        
        if not self.framework:
            self.framework = DeepFakeFramework()
        
        self.print_info(f"Processing: {Path(input_path).name}")
        
        if choice == "1":  
            output_path = self.prompt_save_file("Enter output path", "output_complete.png")
            if not output_path:
                output_path = "output_complete.png"
            
            with self.create_progress() if self.use_rich else None:
                try:
                    result = self.framework.process_complete(input_path, output_path, return_metadata=True)
                    self._display_complete_results(result, output_path)
                except Exception as e:
                    self.print_error(str(e))
        
        elif choice == "2":  
            try:
                result = self.framework.get_clothes_color(input_path)
                self._display_color_results(result)
            except Exception as e:
                self.print_error(str(e))
        
        elif choice == "3":  
            try:
                result = self.framework.get_body_size(input_path)
                self._display_size_results(result)
            except Exception as e:
                self.print_error(str(e))
        
        elif choice == "4":  
            output_path = self.prompt_save_file("Enter output path", "output_removed.png")
            if not output_path:
                output_path = "output_removed.png"
            
            try:
                result = self.framework.remove_clothes(input_path, output_path=output_path)
                self.print_success(f"Clothing removal complete! Saved to: {output_path}")
            except Exception as e:
                self.print_error(str(e))
    
    def handle_face_swap_image(self):
        """Handle image-to-image face swapping"""
        self.print_header("Face Swapping - Image to Image")
        
        source = self.prompt_file("Enter source image path (face to copy)")
        if not source:
            return
        
        target = self.prompt_file("Enter target image path (face to replace)")
        if not target:
            return
        
        output = self.prompt_save_file("Enter output path", "face_swapped.png")
        if not output:
            output = "face_swapped.png"
        
        if not self.framework:
            self.framework = DeepFakeFramework()
        
        self.print_info("Swapping faces...")
        
        try:
            with self.create_progress() if self.use_rich else None:
                metadata = self.framework.swap_faces_image(
                    source, target, output,
                    blend_mode="seamless"
                )
            
            self.print_success(f"Face swap complete! Saved to: {output}")
            self.display_results_table("Swap Results", {
                "Source faces": metadata.get('source_faces_detected', 0),
                "Target faces": metadata.get('target_faces_detected', 0),
                "Blend mode": metadata.get('blend_mode', 'N/A')
            })
        except Exception as e:
            self.print_error(str(e))
    
    def handle_face_swap_video(self):
        """Handle image-to-video face swapping"""
        self.print_header("Face Swapping - Image to Video")
        
        source = self.prompt_file("Enter source image path (face to copy)")
        if not source:
            return
        
        target = self.prompt_file("Enter target video path")
        if not target:
            return
        
        output = self.prompt_save_file("Enter output video path", "face_swapped_video.mp4")
        if not output:
            output = "face_swapped_video.mp4"
        
        if not self.framework:
            self.framework = DeepFakeFramework()
        
        self.print_warning("Video processing may take a while...")
        
        try:
            progress = self.create_progress()
            if progress:
                with progress:
                    task = progress.add_task("Processing video...", total=100)
                    metadata = self.framework.swap_face_to_video(
                        source, target, output
                    )
                    progress.update(task, completed=100)
            else:
                metadata = self.framework.swap_face_to_video(source, target, output)
            
            self.print_success(f"Video processing complete! Saved to: {output}")
            self.display_results_table("Video Processing Results", {
                "Total frames": metadata.get('total_frames', 0),
                "Faces detected": metadata.get('faces_detected', 0),
                "FPS": metadata.get('fps', 0),
                "Resolution": f"{metadata.get('resolution', (0, 0))[0]}x{metadata.get('resolution', (0, 0))[1]}"
            })
        except Exception as e:
            self.print_error(str(e))
    
    def handle_voice_cloning(self):
        """Handle voice cloning"""
        self.print_header("Voice Cloning & Synthesis")
        
        sample = self.prompt_file("Enter voice sample audio path", ["*.wav", "*.mp3", "*.flac"])
        if not sample:
            return
        
        text = self.prompt_text("Enter text to synthesize")
        if not text:
            self.print_error("Text cannot be empty")
            return
        
        output = self.prompt_save_file("Enter output audio path", "synthesized_voice.wav")
        if not output:
            output = "synthesized_voice.wav"
        
        if not self.framework:
            self.framework = DeepFakeFramework()
        
        self.print_info("Cloning voice and synthesizing speech...")
        
        try:
            progress = self.create_progress()
            if progress:
                with progress:
                    task = progress.add_task("Processing...", total=100)
                    result = self.framework.clone_voice(sample, text, output)
                    progress.update(task, completed=100)
            else:
                result = self.framework.clone_voice(sample, text, output)
            
            self.print_success(f"Voice cloning complete! Saved to: {output}")
            
            profile = result.get('voice_profile', {}).get('features', {})
            synth = result.get('synthesis', {})
            
            self.display_results_table("Voice Profile", {
                "Pitch": f"{profile.get('pitch_mean', 0):.1f} Hz",
                "Tempo": f"{profile.get('tempo', 0):.1f} BPM",
                "Duration": f"{profile.get('duration', 0):.2f}s",
                "Synthesis method": synth.get('method', 'N/A'),
                "Output duration": f"{synth.get('duration', 0):.2f}s"
            })
        except Exception as e:
            self.print_error(str(e))
    
    def handle_framework_info(self):
        """Display framework information"""
        if not self.framework:
            self.framework = DeepFakeFramework()
        
        info = self.framework.get_framework_info()
        
        self.print_header("Framework Information")
        
        components_table = {}
        for comp, status in info.get('components', {}).items():
            components_table[comp.replace('_', ' ').title()] = status
        
        self.display_results_table("Framework Status", {
            "Version": info.get('version', 'N/A'),
            **components_table
        })
    
    def _display_complete_results(self, result: Dict, output_path: str):
        """Display complete processing results"""
        self.print_success(f"Processing complete! Saved to: {output_path}")
        
        
        if "clothes_color" in result and result["clothes_color"].get("primary_color"):
            pc = result["clothes_color"]["primary_color"]
            color_data = {
                "Primary Color": f"{pc.get('name', 'N/A')} ({pc.get('hex', 'N/A')})",
                "RGB": str(pc.get('rgb', 'N/A')),
                "Percentage": f"{pc.get('percentage', 0):.1f}%"
            }
            self.display_results_table("Color Detection", color_data)
        
        
        if "body_size" in result and "measurements" in result["body_size"]:
            m = result["body_size"]["measurements"]
            size_data = {
                "Body Type": m.get('body_type', 'N/A'),
                "Shoulder Width": f"{m.get('shoulder_width_cm', 0):.2f} cm" if m.get('shoulder_width_cm') else 'N/A',
                "Hip Width": f"{m.get('hip_width_cm', 0):.2f} cm" if m.get('hip_width_cm') else 'N/A',
                "Leg Length": f"{m.get('leg_length_cm', 0):.2f} cm" if m.get('leg_length_cm') else 'N/A'
            }
            
            if "gender" in result["body_size"]:
                g = result["body_size"]["gender"]
                size_data["Gender"] = f"{g.get('gender', 'N/A')} (confidence: {g.get('confidence', 0):.2f})"
            
            self.display_results_table("Body Measurements", size_data)
    
    def _display_color_results(self, result: Dict):
        """Display color detection results"""
        if result.get("primary_color"):
            pc = result["primary_color"]
            color_data = {
                "Primary Color": f"{pc.get('name', 'N/A')} ({pc.get('hex', 'N/A')})",
                "RGB": str(pc.get('rgb', 'N/A')),
                "Percentage": f"{pc.get('percentage', 0):.1f}%",
                "Total Colors": result.get('total_colors', 0)
            }
            self.display_results_table("Color Detection Results", color_data)
            
            if result.get('dominant_colors'):
                if self.use_rich:
                    colors_table = Table(title="All Dominant Colors")
                    colors_table.add_column("Color", style="cyan")
                    colors_table.add_column("HEX", style="white")
                    colors_table.add_column("Percentage", style="green")
                    
                    for color in result['dominant_colors'][:5]:
                        colors_table.add_row(
                            color.get('name', 'N/A'),
                            color.get('hex', 'N/A'),
                            f"{color.get('percentage', 0):.1f}%"
                        )
                    self.console.print(colors_table)
                else:
                    print("\nAll Dominant Colors:")
                    for color in result['dominant_colors'][:5]:
                        print(f"  - {color.get('name')}: {color.get('hex')} ({color.get('percentage'):.1f}%)")
    
    def _display_size_results(self, result: Dict):
        """Display body size results"""
        if "error" in result:
            self.print_error(result["error"])
            return
        
        measurements = result.get("measurements", {})
        size_data = {
            "Body Type": measurements.get('body_type', 'N/A'),
            "Shoulder Width": f"{measurements.get('shoulder_width_cm', 0):.2f} cm" if measurements.get('shoulder_width_cm') else 'N/A',
            "Hip Width": f"{measurements.get('hip_width_cm', 0):.2f} cm" if measurements.get('hip_width_cm') else 'N/A',
            "Leg Length": f"{measurements.get('leg_length_cm', 0):.2f} cm" if measurements.get('leg_length_cm') else 'N/A'
        }
        
        if "gender" in result:
            g = result["gender"]
            size_data["Gender"] = f"{g.get('gender', 'N/A')} (confidence: {g.get('confidence', 0):.2f})"
        
        self.display_results_table("Body Size Results", size_data)
    
    def handle_clothing_removal(self):
        """Handle dedicated clothing removal"""
        self.print_header("⭐ Clothing Removal - FLAGSHIP FEATURE")
        
        self.print_info("This is the most advanced feature - realistic clothing removal")
        
        
        input_path = self.prompt_file("Enter input image path")
        if not input_path:
            return
        
        
        if self.use_rich:
            use_advanced = Confirm.ask(
                "\n[bold yellow]Use advanced AI model?[/bold yellow] "
                "[dim](Requires model download, better quality)[/dim]",
                default=False
            )
        else:
            use_adv = input("\nUse advanced AI model? (y/n, default=n): ").strip().lower()
            use_advanced = use_adv == 'y'
        
        
        if self.use_rich:
            gender_choice = Prompt.ask(
                "\n[cyan]Gender[/cyan] ([dim]auto/male/female[/dim])",
                choices=["auto", "male", "female"],
                default="auto"
            )
        else:
            gender_choice = input("\nGender (auto/male/female, default=auto): ").strip().lower() or "auto"
        
        gender = None if gender_choice == "auto" else gender_choice
        
        
        if self.use_rich:
            body_choice = Prompt.ask(
                "[cyan]Body Type[/cyan] ([dim]auto/Rectangle/Pear/Inverted Triangle/Hourglass[/dim])",
                choices=["auto", "Rectangle", "Pear", "Inverted Triangle", "Hourglass"],
                default="auto"
            )
        else:
            body_choice = input("Body Type (auto/Rectangle/Pear/Inverted Triangle/Hourglass, default=auto): ").strip() or "auto"
        
        body_type = None if body_choice == "auto" else body_choice
        
        
        if self.use_rich:
            blend_str = Prompt.ask(
                "[cyan]Blend Factor[/cyan] ([dim]0.0-1.0, higher=more realistic, default=0.85[/dim])",
                default="0.85"
            )
        else:
            blend_str = input("Blend Factor (0.0-1.0, default=0.85): ").strip() or "0.85"
        
        try:
            blend_factor = float(blend_str)
            blend_factor = max(0.0, min(1.0, blend_factor))
        except:
            blend_factor = 0.85
        
        output_path = self.prompt_save_file("Enter output path", "clothing_removed.png")
        if not output_path:
            output_path = "clothing_removed.png"
        
        if not self.framework:
            self.framework = DeepFakeFramework()
        
        self.print_info(f"\nProcessing: {Path(input_path).name}")
        self.print_info(f"Settings: blend_factor={blend_factor:.2f}, gender={gender or 'auto'}, body_type={body_type or 'auto'}")
        
        try:
            progress = self.create_progress()
            if progress:
                with progress:
                    task = progress.add_task("[cyan]Removing clothing...", total=100)
                    result = self.framework.remove_clothes(
                        input_path,
                        output_path=output_path,
                        gender=gender,
                        body_type=body_type,
                        blend_factor=blend_factor,
                        use_advanced=use_advanced
                    )
                    progress.update(task, completed=100)
            else:
                result = self.framework.remove_clothes(
                    input_path,
                    output_path=output_path,
                    gender=gender,
                    body_type=body_type,
                    blend_factor=blend_factor,
                    use_advanced=use_advanced
                )
            
            self.print_success(f"\n✓ Clothing removal complete!")
            self.print_success(f"  Saved to: {output_path}")
            self.print_info(f"  Result shape: {result.shape}")
            
        except Exception as e:
            self.print_error(str(e))
    
    def run_interactive(self):
        """Run interactive CLI menu"""
        self.print_header("🔥 DeepFake Framework v2.1")
        
        if not RICH_AVAILABLE and not COLORAMA_AVAILABLE:
            self.print_warning(
                "For better CLI experience, install: pip install rich colorama"
            )
        
        if not self.framework:
            self.print_info("Initializing framework...")
            try:
                self.framework = DeepFakeFramework()
                self.print_success("Framework initialized successfully!")
            except Exception as e:
                self.print_error(f"Failed to initialize framework: {e}")
                return
        
        while True:
            try:
                choice = self.show_menu()
                
                if choice == "0":
                    self.print_success("Goodbye!")
                    break
                elif choice == "1":
                    self.handle_clothing_removal()
                elif choice == "2":
                    self.handle_image_processing()
                elif choice == "3":
                    self.handle_face_swap_image()
                elif choice == "4":
                    self.handle_face_swap_video()
                elif choice == "5":
                    self.handle_voice_cloning()
                elif choice == "6":
                    self.handle_framework_info()
                else:
                    self.print_error("Invalid option")
                
                if self.use_rich:
                    input("\nPress Enter to continue...")
                else:
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                self.print_warning("\nOperation cancelled by user")
                if Confirm.ask("Exit?") if self.use_rich else input("Exit? (y/n): ").lower() == 'y':
                    break
            except Exception as e:
                self.print_error(f"Unexpected error: {e}")
                if self.use_rich:
                    input("\nPress Enter to continue...")
                else:
                    input("\nPress Enter to continue...")


def main_interactive():
    """Main entry point for interactive CLI"""
    cli = CLIInterface()
    cli.run_interactive()


if __name__ == "__main__":
    main_interactive()

