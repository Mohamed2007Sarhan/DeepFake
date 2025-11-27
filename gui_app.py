"""
DeepFake Framework - GUI Application
Modern graphical interface with logo and advanced options
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import threading
from deepfake import DeepFakeFramework
import os


class DeepFakeGUI:
    """
    Modern GUI application for DeepFake Framework
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("DeepFake Framework - Advanced Image Processing")
        self.root.geometry("1200x800")
        self.root.configure(bg="#1e1e1e")
        
        # Initialize framework
        self.framework = None
        self.current_image_path = None
        self.current_image = None
        self.processed_image = None
        
        # Setup UI
        self.setup_ui()
        self.load_framework()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Header with logo
        self.create_header(main_frame)
        
        # Left panel - Controls
        self.create_control_panel(main_frame)
        
        # Right panel - Image display
        self.create_image_panel(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """Create header with logo and title"""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Logo and title
        logo_frame = ttk.Frame(header_frame)
        logo_frame.pack(side=tk.LEFT, padx=10)
        
        logo_label = tk.Label(
            logo_frame,
            text="🔥 DeepFake",
            font=("Arial", 24, "bold"),
            bg="#1e1e1e",
            fg="#ff6b6b"
        )
        logo_label.pack()
        
        subtitle_label = tk.Label(
            logo_frame,
            text="Advanced Image Processing Framework",
            font=("Arial", 10),
            bg="#1e1e1e",
            fg="#cccccc"
        )
        subtitle_label.pack()
        
        version_label = tk.Label(
            header_frame,
            text="v2.1 Enhanced",
            font=("Arial", 9),
            bg="#1e1e1e",
            fg="#888888"
        )
        version_label.pack(side=tk.RIGHT, padx=10)
    
    def create_control_panel(self, parent):
        """Create control panel with options"""
        control_frame = ttk.LabelFrame(parent, text="Controls & Options", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.columnconfigure(0, weight=1)
        
        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            file_frame,
            text="Select Image",
            command=self.select_image
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            file_frame,
            text="Save Result",
            command=self.save_result
        ).pack(side=tk.LEFT, padx=5)
        
        # Tab notebook for different modes
        self.notebook = ttk.Notebook(control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Clothing Removal Tab (FLAGSHIP FEATURE - First tab)
        removal_tab = ttk.Frame(self.notebook)
        self.notebook.add(removal_tab, text="⭐ Remove Clothing")
        
        # Image Processing Tab
        img_tab = ttk.Frame(self.notebook)
        self.notebook.add(img_tab, text="Image Processing")
        
        # Face Swap Tab
        swap_tab = ttk.Frame(self.notebook)
        self.notebook.add(swap_tab, text="Face Swap")
        
        # Voice Clone Tab
        voice_tab = ttk.Frame(self.notebook)
        self.notebook.add(voice_tab, text="Voice Clone")
        
        # Setup clothing removal tab (FIRST)
        self.setup_clothing_removal_tab(removal_tab)
        
        # Setup image processing tab
        self.setup_image_processing_tab(img_tab)
        
        # Setup face swap tab
        self.setup_face_swap_tab(swap_tab)
        
        # Setup voice clone tab
        self.setup_voice_clone_tab(voice_tab)
        
        # Info/Log area (shared across tabs)
        info_frame = ttk.LabelFrame(control_frame, text="Information", padding="5")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.info_text = scrolledtext.ScrolledText(
            info_frame,
            height=8,
            wrap=tk.WORD,
            bg="#2d2d2d",
            fg="#cccccc",
            font=("Consolas", 9)
        )
        self.info_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_clothing_removal_tab(self, parent):
        """Setup dedicated clothing removal tab - FLAGSHIP FEATURE"""
        # Info banner
        info_frame = ttk.LabelFrame(parent, text="⭐ FLAGSHIP FEATURE", padding="10")
        info_frame.pack(fill=tk.X, pady=5)
        
        info_text = tk.Label(
            info_frame,
            text="The Most Advanced Clothing Removal System\nRealistic, body-aware processing with AI-powered results",
            font=("Arial", 9),
            bg="#2d2d2d",
            fg="#FFD700",
            justify=tk.CENTER
        )
        info_text.pack()
        
        # Image selection
        image_frame = ttk.LabelFrame(parent, text="Image Selection", padding="10")
        image_frame.pack(fill=tk.X, pady=5)
        
        self.removal_image_path = None
        self.removal_image_label = ttk.Label(image_frame, text="No image selected")
        self.removal_image_label.pack(anchor=tk.W, pady=2)
        
        ttk.Button(
            image_frame,
            text="Select Image to Process",
            command=self.select_image_for_removal
        ).pack(fill=tk.X, pady=5)
        
        # Advanced options
        options_frame = ttk.LabelFrame(parent, text="Removal Options", padding="10")
        options_frame.pack(fill=tk.X, pady=10)
        
        # Use AI model
        self.use_ai_model_var = tk.BooleanVar(value=False)
        ai_check = ttk.Checkbutton(
            options_frame,
            text="Use Advanced AI Model (Better Quality)",
            variable=self.use_ai_model_var
        )
        ai_check.pack(anchor=tk.W, pady=5)
        
        # Model status
        self.model_status_label = ttk.Label(
            options_frame,
            text="Status: Traditional method (works great!)",
            font=("Arial", 8),
            foreground="#888888"
        )
        self.model_status_label.pack(anchor=tk.W, pady=2)
        
        # Check for model
        self.check_model_availability()
        
        # Gender override
        ttk.Label(options_frame, text="Gender (for better results):").pack(anchor=tk.W, pady=(10, 2))
        self.removal_gender_var = tk.StringVar(value="auto")
        gender_combo = ttk.Combobox(
            options_frame,
            textvariable=self.removal_gender_var,
            values=["auto", "male", "female"],
            state="readonly",
            width=20
        )
        gender_combo.pack(fill=tk.X, pady=2)
        
        # Body type override
        ttk.Label(options_frame, text="Body Type (for better results):").pack(anchor=tk.W, pady=(10, 2))
        self.removal_body_type_var = tk.StringVar(value="auto")
        body_combo = ttk.Combobox(
            options_frame,
            textvariable=self.removal_body_type_var,
            values=["auto", "Rectangle", "Pear", "Inverted Triangle", "Hourglass"],
            state="readonly",
            width=20
        )
        body_combo.pack(fill=tk.X, pady=2)
        
        # Blend factor
        blend_frame = ttk.Frame(options_frame)
        blend_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(blend_frame, text="Realism Level:").pack(anchor=tk.W)
        self.removal_blend_var = tk.DoubleVar(value=0.85)
        blend_scale = ttk.Scale(
            blend_frame,
            from_=0.5,
            to=1.0,
            variable=self.removal_blend_var,
            orient=tk.HORIZONTAL,
            command=lambda v: self.update_removal_blend_label(blend_frame, v)
        )
        blend_scale.pack(fill=tk.X, pady=2)
        self.removal_blend_label = ttk.Label(blend_frame, text="0.85 (Recommended)")
        self.removal_blend_label.pack(anchor=tk.W)
        self.removal_blend_label.pack(anchor=tk.W)
        
        # Realistic mode
        self.removal_realistic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Enable Realistic Mode (Enhanced Skin Generation)",
            variable=self.removal_realistic_var
        ).pack(anchor=tk.W, pady=5)
        
        # Remove button - HIGHLIGHTED
        remove_btn = tk.Button(
            parent,
            text="🔥 REMOVE CLOTHING NOW",
            command=self.remove_clothing_dedicated,
            bg="#FF5722",
            fg="white",
            font=("Arial", 14, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=15,
            cursor="hand2",
            activebackground="#E64A19"
        )
        remove_btn.pack(pady=20, fill=tk.X)
    
    def setup_image_processing_tab(self, parent):
        """Setup image processing tab"""
        options_frame = ttk.LabelFrame(parent, text="Processing Options", padding="10")
        options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(options_frame, text="Processing Mode:").pack(anchor=tk.W, pady=2)
        self.mode_var = tk.StringVar(value="remove_only")
        mode_combo = ttk.Combobox(
            options_frame,
            textvariable=self.mode_var,
            values=["remove_only", "complete", "color_only", "size_only"],
            state="readonly",
            width=20
        )
        mode_combo.pack(fill=tk.X, pady=2)
        
        self.realistic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Realistic Mode (Enhanced)",
            variable=self.realistic_var
        ).pack(anchor=tk.W, pady=5)
        
        ttk.Label(options_frame, text="Gender Override (optional):").pack(anchor=tk.W, pady=2)
        self.gender_var = tk.StringVar(value="auto")
        gender_combo = ttk.Combobox(
            options_frame,
            textvariable=self.gender_var,
            values=["auto", "male", "female"],
            state="readonly",
            width=20
        )
        gender_combo.pack(fill=tk.X, pady=2)
        
        ttk.Label(options_frame, text="Body Type Override (optional):").pack(anchor=tk.W, pady=2)
        self.body_type_var = tk.StringVar(value="auto")
        body_combo = ttk.Combobox(
            options_frame,
            textvariable=self.body_type_var,
            values=["auto", "Rectangle", "Pear", "Inverted Triangle", "Hourglass"],
            state="readonly",
            width=20
        )
        body_combo.pack(fill=tk.X, pady=2)
        
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Settings", padding="10")
        advanced_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(advanced_frame, text="Blend Factor: 0.8").pack(anchor=tk.W)
        self.blend_var = tk.DoubleVar(value=0.8)
        blend_scale = ttk.Scale(
            advanced_frame,
            from_=0.0,
            to=1.0,
            variable=self.blend_var,
            orient=tk.HORIZONTAL
        )
        blend_scale.pack(fill=tk.X, pady=2)
        
        process_btn = tk.Button(
            parent,
            text="🚀 Process Image",
            command=self.process_image,
            bg="#FF5722",
            fg="white",
            font=("Arial", 13, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=12,
            cursor="hand2"
        )
        process_btn.pack(pady=20, fill=tk.X)
    
    def setup_face_swap_tab(self, parent):
        """Setup face swap tab"""
        source_frame = ttk.LabelFrame(parent, text="Source Image", padding="10")
        source_frame.pack(fill=tk.X, pady=5)
        
        self.source_image_path = None
        self.source_label = ttk.Label(source_frame, text="No source image selected")
        self.source_label.pack(anchor=tk.W, pady=2)
        
        ttk.Button(
            source_frame,
            text="Select Source Image",
            command=self.select_source_image
        ).pack(fill=tk.X, pady=5)
        
        target_frame = ttk.LabelFrame(parent, text="Target (Image or Video)", padding="10")
        target_frame.pack(fill=tk.X, pady=5)
        
        self.target_path = None
        self.target_label = ttk.Label(target_frame, text="No target selected")
        self.target_label.pack(anchor=tk.W, pady=2)
        
        target_btn_frame = ttk.Frame(target_frame)
        target_btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            target_btn_frame,
            text="Select Target Image",
            command=lambda: self.select_target(is_video=False)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(
            target_btn_frame,
            text="Select Target Video",
            command=lambda: self.select_target(is_video=True)
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        swap_options_frame = ttk.LabelFrame(parent, text="Swap Options", padding="10")
        swap_options_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(swap_options_frame, text="Blend Mode:").pack(anchor=tk.W, pady=2)
        self.swap_blend_var = tk.StringVar(value="seamless")
        blend_combo = ttk.Combobox(
            swap_options_frame,
            textvariable=self.swap_blend_var,
            values=["seamless", "linear", "feathered"],
            state="readonly",
            width=20
        )
        blend_combo.pack(fill=tk.X, pady=2)
        
        swap_btn = tk.Button(
            parent,
            text="🔄 Swap Faces",
            command=self.swap_faces,
            bg="#2196F3",
            fg="white",
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        swap_btn.pack(pady=20, fill=tk.X)
    
    def setup_voice_clone_tab(self, parent):
        """Setup voice clone tab"""
        sample_frame = ttk.LabelFrame(parent, text="Voice Sample", padding="10")
        sample_frame.pack(fill=tk.X, pady=5)
        
        self.voice_sample_path = None
        self.voice_sample_label = ttk.Label(sample_frame, text="No voice sample selected")
        self.voice_sample_label.pack(anchor=tk.W, pady=2)
        
        ttk.Button(
            sample_frame,
            text="Select Voice Sample",
            command=self.select_voice_sample
        ).pack(fill=tk.X, pady=5)
        
        text_frame = ttk.LabelFrame(parent, text="Text to Synthesize", padding="10")
        text_frame.pack(fill=tk.X, pady=5)
        
        self.voice_text_var = tk.StringVar()
        text_entry = ttk.Entry(text_frame, textvariable=self.voice_text_var, width=50)
        text_entry.pack(fill=tk.X, pady=5)
        
        clone_btn = tk.Button(
            parent,
            text="🎤 Clone Voice",
            command=self.clone_voice,
            bg="#9C27B0",
            fg="white",
            font=("Arial", 12, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2"
        )
        clone_btn.pack(pady=20, fill=tk.X)
    
    def create_image_panel(self, parent):
        """Create image display panel"""
        image_frame = ttk.LabelFrame(parent, text="Image Preview", padding="10")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        canvas_frame = ttk.Frame(image_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#2d2d2d",
            highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Image toggle
        toggle_frame = ttk.Frame(image_frame)
        toggle_frame.pack(fill=tk.X, pady=5)
        
        self.show_original_btn = tk.Button(
            toggle_frame,
            text="Show Original",
            command=lambda: self.show_image(self.current_image) if self.current_image is not None else None,
            bg="#4CAF50",
            fg="white"
        )
        self.show_original_btn.pack(side=tk.LEFT, padx=5)
        
        self.show_processed_btn = tk.Button(
            toggle_frame,
            text="Show Processed",
            command=lambda: self.show_image(self.processed_image) if self.processed_image is not None else None,
            bg="#2196F3",
            fg="white"
        )
        self.show_processed_btn.pack(side=tk.LEFT, padx=5)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(fill=tk.X)
    
    def load_framework(self):
        """Load DeepFake framework"""
        try:
            self.framework = DeepFakeFramework()
            self.log_info("✓ Framework loaded successfully")
        except Exception as e:
            self.log_error(f"Failed to load framework: {e}")
    
    def select_image(self):
        """Select input image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.current_image_path = file_path
            self.load_image(file_path)
            self.log_info(f"Image loaded: {Path(file_path).name}")
    
    def load_image(self, image_path):
        """Load image from path"""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.current_image = image
                self.show_image(image)
                return True
        except Exception as e:
            self.log_error(f"Error loading image: {e}")
        return False
    
    def show_image(self, image):
        """Display image on canvas"""
        if image is None:
            return
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            h, w = image.shape[:2]
            scale = min(canvas_width / w, canvas_height / h) * 0.9
            new_w, new_h = int(w * scale), int(h * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
        else:
            # Fallback if canvas not ready
            h, w = image.shape[:2]
            max_size = 600
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
        
        # Convert to PIL and display
        image_pil = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(image_pil)
        
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2 if canvas_width > 1 else 300,
            canvas_height // 2 if canvas_height > 1 else 300,
            image=self.photo,
            anchor=tk.CENTER
        )
    
    def process_image(self):
        """Process image based on selected mode"""
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        self.status_var.set("Processing...")
        self.log_info("Starting processing...")
        
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Process image in background thread"""
        try:
            mode = self.mode_var.get()
            gender_override = self.gender_var.get()
            body_type_override = self.body_type_var.get()
            
            gender = None if gender_override == "auto" else gender_override
            body_type = None if body_type_override == "auto" else body_type_override
            
            if mode == "complete":
                result = self.framework.process_complete(
                    self.current_image_path,
                    return_metadata=True
                )
                self.processed_image = result["processed_image"]
            elif mode == "remove_only":
                self.processed_image = self.framework.remove_clothes(
                    self.current_image_path,
                    gender=gender,
                    body_type=body_type
                )
            elif mode == "color_only":
                result = self.framework.get_clothes_color(self.current_image_path)
                self.log_info(f"Colors: {result}")
            elif mode == "size_only":
                result = self.framework.get_body_size(self.current_image_path)
                self.log_info(f"Body size: {result}")
            
            if self.processed_image is not None:
                self.root.after(0, lambda: self.show_image(self.processed_image))
                self.root.after(0, lambda: self.status_var.set("Processing complete!"))
                self.root.after(0, lambda: self.log_info("✓ Processing complete"))
        except Exception as e:
            self.root.after(0, lambda: self.log_error(str(e)))
            self.root.after(0, lambda: self.status_var.set("Error occurred"))
    
    def save_result(self):
        """Save processed image"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Result",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            try:
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
                self.log_info(f"Saved: {Path(file_path).name}")
                messagebox.showinfo("Success", f"Image saved to {file_path}")
            except Exception as e:
                self.log_error(f"Error saving: {e}")
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def select_image_for_removal(self):
        """Select image specifically for clothing removal"""
        file_path = filedialog.askopenfilename(
            title="Select Image for Clothing Removal",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.removal_image_path = file_path
            self.current_image_path = file_path
            self.removal_image_label.config(text=f"Image: {Path(file_path).name}")
            self.load_image(file_path)
            self.log_info(f"Image selected for clothing removal: {Path(file_path).name}")
    
    def remove_clothing_dedicated(self):
        """Dedicated clothing removal handler"""
        if not self.removal_image_path and not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        image_path = self.removal_image_path or self.current_image_path
        
        output_path = filedialog.asksaveasfilename(
            title="Save Clothing Removed Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if not output_path:
            return
        
        self.status_var.set("Removing clothing... This may take a moment")
        self.log_info("="*50)
        self.log_info("⭐ CLOTHING REMOVAL - FLAGSHIP FEATURE")
        self.log_info("="*50)
        self.log_info(f"Processing: {Path(image_path).name}")
        self.log_info(f"Using AI model: {self.use_ai_model_var.get()}")
        
        thread = threading.Thread(
            target=self._remove_clothing_thread,
            args=(image_path, output_path)
        )
        thread.daemon = True
        thread.start()
    
    def _remove_clothing_thread(self, image_path, output_path):
        """Remove clothing in background thread"""
        try:
            gender = None if self.removal_gender_var.get() == "auto" else self.removal_gender_var.get()
            body_type = None if self.removal_body_type_var.get() == "auto" else self.removal_body_type_var.get()
            
            result = self.framework.remove_clothes(
                image_path,
                output_path=output_path,
                gender=gender,
                body_type=body_type,
                blend_factor=self.removal_blend_var.get(),
                use_advanced=self.use_ai_model_var.get()
            )
            
            self.processed_image = result
            
            self.root.after(0, lambda: self._update_after_removal(output_path))
        except Exception as e:
            self.root.after(0, lambda: self._handle_removal_error(e))
    
    def _update_after_removal(self, output_path):
        """Update UI after clothing removal"""
        if self.processed_image is not None:
            self.show_image(self.processed_image)
            self.show_processed_btn.config(bg="#FF5722")
            self.status_var.set("Clothing removal complete!")
            self.log_info("="*50)
            self.log_info("✓ CLOTHING REMOVAL COMPLETE!")
            self.log_info(f"  Saved to: {Path(output_path).name}")
            self.log_info(f"  Result shape: {self.processed_image.shape}")
            self.log_info("="*50)
            messagebox.showinfo("Success", f"Clothing removal complete!\n\nSaved to: {output_path}")
    
    def _handle_removal_error(self, error):
        """Handle clothing removal errors"""
        error_msg = f"Clothing removal error: {error}"
        self.log_info(f"✗ {error_msg}")
        self.status_var.set("Error occurred")
        messagebox.showerror("Clothing Removal Error", error_msg)
    
    def update_removal_blend_label(self, parent, value):
        """Update blend factor label for removal tab"""
        val = float(value)
        for widget in parent.winfo_children():
            if isinstance(widget, ttk.Label):
                text = widget.cget("text")
                if text.startswith("0.") or "Recommended" in text or "Fast" in text or "Good" in text or "Excellent" in text or "Maximum" in text:
                    if val < 0.7:
                        label_text = f"{val:.2f} (Fast)"
                    elif val < 0.85:
                        label_text = f"{val:.2f} (Good)"
                    elif val < 0.95:
                        label_text = f"{val:.2f} (Excellent - Recommended)"
                    else:
                        label_text = f"{val:.2f} (Maximum Realism)"
                    widget.config(text=label_text)
                    break
    
    def check_model_availability(self):
        """Check if AI model is available"""
        try:
            from deepfake.utils.model_downloader import ModelDownloader
            downloader = ModelDownloader()
            model_path = downloader.get_model_path()
            
            if model_path and Path(model_path).exists():
                self.model_status_label.config(
                    text=f"Status: ✓ AI Model Ready ({Path(model_path).name})",
                    foreground="#4CAF50"
                )
            else:
                self.model_status_label.config(
                    text="Status: Traditional method (Download model for better quality)",
                    foreground="#FFA500"
                )
        except:
            self.model_status_label.config(
                text="Status: Traditional method",
                foreground="#888888"
            )
    
    def select_source_image(self):
        """Select source image for face swap"""
        file_path = filedialog.askopenfilename(
            title="Select Source Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.source_image_path = file_path
            self.source_label.config(text=f"Source: {Path(file_path).name}")
    
    def select_target(self, is_video=False):
        """Select target image or video for face swap"""
        if is_video:
            file_path = filedialog.askopenfilename(
                title="Select Target Video",
                filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
            )
        else:
            file_path = filedialog.askopenfilename(
                title="Select Target Image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
            )
        if file_path:
            self.target_path = file_path
            self.target_label.config(text=f"Target: {Path(file_path).name}")
    
    def swap_faces(self):
        """Swap faces"""
        if not self.source_image_path or not self.target_path:
            messagebox.showwarning("Warning", "Please select both source and target")
            return
        
        output_path = filedialog.asksaveasfilename(
            title="Save Swapped Result",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if not output_path:
            return
        
        self.status_var.set("Swapping faces...")
        self.log_info("Swapping faces...")
        
        thread = threading.Thread(
            target=lambda: self._swap_faces_thread(output_path)
        )
        thread.daemon = True
        thread.start()
    
    def _swap_faces_thread(self, output_path):
        """Swap faces in background thread"""
        try:
            is_video = Path(self.target_path).suffix.lower() in ['.mp4', '.avi', '.mov']
            
            if is_video:
                result = self.framework.swap_face_to_video(
                    self.source_image_path,
                    self.target_path,
                    output_path
                )
            else:
                result = self.framework.swap_faces_image(
                    self.source_image_path,
                    self.target_path,
                    output_path,
                    blend_mode=self.swap_blend_var.get()
                )
            
            self.root.after(0, lambda: self.log_info("✓ Face swap complete"))
            self.root.after(0, lambda: self.status_var.set("Face swap complete!"))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Saved to: {output_path}"))
        except Exception as e:
            self.root.after(0, lambda: self.log_error(str(e)))
    
    def select_voice_sample(self):
        """Select voice sample for cloning"""
        file_path = filedialog.askopenfilename(
            title="Select Voice Sample",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*")]
        )
        if file_path:
            self.voice_sample_path = file_path
            self.voice_sample_label.config(text=f"Sample: {Path(file_path).name}")
    
    def clone_voice(self):
        """Clone voice"""
        if not self.voice_sample_path:
            messagebox.showwarning("Warning", "Please select a voice sample")
            return
        
        text = self.voice_text_var.get()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to synthesize")
            return
        
        output_path = filedialog.asksaveasfilename(
            title="Save Cloned Voice",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if not output_path:
            return
        
        self.status_var.set("Cloning voice...")
        self.log_info("Cloning voice...")
        
        thread = threading.Thread(
            target=lambda: self._clone_voice_thread(output_path, text)
        )
        thread.daemon = True
        thread.start()
    
    def _clone_voice_thread(self, output_path, text):
        """Clone voice in background thread"""
        try:
            result = self.framework.clone_voice(
                self.voice_sample_path,
                text,
                output_path
            )
            self.root.after(0, lambda: self.log_info("✓ Voice cloning complete"))
            self.root.after(0, lambda: self.status_var.set("Voice cloning complete!"))
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Saved to: {output_path}"))
        except Exception as e:
            self.root.after(0, lambda: self.log_error(str(e)))
    
    def log_info(self, message):
        """Log info message"""
        self.info_text.insert(tk.END, f"[INFO] {message}\n")
        self.info_text.see(tk.END)
    
    def log_error(self, message):
        """Log error message"""
        self.info_text.insert(tk.END, f"[ERROR] {message}\n", "error")
        self.info_text.see(tk.END)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = DeepFakeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

