utf-8"""
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
        
        
        self.framework = None
        self.current_image_path = None
        self.current_image = None
        self.processed_image = None
        
        
        self.setup_ui()
        self.load_framework()
    
    def setup_ui(self):
        """Setup the user interface"""
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        
        self.create_header(main_frame)
        
        
        self.create_control_panel(main_frame)
        
        
        self.create_image_panel(main_frame)
        
        
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """Create header with logo and title"""
        header_frame = ttk.Frame(parent, style="Header.TFrame")
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        
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
        
        
        self.notebook = ttk.Notebook(control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        
        removal_tab = ttk.Frame(self.notebook)
        self.notebook.add(removal_tab, text="⭐ Remove Clothing")
        
        
        img_tab = ttk.Frame(self.notebook)
        self.notebook.add(img_tab, text="Image Processing")
        
        
        swap_tab = ttk.Frame(self.notebook)
        self.notebook.add(swap_tab, text="Face Swap")
        
        
        voice_tab = ttk.Frame(self.notebook)
        self.notebook.add(voice_tab, text="Voice Clone")
        
        
        self.setup_clothing_removal_tab(removal_tab)
        
        
        self.setup_image_processing_tab(img_tab)
        
        
        self.setup_face_swap_tab(swap_tab)
        
        
        self.setup_voice_clone_tab(voice_tab)
    
    def setup_clothing_removal_tab(self, parent):
        """Setup dedicated clothing removal tab - FLAGSHIP FEATURE"""
        
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
        
        
        image_frame = ttk.LabelFrame(parent, text="Image Selection", padding="10")
        image_frame.pack(fill=tk.X, pady=5)
        
        self.removal_image_path = None
        self.removal_image_label = ttk.Label(image_frame, text="No image selected")
        self.removal_image_label.pack(anchor=tk.W, pady=2)
        
        ttk.Button(
            image_frame,
            text="Select Image to Process",
            command=lambda: self.select_image_for_removal()
        ).pack(fill=tk.X, pady=5)
        
        
        options_frame = ttk.LabelFrame(parent, text="Removal Options", padding="10")
        options_frame.pack(fill=tk.X, pady=10)
        
        
        self.use_ai_model_var = tk.BooleanVar(value=False)
        ai_check = ttk.Checkbutton(
            options_frame,
            text="Use Advanced AI Model (Better Quality)",
            variable=self.use_ai_model_var
        )
        ai_check.pack(anchor=tk.W, pady=5)
        
        
        self.model_status_label = ttk.Label(
            options_frame,
            text="Status: Traditional method (works great!)",
            font=("Arial", 8),
            foreground="#888888"
        )
        self.model_status_label.pack(anchor=tk.W, pady=2)
        
        
        self.check_model_availability()
        
        
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
        
        
        self.removal_realistic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Enable Realistic Mode (Enhanced Skin Generation)",
            variable=self.removal_realistic_var
        ).pack(anchor=tk.W, pady=5)
        
        
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
    
    def update_removal_blend_label(self, parent, value):
        """Update blend factor label for removal tab"""
        val = float(value)
        for widget in parent.winfo_children():
            if isinstance(widget, ttk.Label) and widget.cget("text").startswith("0."):
                if val < 0.7:
                    label_text = f"{val:.2f} (Fast)"
                elif val < 0.85:
                    label_text = f"{val:.2f} (Good)"
                elif val < 0.95:
                    label_text = f"{val:.2f} (Excellent - Recommended)"
                else:
                    label_text = f"{val:.2f} (Maximum Realism)"
                widget.config(text=label_text)
    
    def check_model_availability(self):
        """Check if AI model is available"""
        try:
            from deepfake.utils.model_downloader import ModelDownloader
            downloader = ModelDownloader()
            model_path = downloader.get_model_path()
            
            if model_path:
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
    
    def select_image_for_removal(self):
        """Select image specifically for clothing removal"""
        file_path = filedialog.askopenfilename(
            title="Select Image for Clothing Removal",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
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
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
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
            messagebox.showinfo(
                "Success",
                f"Clothing removal complete!\n\nSaved to: {output_path}"
            )
    
    def _handle_removal_error(self, error):
        """Handle clothing removal errors"""
        error_msg = f"Clothing removal error: {error}"
        self.log_info(f"✗ {error_msg}")
        self.status_var.set("Error occurred")
        messagebox.showerror("Clothing Removal Error", error_msg)
