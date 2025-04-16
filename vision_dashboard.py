import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import threading
import time
import os
from text_generation import TextGenerationAI
from vision_ai import VisionAI


class VisionDashboard:
    def __init__(self, root, text_ai=None, vision_ai=None):
        self.root = root
        self.text_ai = text_ai
        self.vision_ai = vision_ai if vision_ai else VisionAI(text_ai=text_ai)

        self.root.title("AI Vision Dashboard")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.setup_ui()

        self.running = False
        self.camera_thread = None

    def setup_ui(self):
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.status_var = tk.StringVar(value="AI Vision System Ready")
        self.status_label = ttk.Label(self.top_frame, textvariable=self.status_var, font=("Helvetica", 12))
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.video_frame = ttk.LabelFrame(self.left_panel, text="Camera Feed")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.control_frame = ttk.Frame(self.left_panel)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)

        self.camera_controls = ttk.LabelFrame(self.control_frame, text="Camera Controls")
        self.camera_controls.pack(fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(self.camera_controls, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = ttk.Button(self.camera_controls, text="Stop Camera", command=self.stop_camera,
                                      state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_button = ttk.Button(self.camera_controls, text="Save Frame", command=self.save_frame)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.save_button.config(state=tk.DISABLED)

        self.mode_frame = ttk.LabelFrame(self.control_frame, text="Processing Mode")
        self.mode_frame.pack(fill=tk.X, padx=5, pady=5)

        self.mode_var = tk.StringVar(value="normal")
        modes = [
            ("Normal", "normal"),
            ("Grayscale", "grayscale"),
            ("Edges", "edges"),
            ("Blur", "blur"),
            ("Face Detection", "face_detection"),
            ("Text Detection", "text_detection")
        ]
        for i, (text, mode) in enumerate(modes):
            row, col = divmod(i, 3)
            radio = ttk.Radiobutton(self.mode_frame, text=text, value=mode, variable=self.mode_var,
                                    command=self.change_mode)
            radio.grid(row=row, column=col, sticky="w", padx=5, pady=2)

        self.text_frame = ttk.LabelFrame(self.right_panel, text="Text Generation")
        self.text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.text_controls = ttk.Frame(self.text_frame)
        self.text_controls.pack(fill=tk.X, padx=5, pady=5)

        self.describe_button = ttk.Button(self.text_controls, text="Generate Description",
                                          command=self.generate_description)
        self.describe_button.pack(side=tk.LEFT, padx=5)
        self.describe_button.config(state=tk.DISABLED)

        ttk.Label(self.text_controls, text="Temperature:").pack(side=tk.LEFT, padx=(10, 0))
        self.temp_var = tk.DoubleVar(value=0.7)
        self.temp_slider = ttk.Scale(self.text_controls, from_=0.1, to=1.5, variable=self.temp_var,
                                     orient=tk.HORIZONTAL, length=150)
        self.temp_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.text_controls, textvariable=self.temp_var).pack(side=tk.LEFT)

        ttk.Label(self.text_controls, text="Length:").pack(side=tk.LEFT, padx=(10, 0))
        self.length_var = tk.IntVar(value=100)
        self.length_slider = ttk.Scale(self.text_controls, from_=20, to=300, variable=self.length_var,
                                       orient=tk.HORIZONTAL, length=150)
        self.length_slider.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.text_controls, textvariable=self.length_var).pack(side=tk.LEFT)

        self.text_output = tk.Text(self.text_frame, wrap=tk.WORD)
        self.text_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.scrollbar = ttk.Scrollbar(self.text_output, command=self.text_output.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_output.config(yscrollcommand=self.scrollbar.set)

        self.model_frame = ttk.LabelFrame(self.right_panel, text="Model Status")
        self.model_frame.pack(fill=tk.X, padx=5, pady=5)

        self.model_status = tk.StringVar(value="No model loaded")
        ttk.Label(self.model_frame, textvariable=self.model_status).pack(padx=5, pady=5)

        self.load_model_button = ttk.Button(self.model_frame, text="Load Text Model", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.current_frame = None

    def start_camera(self):
        if not self.running:
            self.running = True
            self.status_var.set("Starting camera...")

            self.camera_thread = threading.Thread(target=self.camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()

            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)

            if self.text_ai and self.text_ai.model is not None:
                self.describe_button.config(state=tk.NORMAL)

    def stop_camera(self):
        if self.running:
            self.running = False
            self.status_var.set("Stopping camera...")

            if self.camera_thread:
                self.camera_thread.join(timeout=1.0)

            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            self.describe_button.config(state=tk.DISABLED)

    def camera_loop(self):
        if not self.vision_ai.start_camera():
            self.status_var.set("Error: Failed to start camera")
            self.running = False
            return

        try:
            while self.running:
                ret, frame = self.vision_ai.camera.read()
                if not ret:
                    self.status_var.set("Error: Failed to capture frame")
                    break

                self.vision_ai.processing_mode = self.mode_var.get()
                processed_frame = self.vision_ai.process_frame(frame)

                if len(processed_frame.shape) == 2:
                    self.current_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                else:
                    self.current_frame = processed_frame.copy()

                if len(processed_frame.shape) == 2:
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
                else:
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                if canvas_width > 1 and canvas_height > 1:
                    h, w = display_frame.shape[:2]
                    aspect = w / h

                    if canvas_width / canvas_height > aspect:
                        new_h = canvas_height
                        new_w = int(aspect * new_h)
                    else:
                        new_w = canvas_width
                        new_h = int(new_w / aspect)

                    display_frame = cv2.resize(display_frame, (new_w, new_h))

                img = Image.fromarray(display_frame)
                img_tk = ImageTk.PhotoImage(image=img)

                self.canvas.config(width=img_tk.width(), height=img_tk.height())
                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.image = img_tk

                self.status_var.set(f"Processing mode: {self.vision_ai.processing_mode}")

                time.sleep(0.01)

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
        finally:
            self.vision_ai.stop_camera()
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def change_mode(self):
        self.vision_ai.processing_mode = self.mode_var.get()
        self.status_var.set(f"Changed mode to: {self.vision_ai.processing_mode}")

    def generate_description(self):
        if self.running and self.current_frame is not None and self.text_ai and self.text_ai.model is not None:
            self.text_output.insert(tk.END, "\n\n--- New Description ---\n")
            temperature = self.temp_var.get()
            length = self.length_var.get()

            self.text_output.insert(tk.END, f"[Temperature: {temperature:.2f}, Length: {length}]\n")

            description = self.vision_ai.describe_scene(self.current_frame)

            self.text_output.insert(tk.END, description)
            self.text_output.see(tk.END)
        else:
            if not self.text_ai or self.text_ai.model is None:
                self.status_var.set("Error: Text generation model not loaded")
            elif not self.running:
                self.status_var.set("Error: Camera not running")
            else:
                self.status_var.set("Error: No frame available")

    def save_frame(self):
        if self.current_frame is not None:
            filename = filedialog.asksaveasfilename(
                initialdir="./",
                title="Save Frame",
                filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")),
                defaultextension=".jpg"
            )

            if filename:
                cv2.imwrite(filename, self.current_frame)
                self.status_var.set(f"Saved frame as {filename}")
                self.text_output.insert(tk.END, f"\nSaved current frame as {filename}\n")
                self.text_output.see(tk.END)

    def load_model(self):
        """Let user choose a model directory"""
        if not os.path.exists("models"):
            self.status_var.set("Error: No 'models' directory found")
            return

        models = [d for d in os.listdir("models") if os.path.isdir(os.path.join("models", d))]

        if not models:
            self.status_var.set("Error: No models found in 'models' directory")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Select Model")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select a model to load:").pack(padx=10, pady=10)

        model_listbox = tk.Listbox(dialog)
        model_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        for model in models:
            model_listbox.insert(tk.END, model)

        def on_select():
            selection = model_listbox.curselection()
            if selection:
                selected_model = model_listbox.get(selection[0])
                dialog.destroy()
                self._load_selected_model(selected_model)
            else:
                dialog.destroy()

        ttk.Button(dialog, text="Load Selected Model", command=on_select).pack(pady=10)

    def _load_selected_model(self, model_name):
        """Internal function to load the selected model"""
        self.status_var.set(f"Loading model: {model_name}...")

        if not self.text_ai:
            self.text_ai = TextGenerationAI(model_name=model_name)
            self.vision_ai.text_ai = self.text_ai
        else:
            self.text_ai.model_name = model_name

        if self.text_ai.load():
            self.model_status.set(f"Model loaded: {model_name}")
            self.status_var.set(f"Successfully loaded model: {model_name}")

            if self.running:
                self.describe_button.config(state=tk.NORMAL)
        else:
            self.model_status.set(f"Failed to load model: {model_name}")
            self.status_var.set(f"Error: Failed to load model: {model_name}")

    def on_close(self):
        self.stop_camera()
        self.root.destroy()


def run_dashboard(model_name=None):
    text_ai = None
    if model_name:
        text_ai = TextGenerationAI(model_name=model_name)
        if text_ai.load():
            print(f"Successfully loaded model: {model_name}")
        else:
            print(f"Warning: Failed to load model: {model_name}")

    vision_ai = VisionAI(text_ai=text_ai)
    root = tk.Tk()
    dashboard = VisionDashboard(root, text_ai=text_ai, vision_ai=vision_ai)
    root.mainloop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run AI Vision Dashboard')
    parser.add_argument('--model', type=str, help='Model name to use (e.g., keras_rnn_model)')
    args = parser.parse_args()

    run_dashboard(model_name=args.model)