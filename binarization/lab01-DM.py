import tkinter as tk
from PIL import Image, ImageTk, ImageEnhance

try:
    resample_filter = Image.Resampling.LANCZOS  # Pillow >= 10.0.0
except AttributeError:
    resample_filter = Image.LANCZOS  # Pillow < 10.0.0


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lab01")
        self.image = None
        self.original_image = None

        main_frame = tk.Frame(root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        self.image_label = tk.Label(image_frame)
        self.image_label.pack()

        # Binary threshold
        self.threshold_scale = tk.Scale(
            controls_frame,
            from_=0,
            to=255,
            orient=tk.HORIZONTAL,
            command=self.update_image,
            length=300,
            sliderlength=20,
            label="Binary Threshold"
        )
        self.threshold_scale.set(128)  # 50%

        self.threshold_scale.pack(padx=5, pady=5)

        # Darken threshold
        self.darken_scale = tk.Scale(
            controls_frame,
            from_=1,
            to=99,
            orient=tk.HORIZONTAL,
            command=self.darken_image,
            length=300,
            sliderlength=20,
            label="Darken Percentage (%)"
        )
        self.darken_scale.set(10)
        self.darken_scale.pack(padx=5, pady=5)

        # Brighten threshold
        self.brighten_scale = tk.Scale(
            controls_frame,
            from_=10,
            to=20,
            orient=tk.HORIZONTAL,
            length=300,
            sliderlength=20,
            label="Brighten Step Percentage (%)"
        )
        self.brighten_scale.set(10)
        self.brighten_scale.pack(padx=5, pady=5)

        # Serial brighten button
        self.brighten_button = tk.Button(
            controls_frame,
            text="Brighten up in series",
            command=self.brighten_image
        )
        self.brighten_button.pack(padx=5, pady=5)

        # Save image button
        self.save_button = tk.Button(
            controls_frame,
            text="Save Image",
            command=self.save_image
        )
        self.save_button.pack(padx=5, pady=5)

        # Load image
        path = "sultan.png"
        self.load_image(path)
        self.update_image()

    def load_image(self, image_path):
        max_width = 500
        max_height = 500

        original_image = Image.open(image_path).convert("L")

        # Rescale image
        width_ratio = max_width / original_image.width
        height_ratio = max_height / original_image.height
        scaling_factor = min(width_ratio, height_ratio, 1)

        new_width = int(original_image.width * scaling_factor)
        new_height = int(original_image.height * scaling_factor)

        self.original_image = original_image.resize((new_width, new_height), resample_filter)
        self.image = self.original_image.copy()
        self.full_resolution_image = original_image  # Store the original resolution image

    # For binary threshold
    def update_image(self, event=None):
        if self.image:
            threshold = self.threshold_scale.get()
            # Convert to binary
            binary_image = self.image.point(lambda p: 255 if p > threshold else 0)
            binary_image = binary_image.convert("1")  # Konwersja do trybu binarnego
            self.tk_image = ImageTk.PhotoImage(binary_image)
            self.image_label.config(image=self.tk_image)
            self.image_label.image = self.tk_image

    def darken_image(self, event=None):
        if self.image:
            percentage = self.darken_scale.get()
            factor = 1 - (percentage / 100.0)
            enhancer = ImageEnhance.Brightness(self.original_image)
            darkened_image = enhancer.enhance(factor)
            self.image = darkened_image
            self.update_image()

    def brighten_image(self):
        if self.image:
            percentage = self.brighten_scale.get()
            step = percentage / 100.0
            images = []
            for i in range(0, 3):
                factor = 1 + (step * i)
                enhancer = ImageEnhance.Brightness(self.original_image)
                brightened_image = enhancer.enhance(factor)
                images.append(brightened_image)

            # Print images
            for idx, img in enumerate(images):
                window = tk.Toplevel(self.root)
                window.title(f"Brightened image {idx + 1}")

                # Rescale image
                tk_image = ImageTk.PhotoImage(img)
                label = tk.Label(window, image=tk_image)
                label.image = tk_image
                label.pack()

    def save_image(self):
        if self.image:
            threshold = self.threshold_scale.get()
            binary_image = self.full_resolution_image.point(lambda p: 255 if p > threshold else 0)
            binary_image = binary_image.convert("1")  # Konwersja do trybu binarnego
            binary_image.save("newSultan.png")
            print("Image saved as 'newSultan.png'")


# Main
root = tk.Tk()
app = ImageApp(root)
root.mainloop()
