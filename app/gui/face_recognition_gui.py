import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from app.settings import load_dotenv_if_available


class FaceRecognitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("💡 Face ID - Smart Control System")
        self.root.geometry("620x600")
        self.root.configure(bg="#f5f5f5")

        self.smart_match = tk.BooleanVar(value=True)

        # Loaded on first use so the window opens immediately
        self.service = None

        self.image_path = None
        self.tk_image = None

        self._setup_ui()

    def _lazy_init(self):
        loading = tk.Toplevel(self.root)
        loading.title("Please wait...")
        tk.Label(loading, text="Loading models...").pack(padx=20, pady=20)
        self.root.update()

        # Same pipeline as the HTTP API, so GUI and API enrollments are compatible
        from api.recognition_service import RecognitionService

        try:
            self.service = RecognitionService.from_settings()
        finally:
            self.root.after(100, loading.destroy)

    def _setup_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(frame, text="📁 Upload Image", command=self.upload_image).pack(
            pady=8
        )
        self.image_label = ttk.Label(frame)
        self.image_label.pack(pady=5)

        self.result_label = tk.Label(frame, text="", font=("Helvetica", 14, "bold"))
        self.result_label.pack(pady=10)

        self.top_matches_label = tk.Label(frame, text="", font=("Helvetica", 12))
        self.top_matches_label.pack(pady=5)

        self.match_mode = ttk.Checkbutton(
            frame,
            text="Enable Smart Matching (Top-2 Margin)",
            variable=self.smart_match,
        )
        self.match_mode.pack(pady=5)

        ttk.Button(frame, text="🔍 Recognize Face", command=self.recognize).pack(pady=8)
        ttk.Button(frame, text="🔁 Try Another Image", command=self.reset_ui).pack(
            pady=5
        )
        ttk.Button(frame, text="➕ Add Person to DB", command=self.add_new_person).pack(
            pady=8
        )

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            image.thumbnail((300, 300))
            self.tk_image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.tk_image)
            self.result_label.configure(text="")
            self.top_matches_label.configure(text="")

    def recognize(self):
        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        if self.service is None:
            self._lazy_init()

        original_img = cv2.imread(self.image_path)
        if original_img is None:
            messagebox.showerror("Error", "Could not read the selected image.")
            return

        # The checkbox toggles the top-2 margin rule in FaceRecognizer
        self.service.recognizer.ambiguity_penalty = (
            0.1 if self.smart_match.get() else 0.0
        )

        result = self.service.recognize_image(original_img)
        if result.faces_detected == 0:
            self.result_label.configure(text="No face detected.", foreground="red")
            return

        rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((300, 300))
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=self.tk_image)

        name, score, top_scores = result.name, result.score, result.top_matches

        if name == "Unknown":
            self.result_label.configure(
                text=f"Match: {name} (Score: {round(score, 4)})", foreground="#d35400"
            )
        else:
            self.result_label.configure(
                text=f"Match: {name} (Score: {round(score, 4)})", foreground="#27ae60"
            )

        # Show Top-3 Matches in GUI
        top_matches_text = "\nTop Matches:\n"
        for label, s in top_scores:
            top_matches_text += f"• {label}: {round(s, 4)}\n"
        self.top_matches_label.configure(text=top_matches_text)

    def reset_ui(self):
        self.image_label.configure(image="")
        self.image_path = None
        self.result_label.configure(text="")
        self.top_matches_label.configure(text="")

    def add_new_person(self):
        from api.utils import normalize_person_name

        name = simpledialog.askstring("Add Person", "Enter name:")
        if not name:
            return
        try:
            name = normalize_person_name(name)
        except ValueError as exc:
            messagebox.showerror("Error", str(exc))
            return

        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Error", "Could not read the selected image.")
            return

        if self.service is None:
            self._lazy_init()

        result = self.service.enroll(image, name, image_path=file_path)
        if not result.success:
            messagebox.showerror("Error", f"Could not add {name}: {result.reason}")
            return
        messagebox.showinfo("Success", f"{name} has been added to the database!")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    load_dotenv_if_available()
    app = FaceRecognitionGUI()
    app.run()
