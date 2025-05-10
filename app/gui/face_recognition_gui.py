import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import os
import cv2
import numpy as np

from app.database.db_manager import EmbeddingDatabase


class FaceRecognitionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("💡 Face ID - Smart Control System")
        self.root.geometry("620x600")
        self.root.configure(bg="#f5f5f5")

        self.smart_match = tk.BooleanVar(value=True)

        self.db = None
        self.detector = None
        self.yolo = None
        self.embedder = None
        self.recognizer = None

        self.image_path = None
        self.tk_image = None

        self._setup_ui()

    def _lazy_init(self):
        loading = tk.Toplevel(self.root)
        loading.title("Please wait...")
        tk.Label(loading, text="Loading models...").pack(padx=20, pady=20)
        self.root.update()

        from app.detection.face_detector import FaceDetector
        from app.detection.yolo_detector import YOLOFaceDetector
        from app.embedding.face_embedder import FaceEmbedder
        from app.recognition.face_recognizer import FaceRecognizer

        self.db = EmbeddingDatabase()
        self.detector = FaceDetector()
        self.yolo = YOLOFaceDetector(confidence=0.2)
        self.embedder = FaceEmbedder()
        self.recognizer = FaceRecognizer(embedder=self.embedder, database=self.db)

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
        if self.recognizer is None:
            self._lazy_init()

        if not self.image_path:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        original_img = cv2.imread(self.image_path)
        face = self.detector.extract_face(original_img)
        if face is None:
            self.result_label.configure(text="No face detected.", foreground="red")
            return

        boxes = self.yolo.detect_faces(original_img)
        boxed_img = self.yolo.draw_boxes(original_img.copy(), boxes)

        rgb = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((300, 300))
        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=self.tk_image)

        name, score, top_scores = self.recognizer.recognize(face)

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

        self.db.log_recognition_event(name)

    def reset_ui(self):
        self.image_label.configure(image="")
        self.image_path = None
        self.result_label.configure(text="")
        self.top_matches_label.configure(text="")

    def add_new_person(self):
        if self.recognizer is None:
            self._lazy_init()

        name = simpledialog.askstring("Add Person", "Enter name:")
        if not name:
            return

        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        face = self.detector.extract_face(file_path)
        if face is None:
            messagebox.showerror("Error", "No face detected in the image.")
            return

        embedding = self.embedder.get_embedding(face)
        self.db.insert_embedding(name, embedding.tolist(), file_path)
        messagebox.showinfo("Success", f"{name} has been added to the database!")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = FaceRecognitionGUI()
    app.run()
