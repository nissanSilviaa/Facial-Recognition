import os
import pickle
import numpy as np
import torch
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from deepface import DeepFace

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = 'models'
EMB_DB_PATH = 'embeddings_db.pkl'
DIST_THRESHOLD = 5            # threshold for 1-NN identity match

embed_net = torch.load(os.path.join(CKPT_DIR, 'Josh_Face_Net.pt'), map_location=DEVICE)
embed_net.eval()

#Database
if os.path.exists(EMB_DB_PATH):
    with open(EMB_DB_PATH, 'rb') as f:
        emb_db = pickle.load(f)
else:
    emb_db = {}

def is_live_face(pil_face, thresh=100.0):
    gray = cv2.cvtColor(np.array(pil_face), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var() >= thresh




def detect_emotion(pil_face):
    img = np.array(pil_face)
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    # DeepFace may return a list for batch results
    if isinstance(result, list):
        result = result[0]
    return result['dominant_emotion']

class FaceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Face Attendance System')
        self.label = tk.Label(self, text='Select an image')
        self.label.pack(pady=10)
        self.img_panel = tk.Label(self)
        self.img_panel.pack(pady=10)
        self.process_btn = tk.Button(self, text='Load Image', command=self.load_image)
        self.process_btn.pack(pady=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image files','*.jpg *.jpeg *.png')])
        if not path:
            return
        img = Image.open(path).resize((64,64))
        self.img_panel.img = ImageTk.PhotoImage(img)
        self.img_panel.config(image=self.img_panel.img)
        self.process_face(path)

    def process_face(self, img_path):
        # 1) Load and resize face
        pil_img_full = Image.open(img_path).convert('RGB')
        pil_face = pil_img_full.resize((64,64))

        # 2) Convert to tensor
        arr = np.array(pil_face)
        tensor = torch.tensor(arr, dtype=torch.float32, device=DEVICE)
        tensor = tensor.permute(2,0,1) / 255.0
        tensor = (tensor - 0.5) / 0.5
        face_tensor = tensor.unsqueeze(0)

        if not is_live_face(pil_face):
            messagebox.showwarning('Spoof', 'Spoof detected!')
            return

        # 4) Embedding & identification
        emb_tensor = embed_net(face_tensor).cpu().detach().squeeze(0)
        emb = emb_tensor.tolist()
        identity = None
        if emb_db:
            names = list(emb_db.keys())
            embs = np.vstack([emb_db[n] for n in names])
            dists = np.linalg.norm(embs - emb, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < DIST_THRESHOLD:
                identity = names[idx]
        if identity is None:
            identity = self.register_new(emb)

        # 5) Emotion detection
        emotion = detect_emotion(pil_face)
        print(dists[idx])
        messagebox.showinfo('Result', f'Identity: {identity},\nEmotion: {emotion}')

    def register_new(self, emb):
        name = simpledialog.askstring('Register','Enter name for new identity:')
        if not name:
            return 'Unknown'
        emb_db[name] = emb
        with open(EMB_DB_PATH,'wb') as f:
            pickle.dump(emb_db, f)
        return name

if __name__ == '__main__':
    app = FaceApp()
    app.mainloop()