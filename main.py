import os
import pickle
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from deepface import DeepFace

#Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = 'models'  # folder where models are saved
EMB_DB_PATH = './embeddings_db.pkl'
DIST_THRESHOLD = 0.8  # euclidean threshold for identity match 

# Face detector & aligner
mtcnn = MTCNN(image_size=64, margin=0, keep_all=False, device=DEVICE)

#Model
embed_net = torch.load(os.path.join(CKPT_DIR, 'embed_net_full.pt'), map_location=DEVICE)
embed_net.eval()


val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
#Module for liveness detection and antispoofing
def is_live_face(pil_face, thresh=100.0):
    gray = cv2.cvtColor(np.array(pil_face), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var() >= thresh

def detect_emotion(pil_face):
    img = np.array(pil_face)
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    return result['dominant_emotion']

#Embedding DB
if os.path.exists(EMB_DB_PATH):
    with open(EMB_DB_PATH, 'rb') as f:
        emb_db = pickle.load(f)
else:
    emb_db = {} 

class FaceApp:
    def __init__(self, master):
        self.master = master
        master.title('Face Attendance System')
        self.label = tk.Label(master, text='Select an image')
        self.label.pack()
        self.img_panel = tk.Label(master)
        self.img_panel.pack()
        self.process_btn = tk.Button(master, text='Load Image', command=self.load_image)
        self.process_btn.pack()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg *.jpeg *.png')])
        if not path:
            return
        # Show image
        img = Image.open(path).resize((256,256))
        self.img_panel.img = ImageTk.PhotoImage(img)
        self.img_panel.config(image=self.img_panel.img)
        # Process
        self.process_face(path)

    def process_face(self, img_path):
        # 1) detect & crop
        orig = cv2.imread(img_path)
        face_tensor = mtcnn(orig)
        if face_tensor is None:
            messagebox.showerror('Error', 'No face detected')
            return
        pil_face = transforms.ToPILImage()((face_tensor * 255).byte())
        # 2) liveness
        live = is_live_face(pil_face)
        if not live:
            messagebox.showwarning('Spoof', 'Spoof detected!')
            return
        # 3) embedding
        emb = embed_net(val_transform(pil_face).unsqueeze(0).to(DEVICE)).cpu().detach().numpy()[0]
        # 4) identify or register
        if emb_db:
            names = list(emb_db.keys())
            embs = np.vstack([emb_db[n] for n in names])
            dists = np.linalg.norm(embs - emb, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < DIST_THRESHOLD:
                identity = names[idx]
            else:
                identity = self.register_new(emb)
        else:
            identity = self.register_new(emb)
        # 5) emotion
        emotion = detect_emotion(pil_face)
        messagebox.showinfo('Result', f'Identity: {identity}\nEmotion: {emotion}')

    def register_new(self, emb):
        name = simpledialog.askstring('Register', 'Enter name for new identity:')
        if not name:
            return 'Unknown'
        emb_db[name] = emb
        with open(EMB_DB_PATH, 'wb') as f:
            pickle.dump(emb_db, f)
        return name

if __name__ == '__main__':
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
