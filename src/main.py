import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model, save_model
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, Toplevel, filedialog
from ttkbootstrap import Style
from PIL import Image, ImageTk
import os
import random
import threading
import sys
import traceback
import datetime
import time
import albumentations as A

# Configurações globais
DATA_DIR = "data"
MODEL_DIR = "models"
LOGS_DIR = "logs"
tamanho_img = (128, 128)  # Reduzido para melhorar desempenho
modelo_path = os.path.join(MODEL_DIR, "modelo.keras")
log_path = os.path.join(LOGS_DIR, "recognition_log.txt")
MINIATURA_TAMANHO = (56, 56)
MAX_MINIATURAS = 5
CONFIDENCE_THRESHOLD = 0.5

# Criar pastas se não existirem
for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Carrega o modelo YOLO
try:
    model_yolo = YOLO(os.path.join(MODEL_DIR, "yolov8n.pt"))  # Modelo leve
    print("Modelo YOLO carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar modelo YOLO: {e}")
    sys.exit(1)

# Data augmentation ajustado (sem flips para evitar inversões)
transform = A.Compose([
    A.RandomRotate90(p=0.5),  # Mantém apenas rotações de 90 graus
    A.RandomBrightnessContrast(p=0.5),  # Ajuste de brilho e contraste
    A.Affine(translate_percent=0.1, scale=0.1, rotate=15, p=0.5),  # Deslocamento, escala e rotação
])

class TeachableMachineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TreinadorVisaoDeMaquina")
        self.root.geometry("1100x800")

        self.classes = []
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            messagebox.showerror("Erro", "Não foi possível inicializar a webcam!")
            self.root.destroy()
            return
        self.modelo = None
        self.captura_continua_ativa = {}
        self.captura_threads = {}
        self.preview_active = False
        self.class_previews = {}
        self.active_class = None
        self.class_confidences = {}
        self.log_file = open(log_path, "a")
        self.preview_running = False

        self.style = Style(theme="flatly")
        self.setup_ui()
        self.load_classes_from_folder()
        self.update_class_previews()  # Inicia o preview nas classes

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10", style="TFrame")
        main_frame.pack(fill="both", expand=True)

        classes_frame = ttk.LabelFrame(main_frame, text="Classes", padding="10", style="TLabelframe")
        classes_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.classes_canvas = tk.Canvas(classes_frame, height=500, bg="#f5f5f5")
        scrollbar = ttk.Scrollbar(classes_frame, orient="vertical", command=self.classes_canvas.yview)
        self.classes_inner_frame = ttk.Frame(self.classes_canvas, style="TFrame")
        self.classes_inner_frame.bind("<Configure>", lambda e: self.classes_canvas.configure(scrollregion=self.classes_canvas.bbox("all")))
        self.classes_canvas.create_window((0, 0), window=self.classes_inner_frame, anchor="nw")
        self.classes_canvas.configure(yscrollcommand=scrollbar.set)
        self.classes_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y")

        ttk.Button(self.classes_inner_frame, text="Add a Class", command=self.add_class, style="TButton").pack(fill="x", pady=10, padx=10)

        training_frame = ttk.LabelFrame(main_frame, text="Training", padding="10", style="TLabelframe")
        training_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        ttk.Button(training_frame, text="Train Model", command=self.train_model, style="TButton").pack(pady=10)

        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10", style="TLabelframe")
        preview_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.preview_label = ttk.Label(preview_frame, style="TLabel")
        self.preview_label.pack(fill="both", expand=True, pady=10)

        self.progress_frame = ttk.Frame(preview_frame, style="TFrame")
        self.progress_frame.pack(fill="x", pady=10)
        ttk.Button(preview_frame, text="Preview", command=self.toggle_preview, style="TButton").pack(side="left", padx=5)
        ttk.Button(preview_frame, text="Export Model", command=self.export_model, style="TButton").pack(side="left", padx=5)

    def load_classes_from_folder(self):
        for class_name in os.listdir(DATA_DIR):
            class_pasta = os.path.join(DATA_DIR, class_name)
            if os.path.isdir(class_pasta):
                self.add_class_from_folder(class_name)

    def add_class_from_folder(self, class_name):
        class_frame = ttk.LabelFrame(self.classes_inner_frame, text=class_name, padding="10", style="TLabelframe")
        class_frame.pack(fill="x", pady=5, padx=10)
        preview_label = ttk.Label(class_frame, style="TLabel")
        preview_label.pack(fill="both", pady=5)
        self.class_previews[class_name] = preview_label
        self.active_class = class_name

        control_frame = ttk.Frame(class_frame, style="TFrame")
        control_frame.pack(fill="x", pady=5)
        ttk.Button(control_frame, text="📷 Webcam", command=lambda: self.start_continuous_capture(class_name), style="TButton").pack(side="left", padx=5)
        ttk.Button(control_frame, text="⬆️ Upload", command=lambda: self.upload_image(class_name), style="TButton").pack(side="left", padx=5)
        ttk.Button(control_frame, text="⏹ Stop", command=lambda: self.stop_continuous_capture(class_name), style="TButton").pack(side="left", padx=5)
        ttk.Button(control_frame, text="🗑 Excluir", command=lambda: self.delete_class(class_name), style="TButton").pack(side="left", padx=5)

        image_count_label = ttk.Label(control_frame, text="Imagens: 0", style="TLabel")
        image_count_label.pack(side="left", padx=5)
        self.update_image_count(class_name, image_count_label)

        thumbs_frame = ttk.Frame(class_frame, width=280, height=56, style="TFrame")
        thumbs_frame.pack(fill="x", pady=5)
        ttk.Label(thumbs_frame, text=f"{class_name} Image Samples", style="TLabel").pack()

        self.classes.append({"name": class_name, "frame": class_frame, "thumbs_frame": thumbs_frame, "image_count_label": image_count_label})
        self.update_thumbnails(class_name)
        self.update_progress_bars()

    def update_image_count(self, class_name, label):
        try:
            class_pasta = os.path.join(DATA_DIR, class_name)
            if os.path.exists(class_pasta) and label.winfo_exists():  # Verifica se o widget ainda existe
                image_count = len([f for f in os.listdir(class_pasta) if f.endswith((".jpg", ".png"))])
                label.config(text=f"Imagens: {image_count}")
            self.root.after(1000, lambda: self.update_image_count(class_name, label))
        except Exception as e:
            print(f"Erro ao atualizar contagem de imagens para {class_name}: {e}")

    def update_thumbnails(self, class_name):
        class_pasta = os.path.join(DATA_DIR, class_name)
        thumbs_frame = next(c["thumbs_frame"] for c in self.classes if c["name"] == class_name)
        for widget in thumbs_frame.winfo_children()[1:]:
            widget.destroy()

        fotos = [f for f in os.listdir(class_pasta) if f.endswith((".jpg", ".png"))][:MAX_MINIATURAS]
        for foto in fotos:
            caminho = os.path.join(class_pasta, foto)
            img = cv2.imread(caminho)
            if img is not None:
                img = cv2.resize(img, MINIATURA_TAMANHO)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                thumb_label = ttk.Label(thumbs_frame, image=img_tk, style="TLabel")
                thumb_label.image = img_tk
                thumb_label.pack(side="left", padx=5, pady=5)

    def update_class_previews(self):
        try:
            if not self.webcam or not self.webcam.isOpened():
                return

            ret, frame = self.webcam.read()
            if not ret or frame is None or frame.size == 0:  # Verifica frames inválidos (pretos ou corrompidos)
                self.root.after(50, self.update_class_previews)
                return

            frame = cv2.resize(frame, (160, 120))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            if self.active_class and self.active_class in self.class_previews:
                self.class_previews[self.active_class].configure(image=img_tk)
                self.class_previews[self.active_class].image = img_tk

            self.root.after(50, self.update_class_previews)
        except Exception as e:
            print(f"Erro na atualização dos previews das classes: {e}")
            self.root.after(50, self.update_class_previews)

    def update_progress_bars(self):
        try:
            # Limpa completamente o frame de progresso
            for widget in self.progress_frame.winfo_children():
                widget.destroy()
            self.class_confidences.clear()  # Limpa referências antigas

            colors = ["#FF9800", "#F48FB1", "#9C27B0", "#4CAF50", "#2196F3"]  # Laranja, Rosa, Roxo, Verde, Azul
            for class_info in self.classes:
                class_name = class_info["name"]
                if class_name not in self.class_confidences:
                    color_index = len(self.class_confidences) % len(colors)
                    progress_style = f"Progressbar{color_index}.Horizontal.TProgressbar"
                    self.style.configure(progress_style, troughcolor="#e0e0e0", background=colors[color_index], thickness=10)
                    progress = ttk.Progressbar(self.progress_frame, length=200, style=progress_style, maximum=100)
                    progress.pack(side="left", pady=2, padx=5)
                    label = ttk.Label(self.progress_frame, text=f"{class_name} 0%", style="TLabel", background="#ffffff", foreground="#333333")
                    label.pack(side="left", padx=5)
                    self.class_confidences[class_name] = (progress, label)
                else:
                    self.class_confidences[class_name][0].pack(side="left", pady=2, padx=5)
                    self.class_confidences[class_name][0]['value'] = 0  # Zera a barra por padrão
                    self.class_confidences[class_name][1].pack(side="left", padx=5)
                    self.class_confidences[class_name][1].config(text=f"{class_name} 0%")
        except Exception as e:
            print(f"Erro ao atualizar barras de progresso: {e}")

    def start_continuous_capture(self, class_name):
        if class_name not in self.captura_continua_ativa or not self.captura_continua_ativa[class_name]:
            self.captura_continua_ativa[class_name] = True
            self.captura_threads[class_name] = threading.Thread(target=self.continuous_capture, args=(class_name,))
            self.captura_threads[class_name].daemon = True
            self.captura_threads[class_name].start()

    def stop_continuous_capture(self, class_name):
        if class_name in self.captura_continua_ativa:
            self.captura_continua_ativa[class_name] = False

    def continuous_capture(self, class_name):
        class_pasta = os.path.join(DATA_DIR, class_name)
        contador_fotos = len(os.listdir(class_pasta))
        while self.captura_continua_ativa.get(class_name, False):
            ret, frame = self.webcam.read()
            if not ret or frame is None or frame.size == 0:  # Verifica frames inválidos
                time.sleep(0.5)  # Aumenta o intervalo para maior estabilidade
                continue
            frame = cv2.resize(frame, tamanho_img)
            # Remove flips para evitar inversões, mantendo apenas ajustes básicos
            augmented = transform(image=frame)["image"]
            caminho_foto = os.path.join(class_pasta, f"{class_name}_{contador_fotos}.jpg")
            cv2.imwrite(caminho_foto, augmented)
            contador_fotos += 1
            self.update_thumbnails(class_name)
            time.sleep(0.5)  # Aumenta o intervalo para maior estabilidade

    def upload_image(self, class_name):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            class_pasta = os.path.join(DATA_DIR, class_name)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, tamanho_img)
                # Remove flips para evitar inversões, mantendo apenas ajustes básicos
                augmented = transform(image=img)["image"]
                contador_fotos = len(os.listdir(class_pasta))
                novo_nome = os.path.join(class_pasta, f"{class_name}_{contador_fotos}.jpg")
                cv2.imwrite(novo_nome, augmented)
                self.update_thumbnails(class_name)

    def train_model(self):
        # Exibe mensagem de espera durante o treinamento
        loading_window = Toplevel(self.root)
        loading_window.title("Treinando Modelo")
        loading_window.geometry("300x100")
        loading_label = ttk.Label(loading_window, text="Treinando modelo... Aguarde...", padding=10, style="TLabel")
        loading_label.pack(expand=True)
        self.root.update()

        imagens = []
        rotulos = []
        categorias = [c["name"] for c in self.classes]
        for class_info in self.classes:
            categoria = class_info["name"]
            pasta_categoria = os.path.join(DATA_DIR, categoria)
            for foto in os.listdir(pasta_categoria):
                caminho = os.path.join(pasta_categoria, foto)
                img = cv2.imread(caminho)
                if img is not None:
                    img = cv2.resize(img, tamanho_img)
                    imagens.append(img)
                    rotulos.append(categorias.index(categoria))

        if not imagens:
            loading_window.destroy()
            messagebox.showerror("Erro", "Nenhuma imagem para treinamento!")
            return

        imagens = np.array(imagens, dtype="float32") / 255.0
        rotulos = np.array(rotulos)
        X_train, X_test, y_train, y_test = train_test_split(imagens, rotulos, test_size=0.2, random_state=42)

        modelo = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(categorias), activation='softmax')
        ])

        modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        modelo.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

        modelo.save(modelo_path, save_format="keras")  # Usa o formato nativo Keras
        self.modelo = modelo
        loading_window.destroy()
        messagebox.showinfo("Sucesso", "Modelo treinado e salvo!")

    def toggle_preview(self):
        self.preview_active = not self.preview_active
        if self.preview_active:
            self.preview_running = True
            threading.Thread(target=self.preview_worker, daemon=True).start()
        else:
            self.preview_running = False

    def preview_worker(self):
        frame_count = 0
        while self.preview_running:
            frame_count += 1
            if frame_count % 3 == 0:
                frame, _, confidences = self.preview_frame()
                if frame is None:
                    break
                self.root.after(0, lambda: self.update_preview_ui(frame, confidences))
            time.sleep(0.01)

    def preview_frame(self):
        if not self.preview_active:
            return None, [], {}
        ret, frame = self.webcam.read()
        if not ret or frame is None or frame.size == 0:
            return None, [], {}

        frame_red = cv2.resize(frame, (96, 72))
        resultados = model_yolo.predict(frame_red, classes=[0], verbose=False)
        objetos = []
        confidences = {c["name"]: 0.0 for c in self.classes}

        if self.modelo:
            img = cv2.resize(frame, tamanho_img)
            img = np.expand_dims(img, axis=0) / 255.0
            try:
                previsao = self.modelo.predict(img, verbose=0)[0]
                for i, conf in enumerate(previsao):
                    class_name = self.classes[i]["name"]
                    confidence_value = conf * 100
                    confidences[class_name] = confidence_value
                    if conf >= CONFIDENCE_THRESHOLD:
                        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]  # Coordenadas simplificadas
                        objetos.append((class_name, conf, (x1, y1, x2, y2)))
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = f"{timestamp}: Detectado {class_name} com confiança {conf:.2f} em ({x1}, {y1}, {x2}, {y2})\n"
                        self.log_file.write(log_entry)
                        self.log_file.flush()
            except Exception as e:
                print(f"Erro na previsão do modelo: {e}")
                return frame, [], {}

        return frame, objetos, confidences

    def update_preview_ui(self, frame, confidences):
        try:
            # Atualiza o preview da imagem
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            if hasattr(self.preview_label, 'img_tk'):
                del self.preview_label.img_tk
            self.preview_label.img_tk = img_tk
            self.preview_label.configure(image=img_tk)

            # Atualiza as barras de progresso
            for class_name, confidence in confidences.items():
                if class_name in self.class_confidences and self.class_confidences[class_name][0].winfo_exists() and self.class_confidences[class_name][1].winfo_exists():
                    self.class_confidences[class_name][0]['value'] = confidence
                    self.class_confidences[class_name][1].config(text=f"{class_name} {int(confidence)}%")
        except Exception as e:
            print(f"Erro na atualização da UI do preview: {e}")

    def export_model(self):
        if self.modelo:
            save_model(self.modelo, modelo_path, save_format="keras")  # Usa o formato nativo Keras
            messagebox.showinfo("Sucesso", f"Modelo exportado para {modelo_path}")

    def add_class(self):
        nome = simpledialog.askstring("Nome da Classe", "Digite o nome da classe (ex.: Controle, Fundo):", parent=self.root)
        if not nome or nome.strip() == "":
            messagebox.showerror("Erro", "Nome da classe não pode ser vazio!", parent=self.root)
            return
        class_pasta = os.path.join(DATA_DIR, nome.replace(" ", "_"))
        if not os.path.exists(class_pasta):
            os.makedirs(class_pasta)
        self.add_class_from_folder(nome)

    def delete_class(self, class_name):
        class_info = next((c for c in self.classes if c["name"] == class_name), None)
        if class_info:
            class_pasta = os.path.join(DATA_DIR, class_name.replace(" ", "_"))
            if os.path.exists(class_pasta):
                import shutil
                shutil.rmtree(class_pasta)
            if class_name in self.class_previews:
                del self.class_previews[class_name]
            class_info["frame"].destroy()
            self.classes.remove(class_info)
            if self.active_class == class_name:
                self.active_class = None
            if class_name in self.class_confidences:
                del self.class_confidences[class_name]
            self.update_progress_bars()

if __name__ == "__main__":
    root = tk.Tk()
    app = TeachableMachineApp(root)
    root.mainloop()