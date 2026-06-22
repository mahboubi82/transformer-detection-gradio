import gradio as gr
import cv2
import numpy as np
import pandas as pd
import tempfile
import requests
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import io

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_NAME     = "Transformer"
CONF_DEFAULT   = 0.75
DEVICE         = "mps"   # → "cpu" sur serveur sans GPU

# ── Modeles disponibles ───────────────────────────────────────────────────────
MODELS = {
    "YOLO26m   — mAP@50: 0.943 (recommandé)"  : "runs/yolo26m/weights/best.pt",
    "RT-DETR-l — mAP@50: 0.924 (haute precision)" : "runs/rtdetr-l/weights/best.pt",
}

# ── Cache modeles ─────────────────────────────────────────────────────────────
_model_cache = {}

def get_model(model_label: str):
    path = MODELS[model_label]
    if path not in _model_cache:
        _model_cache[path] = YOLO(path)
    return _model_cache[path]

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(model, image_bgr: np.ndarray, conf: float):
    result = model.predict(source=image_bgr, conf=conf, device=DEVICE, verbose=False)[0]
    return result

def draw_boxes(image_bgr: np.ndarray, result) -> np.ndarray:
    img = image_bgr.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 120, 255), 2)
        label = f"{CLASS_NAME} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 120, 255), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img

def build_dataframe(result) -> pd.DataFrame:
    rows = []
    for i, box in enumerate(result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        rows.append({
            "ID": i + 1,
            "Confiance": f"{conf:.2%}",
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "Largeur (px)": x2 - x1,
            "Hauteur (px)": y2 - y1,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

# ── Fonction principale image ─────────────────────────────────────────────────
def predict_image(image: np.ndarray, model_label: str, conf: float):
    if image is None:
        return None, "Aucune image fournie.", pd.DataFrame()

    model     = get_model(model_label)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result    = run_inference(model, image_bgr, conf)

    n     = len(result.boxes)
    confs = [float(b.conf[0]) for b in result.boxes]
    avg   = np.mean(confs) if confs else 0.0
    best  = max(confs)     if confs else 0.0

    annotated_rgb = cv2.cvtColor(draw_boxes(image_bgr, result), cv2.COLOR_BGR2RGB)
    stats = f"Détectés : {n}  |  Confiance moy : {avg:.0%}  |  Meilleure : {best:.0%}"
    df    = build_dataframe(result)

    return annotated_rgb, stats, df

# ── Fonction URL ──────────────────────────────────────────────────────────────
def predict_url(url: str, model_label: str, conf: float):
    if not url:
        return None, "URL vide.", pd.DataFrame()
    try:
        response  = requests.get(url, timeout=10)
        response.raise_for_status()
        image_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
        image_np  = np.array(image_pil)
        return predict_image(image_np, model_label, conf)
    except Exception as e:
        return None, f"Erreur : {e}", pd.DataFrame()

# ── Fonction vidéo ────────────────────────────────────────────────────────────
def predict_video(video_path: str, model_label: str, conf: float):
    if not video_path:
        return None, "Aucune vidéo fournie.", pd.DataFrame()

    model = get_model(model_label)
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_detections = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 5 == 0:
            result = run_inference(model, frame, conf)
            frame  = draw_boxes(frame, result)
            for box in result.boxes:
                all_detections.append({
                    "Frame": frame_idx,
                    "Temps (s)": round(frame_idx / fps, 2),
                    "Confiance": f"{float(box.conf[0]):.2%}",
                })
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    df    = pd.DataFrame(all_detections) if all_detections else pd.DataFrame()
    stats = f"{len(all_detections)} détections sur {frame_idx} frames"
    return out_path, stats, df

# ── Interface Gradio ──────────────────────────────────────────────────────────
with gr.Blocks(title="Transformer Detection — NBPower", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🔌 Transformer Detection — NBPower")
    gr.Markdown("Détection automatique de transformateurs sur poteau | YOLOv26m & RT-DETR-l")

    # Paramètres communs
    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(MODELS.keys()),
            value=list(MODELS.keys())[0],
            label="Modèle",
        )
        conf_slider = gr.Slider(
            minimum=0.1, maximum=1.0, value=CONF_DEFAULT, step=0.05,
            label="Seuil de confiance",
            info="0.75 = optimal (F1 max) | 0.856 = zéro faux positif"
        )

    with gr.Tabs():

        # ── Tab Image ────────────────────────────────────────────────────────
        with gr.Tab("📷 Image"):
            with gr.Row():
                img_input  = gr.Image(label="Image source", type="numpy")
                img_output = gr.Image(label="Résultat")
            img_stats = gr.Textbox(label="Statistiques", interactive=False)
            img_table = gr.Dataframe(label="Détails", interactive=False)
            img_btn   = gr.Button("🔍 Analyser", variant="primary")

            img_btn.click(
                fn=predict_image,
                inputs=[img_input, model_dd, conf_slider],
                outputs=[img_output, img_stats, img_table],
            )

        # ── Tab Vidéo ────────────────────────────────────────────────────────
        with gr.Tab("🎬 Vidéo"):
            vid_input  = gr.Video(label="Vidéo source")
            vid_output = gr.Video(label="Résultat annoté")
            vid_stats  = gr.Textbox(label="Statistiques", interactive=False)
            vid_table  = gr.Dataframe(label="Détections par frame", interactive=False)
            vid_btn    = gr.Button("▶️ Analyser", variant="primary")

            vid_btn.click(
                fn=predict_video,
                inputs=[vid_input, model_dd, conf_slider],
                outputs=[vid_output, vid_stats, vid_table],
            )

        # ── Tab URL ──────────────────────────────────────────────────────────
        with gr.Tab("🌐 URL"):
            url_input  = gr.Textbox(label="URL de l'image", placeholder="https://example.com/image.jpg")
            url_output = gr.Image(label="Résultat")
            url_stats  = gr.Textbox(label="Statistiques", interactive=False)
            url_table  = gr.Dataframe(label="Détails", interactive=False)
            url_btn    = gr.Button("🔍 Analyser", variant="primary")

            url_btn.click(
                fn=predict_url,
                inputs=[url_input, model_dd, conf_slider],
                outputs=[url_output, url_stats, url_table],
            )

if __name__ == "__main__":
    demo.launch(share=False)
