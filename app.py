import os
import uuid
import glob
import csv
import subprocess
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Response, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from tqdm import tqdm
import imageio_ffmpeg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

MODEL_PATH = "new.onnx"

# ── Class names from your training notebook (8-class vehicle model) ──────────
CLASS_NAMES = {
    0: 'auto',
    1: 'bus',
    2: 'car',
    3: 'lcv',
    4: 'motorcycle',
    5: 'multiaxle',
    6: 'tractor',
    7: 'truck',
}

# Distinct BGR colours per class for bounding boxes
CLASS_COLORS = {
    0: (0,   165, 255),   # auto       → orange
    1: (255,   0,   0),   # bus        → blue
    2: (0,   255,   0),   # car        → green
    3: (128,   0, 128),   # lcv        → purple
    4: (0,   255, 255),   # motorcycle → yellow
    5: (0,   128, 255),   # multiaxle  → light-blue
    6: (255, 255,   0),   # tractor    → cyan
    7: (0,     0, 255),   # truck      → red
}

try:
    model = YOLO(MODEL_PATH, task='detect')
    # Inject class names so r.plot() always has them,
    # even when ONNX metadata is missing
    if not model.names or len(model.names) == 0:
        model.names = CLASS_NAMES
    print(f"✅ Model loaded. Classes: {model.names}")
except Exception as e:
    model = None
    print(f"⚠️  Could not load model ({MODEL_PATH}): {e}")


# ── Helper: draw boxes manually (fallback / guaranteed path) ─────────────────
def draw_boxes(image: np.ndarray, boxes, names: dict, orig_dims=None, infer_size=416) -> np.ndarray:
    """Draw bounding boxes + labels directly on a copy of *image*.

    Works regardless of whether r.plot() is functioning correctly,
    because it reads raw xyxy tensors and paints them with OpenCV.
    """
    img = image.copy()
    h, w = img.shape[:2]

    for box in boxes:
        cls_id  = int(box.cls[0].item())
        conf    = float(box.conf[0].item())
        
        raw_xyxy = box.xyxy[0].tolist()
        if orig_dims:
            orig_w, orig_h = orig_dims
            # Use raw_xyxy values to reconstruct size.
            # If coordinates are extremely small (normalized), scale relative to original.
            if max(raw_xyxy) <= 2.0:
                scaled_xyxy = [raw_xyxy[0]*orig_w, raw_xyxy[1]*orig_h, raw_xyxy[2]*orig_w, raw_xyxy[3]*orig_h]
            else:
                scaled_xyxy = [raw_xyxy[0]*orig_w/infer_size, raw_xyxy[1]*orig_h/infer_size, raw_xyxy[2]*orig_w/infer_size, raw_xyxy[3]*orig_h/infer_size]
            x1, y1, x2, y2 = [int(v) for v in scaled_xyxy]
        else:
            x1, y1, x2, y2 = [int(v) for v in raw_xyxy]

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color     = CLASS_COLORS.get(cls_id, (0, 255, 0))
        label     = f"{names.get(cls_id, str(cls_id))} {conf:.2f}"
        thickness = max(2, int(min(w, h) / 300))   # scale with image size
        font_scale = max(0.4, min(w, h) / 1000)

        # Rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(
            img,
            (x1, label_y1),
            (x1 + tw + 4, label_y1 + th + baseline + 4),
            color, -1
        )

        # Label text (white)
        cv2.putText(
            img, label,
            (x1 + 2, label_y1 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA
        )

    return img


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/static/results/<path:filename>')
def serve_result(filename):
    """Serve result files with HTTP range request support.

    Chrome always sends a Range header when loading <video> elements.
    Without a 206 Partial Content response the video element stalls and
    never renders.  Flask's built-in static handler does NOT support range
    requests, so we handle them here.
    """
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)

    # Determine MIME type
    ext = filename.rsplit('.', 1)[-1].lower()
    mime_map = {'mp4': 'video/mp4', 'avi': 'video/x-msvideo',
                'mov': 'video/quicktime', 'mkv': 'video/x-matroska',
                'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png',
                'csv': 'text/csv'}
    mime_type = mime_map.get(ext, 'application/octet-stream')

    if not range_header:
        # Non-range request — send the whole file
        return send_file(file_path, mimetype=mime_type)

    # Parse the Range header (e.g. "bytes=0-1023")
    byte_range = range_header.replace('bytes=', '').strip()
    parts = byte_range.split('-')
    start = int(parts[0]) if parts[0] else 0
    end   = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
    end   = min(end, file_size - 1)
    length = end - start + 1

    def generate_chunk(path, offset, size, chunk=65536):
        with open(path, 'rb') as f:
            f.seek(offset)
            remaining = size
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    resp = Response(
        generate_chunk(file_path, start, length),
        status=206,
        mimetype=mime_type,
        direct_passthrough=True,
    )
    resp.headers['Content-Range']  = f'bytes {start}-{end}/{file_size}'
    resp.headers['Accept-Ranges']  = 'bytes'
    resp.headers['Content-Length'] = str(length)
    resp.headers['Cache-Control']  = 'no-cache'
    return resp


@app.route('/upload', methods=['POST'])
def upload():
    if not model:
        return jsonify({'error': f'Model not found. Place {MODEL_PATH} in the root directory.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    ext      = file.filename.rsplit('.', 1)[-1].lower()
    run_id   = uuid.uuid4().hex
    filename = f"{run_id}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    project_path = os.path.abspath(app.config['RESULT_FOLDER'])
    out_dir      = os.path.join(project_path, run_id)
    os.makedirs(out_dir, exist_ok=True)

    csv_path  = os.path.join(out_dir, "detections.csv")
    is_video  = ext in ['mp4', 'avi', 'mov', 'mkv']

    # Use the model's name dict (patched at startup if needed)
    names = model.names or CLASS_NAMES

    try:
        
        with open(csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['frame_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
            
            if is_video:
                cap    = cv2.VideoCapture(filepath)
                fps    = cap.get(cv2.CAP_PROP_FPS)
                fps    = fps if fps and fps == fps else 25.0   # guard NaN/0
                orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
                out_media = os.path.join(out_dir, f"{run_id}_yolo.mp4")
                fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
                vid_writer = cv2.VideoWriter(out_media, fourcc, int(fps), (orig_w, orig_h))
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_id = 0
                with tqdm(total=total_frames, desc="Inferencing video") as pbar:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # Squish to bypass YOLO's internal letterbox coordinate logic
                        frame_resized = cv2.resize(frame, (416, 416))
                        r = model.predict(source=frame_resized, imgsz=416, verbose=False)[0]
                        
                        annotated = draw_boxes(frame, r.boxes, names, orig_dims=(orig_w, orig_h), infer_size=416)
                        vid_writer.write(annotated)
                        
                        for box in r.boxes:
                            cls_id   = int(box.cls[0].item())
                            cls_name = names.get(cls_id, str(cls_id))
                            conf     = float(box.conf[0].item())
                            
                            raw_xyxy = box.xyxy[0].tolist()
                            if max(raw_xyxy) <= 2.0:
                                scaled_xyxy = [raw_xyxy[0]*orig_w, raw_xyxy[1]*orig_h, raw_xyxy[2]*orig_w, raw_xyxy[3]*orig_h]
                            else:
                                scaled_xyxy = [raw_xyxy[0]*orig_w/416.0, raw_xyxy[1]*orig_h/416.0, raw_xyxy[2]*orig_w/416.0, raw_xyxy[3]*orig_h/416.0]
                            x1, y1, x2, y2 = [int(max(0, v)) for v in scaled_xyxy]
                            csv_writer.writerow(
                                [frame_id, cls_name, f"{conf:.2f}", x1, y1, x2, y2]
                            )
                        frame_id += 1
                        pbar.update(1)
                cap.release()
                vid_writer.release()
            else:
                out_media  = os.path.join(out_dir, f"{run_id}.jpg")
                frame = cv2.imread(filepath)
                orig_h, orig_w = frame.shape[:2]
                
                frame_resized = cv2.resize(frame, (416, 416))
                r = model.predict(source=frame_resized, imgsz=416, verbose=False)[0]
                
                annotated = draw_boxes(frame, r.boxes, names, orig_dims=(orig_w, orig_h), infer_size=416)
                cv2.imwrite(out_media, annotated)
                
                for box in r.boxes:
                    cls_id   = int(box.cls[0].item())
                    cls_name = names.get(cls_id, str(cls_id))
                    conf     = float(box.conf[0].item())
                    
                    raw_xyxy = box.xyxy[0].tolist()
                    if max(raw_xyxy) <= 2.0:
                        scaled_xyxy = [raw_xyxy[0]*orig_w, raw_xyxy[1]*orig_h, raw_xyxy[2]*orig_w, raw_xyxy[3]*orig_h]
                    else:
                        scaled_xyxy = [raw_xyxy[0]*orig_w/416.0, raw_xyxy[1]*orig_h/416.0, raw_xyxy[2]*orig_w/416.0, raw_xyxy[3]*orig_h/416.0]
                    x1, y1, x2, y2 = [int(max(0, v)) for v in scaled_xyxy]
                    csv_writer.writerow(
                        [0, cls_name, f"{conf:.2f}", x1, y1, x2, y2]
                    )

    except Exception as e:
        return jsonify({'error': f'Inference error: {str(e)}'}), 500

    # ── Re-encode video to browser-compatible H.264 ───────────────────────────
    if is_video:
        temp_media = out_media + ".temp.mp4"
        final_mp4  = os.path.join(out_dir, f"{run_id}.mp4")
        os.rename(out_media, temp_media)
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', temp_media,
            '-vcodec', 'libx264', '-crf', '28', '-preset', 'fast',
            '-pix_fmt', 'yuv420p',   # required for browser compatibility
            final_mp4
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_media)
            saved_media = final_mp4
        except Exception as e:
            os.rename(temp_media, out_media)
            saved_media = out_media
            print(f"FFmpeg error (falling back to raw output): {e}")
    else:
        saved_media = out_media

    if not os.path.exists(saved_media):
        return jsonify({'error': 'Inference produced no output file.'}), 500

    result_file = os.path.basename(saved_media)
    file_type   = "video" if is_video else "image"

    return jsonify({
        'url':      f"/static/results/{run_id}/{result_file}",
        'csv_url':  f"/static/results/{run_id}/detections.csv",
        'type':     file_type,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)