#!/usr/bin/env python3
"""
pipeline_router.py

- Detects whether each media file has audio or not (video-only vs audio+video).
- Runs the appropriate processing steps (frame extraction, face cropping,
  audio extraction & preprocessing, encoder placeholders, fusion placeholders).

Usage:
    python pipeline_router.py /path/to/dataset /path/to/output_dir
Notes:
 - ffmpeg and ffprobe must be installed and on PATH.
 - Replace encoder / classifier stubs with real model code for production.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np

# Optional libs used in example implementations
try:
    import cv2
except Exception as e:
    raise ImportError("cv2 (opencv-python) is required. Install with `pip install opencv-python`.") from e

try:
    from mtcnn import MTCNN
except Exception as e:
    raise ImportError("mtcnn is required. Install with `pip install mtcnn`.") from e

try:
    import librosa
except Exception as e:
    raise ImportError("librosa is required. Install with `pip install librosa`.") from e

# -------------------------
# Sanity checks for ffmpeg/ffprobe
# -------------------------
def check_executables():
    from shutil import which
    if which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg not found on PATH. Install ffmpeg and make sure it's available.")
    if which("ffprobe") is None:
        raise EnvironmentError("ffprobe not found on PATH. Install ffmpeg (which provides ffprobe) and make sure it's available.")

check_executables()

# -------------------------
# Utility: FFPROBE / FFMPEG
# -------------------------
def has_audio_stream(video_path: str) -> bool:
    """Return True if file has at least one audio stream (uses ffprobe)."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_type",
        "-of", "json",
        str(video_path)
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        # subprocess returns bytes; decode to str
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="ignore")
        info = json.loads(out)
        streams = info.get("streams", [])
        for s in streams:
            if s.get("codec_type") == "audio":
                return True
        return False
    except subprocess.CalledProcessError as e:
        # e.output may be bytes
        err_out = e.output.decode("utf-8", errors="ignore") if isinstance(e.output, bytes) else str(e.output)
        print(f"[ffprobe error] {err_out[:400]}")
        # conservative choice: if ffprobe failed, assume file might have audio (so caller can handle errors)
        return False
    except json.JSONDecodeError as e:
        print(f"[ffprobe] could not parse ffprobe output for {video_path}: {e}")
        return False

def extract_audio(video_path: str, out_wav: str, sr: int = 16000):
    """Extract audio to WAV using ffmpeg and resample to sr."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",               # no video
        "-ar", str(sr),      # sample rate
        "-ac", "1",          # mono
        "-f", "wav",
        str(out_wav)
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed for {video_path}") from e

def extract_frames(video_path: str, out_dir: str, fps: int = 5):
    """Extract frames using ffmpeg at given fps, store as PNG files."""
    os.makedirs(out_dir, exist_ok=True)
    out_pattern = os.path.join(out_dir, "frame_%06d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        out_pattern
    ]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg frame extraction failed for {video_path}") from e

# -------------------------
# Face detection & cropping
# -------------------------
mtcnn_detector = MTCNN()

def detect_and_crop_faces(frame_path: str, out_dir: str, margin: float = 0.2):
    """Detect faces in an image and save cropped face(s). Returns list of crop paths."""
    os.makedirs(out_dir, exist_ok=True)
    img_bgr = cv2.imread(frame_path)
    if img_bgr is None:
        return []
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        detections = mtcnn_detector.detect_faces(img)
    except Exception as e:
        print(f"[mtcnn] detection failed for {frame_path}: {e}")
        return []
    out_paths = []
    for i, d in enumerate(detections):
        box = d.get("box")
        if not box or len(box) != 4:
            continue
        x, y, w, h = box
        # ensure positive coordinates
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = x1 + max(1, int(w))
        y2 = y1 + max(1, int(h))
        # add margin
        mw = int((x2 - x1) * margin)
        mh = int((y2 - y1) * margin)
        x1m = max(0, x1 - mw)
        y1m = max(0, y1 - mh)
        x2m = min(img.shape[1], x2 + mw)
        y2m = min(img.shape[0], y2 + mh)
        crop = img[y1m:y2m, x1m:x2m]
        if crop.size == 0:
            continue
        out_path = os.path.join(out_dir, f"{Path(frame_path).stem}_face_{i}.png")
        # write as BGR
        cv2.imwrite(out_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        out_paths.append(out_path)
    return out_paths

# -------------------------
# Audio preprocessing (example)
# -------------------------
def audio_preprocess_wav(wav_path: str, sr: int = 16000, n_mels=80):
    """Load audio, compute mel spectrogram (placeholder)."""
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio loaded from {wav_path}")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel  # numpy array

# -------------------------
# Encoder & projection placeholders
# -------------------------
def load_visual_encoder():
    """Load your visual encoder (e.g., MobileNet3D/X3D-T). Placeholder."""
    print("[stub] load_visual_encoder called")
    # returns a callable that accepts list-of-images and returns (N, D) embeddings
    return lambda images: np.random.randn(len(images), 512)

def load_audio_encoder():
    """Load your audio encoder (e.g., Wav2Vec2, TRILL-lite). Placeholder."""
    print("[stub] load_audio_encoder called")
    # accepts list-of-mel-spectrograms and returns (N, D) embeddings
    return lambda mel_specs: np.random.randn(len(mel_specs), 512)

def projection_head(embeddings: np.ndarray):
    """Project embeddings to shared space (placeholder). Accepts 1D or 2D arrays."""
    arr = np.array(embeddings)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms

# -------------------------
# Contrastive loss / training step (placeholder)
# -------------------------
def contrastive_step(visual_emb, audio_emb):
    """
    Placeholder for contrastive loss computation (NT-Xent / triplet).
    """
    v = np.asarray(visual_emb)
    a = np.asarray(audio_emb)
    # ensure same shape conformity for the stub
    if v.ndim == 2 and a.ndim == 2 and v.shape[1] != a.shape[1]:
        # reduce to min dim
        min_dim = min(v.shape[1], a.shape[1])
        v = v[:, :min_dim]
        a = a[:, :min_dim]
    # dummy loss: absolute difference of means
    return float(np.abs(np.mean(v) - np.mean(a)))

# -------------------------
# LipSync checker & classifier placeholders
# -------------------------
def lip_sync_check(face_frame_paths, wav_segment_path):
    """Placeholder: run SyncNet or pretrained lip-sync model."""
    print("[stub] lip_sync_check called for", len(face_frame_paths), "frames and", wav_segment_path)
    return True

def classifier_predict(fused_feature):
    """Placeholder classifier head to predict deepfake or clean."""
    arr = np.asarray(fused_feature).ravel()
    score = float(np.tanh(np.sum(arr)))
    # Map score to [0,1]
    fake_score = (score + 1.0) / 2.0
    return {"fake_score": float(np.clip(fake_score, 0.0, 1.0))}

# -------------------------
# Feature fusion (concat + gating placeholder)
# -------------------------
def feature_fusion(visual_proj, audio_proj):
    """Concatenate + gating (simple element-wise multiply placeholder)."""
    v = np.asarray(visual_proj)
    a = np.asarray(audio_proj)
    # make sure 2D
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    min_dim = min(v.shape[1], a.shape[1])
    v0 = v[:, :min_dim]
    a0 = a[:, :min_dim]
    fused = np.concatenate([v0, a0, v0 * a0], axis=1)
    return fused

# -------------------------
# High-level file processing
# -------------------------
def process_video_only(file_path: str, out_dir: str, visual_model):
    """Process video-only pipeline (no audio)."""
    print(f"[video-only] Processing {file_path}")
    tmp_frames = os.path.join(out_dir, "frames")
    extract_frames(file_path, tmp_frames, fps=5)

    face_crops_dir = os.path.join(out_dir, "face_crops")
    os.makedirs(face_crops_dir, exist_ok=True)

    all_face_embeddings = []
    frame_files = sorted(Path(tmp_frames).glob("*.png"))
    if not frame_files:
        print(f"[video-only] No frames extracted for {file_path}")
        return

    for f in frame_files:
        crops = detect_and_crop_faces(str(f), face_crops_dir)
        if not crops:
            continue
        images = []
        for c in crops:
            img_bgr = cv2.imread(c)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
        if not images:
            continue
        emb = visual_model(images)  # (num_crops, D)
        proj = projection_head(emb)  # (num_crops, D)
        # aggregate per-frame by mean then append
        if proj.size:
            frame_agg = proj.mean(axis=0, keepdims=True)
            all_face_embeddings.append(frame_agg)

    if all_face_embeddings:
        all_face_embeddings = np.vstack(all_face_embeddings)
        np.save(os.path.join(out_dir, "visual_embeddings.npy"), all_face_embeddings)
        fused = all_face_embeddings.mean(axis=0)
        pred = classifier_predict(fused)
        print("[video-only] classifier:", pred)
        # Save prediction
        with open(os.path.join(out_dir, "pred.json"), "w") as wf:
            json.dump({"pred": pred}, wf)
    else:
        print("[video-only] No face embeddings found for", file_path)

def process_audio_video(file_path: str, out_dir: str, visual_model, audio_model):
    """Process audio+video pipeline."""
    print(f"[audio+video] Processing {file_path}")
    tmpdir = tempfile.mkdtemp(prefix="proc_")
    try:
        wav_path = os.path.join(tmpdir, "audio.wav")
        extract_audio(file_path, wav_path, sr=16000)

        # Audio preprocessing -> spectrograms / chunks
        try:
            log_mel = audio_preprocess_wav(wav_path)
        except Exception as e:
            print(f"[audio] preprocessing failed: {e}")
            log_mel = None

        if log_mel is None:
            audio_proj = np.zeros((1, 512))
        else:
            audio_emb = audio_model([log_mel])  # list of one
            audio_proj = projection_head(np.vstack(audio_emb))

        # Visual branch
        tmp_frames = os.path.join(out_dir, "frames")
        extract_frames(file_path, tmp_frames, fps=5)
        face_crops_dir = os.path.join(out_dir, "face_crops")
        os.makedirs(face_crops_dir, exist_ok=True)

        all_face_embeddings = []
        for f in sorted(Path(tmp_frames).glob("*.png")):
            crops = detect_and_crop_faces(str(f), face_crops_dir)
            if not crops:
                continue
            images = []
            for c in crops:
                img_bgr = cv2.imread(c)
                if img_bgr is None:
                    continue
                images.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if not images:
                continue
            emb = visual_model(images)
            proj = projection_head(emb)
            # aggregate faces in this frame -> mean vector
            if proj.size:
                frame_vec = proj.mean(axis=0, keepdims=True)
                all_face_embeddings.append(frame_vec)

        if all_face_embeddings:
            all_face_embeddings = np.vstack(all_face_embeddings)
            visual_proj = all_face_embeddings.mean(axis=0, keepdims=True)
        else:
            # fallback: zero vector matching audio_proj dim
            dim = audio_proj.shape[1] if hasattr(audio_proj, "shape") else 512
            visual_proj = np.zeros((1, dim))

        # Contrastive step (stub)
        loss = contrastive_step(visual_proj, audio_proj)
        print("[audio+video] contrastive loss (stub):", loss)

        # Feature fusion and lip-sync check
        fused_feature = feature_fusion(visual_proj, audio_proj)
        face_frame_paths = list(Path(face_crops_dir).glob("*.png"))[:16]
        lip_sync_ok = lip_sync_check(face_frame_paths, wav_path)
        print("[audio+video] lip_sync_ok:", lip_sync_ok)

        # Classifier on fused features
        pred = classifier_predict(fused_feature.flatten())
        print("[audio+video] classifier:", pred)

        # Save artifacts
        np.save(os.path.join(out_dir, "audio_embedding.npy"), audio_proj)
        np.save(os.path.join(out_dir, "visual_embedding.npy"), visual_proj)
        with open(os.path.join(out_dir, "pred.json"), "w") as wf:
            json.dump({"pred": pred, "lip_sync": lip_sync_ok, "loss_stub": loss}, wf)

    finally:
        shutil.rmtree(tmpdir)

# -------------------------
# Directory / dataset driver
# -------------------------
def process_dataset(dataset_dir: str, out_root: str):
    dataset_dir = Path(dataset_dir)
    out_root = Path(out_root)
    os.makedirs(out_root, exist_ok=True)

    visual_model = load_visual_encoder()
    audio_model = load_audio_encoder()

    for media in sorted(dataset_dir.glob("*")):
        if media.is_dir():
            continue
        if media.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
            print(f"[skip] Not a video: {media.name}")
            continue

        out_dir = out_root / media.stem
        os.makedirs(out_dir, exist_ok=True)

        try:
            if has_audio_stream(str(media)):
                process_audio_video(str(media), str(out_dir), visual_model, audio_model)
            else:
                process_video_only(str(media), str(out_dir), visual_model)
        except Exception as e:
            print(f"[error] processing {media.name}: {e}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pipeline_router.py /path/to/dataset /path/to/output_dir")
        sys.exit(1)
    dataset_dir = sys.argv[1]
    out_root = sys.argv[2]
    process_dataset(dataset_dir, out_root)
    print("Done.")
