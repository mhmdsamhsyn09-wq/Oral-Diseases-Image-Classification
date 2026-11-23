from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# ===================================================================== page & style
st.set_page_config(
    page_title="Oral Disease Classifier",
    page_icon="🦷",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .block-container {
        max-width: 1100px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
    }
    #hero {
        text-align: center;
        padding: 22px 14px;
        border-radius: 16px;
        background: linear-gradient(135deg, #0ea5e9, #2563eb);
        color: #ffffff;
        margin-bottom: 18px;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.18);
    }
    #hero h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    #hero p {
        color: #e0f2fe;
        margin: 6px 0 0;
        font-size: 1rem;
    }
    .result-card-ok {
        background: #f0f9ff;
        border-left: 4px solid #0284c7;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .result-card-warn {
        background: #fffbeb;
        border-left: 4px solid #d97706;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===================================================================== config
MODEL_PATH = os.getenv("MODEL_PATH", "ResNet50V2.keras")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "mo123essam/oral-disease-resnet50v2")
HF_MODEL_FILENAME = os.getenv("HF_MODEL_FILENAME", "ResNet50V2.keras")

PREPROCESS = os.getenv("PREPROCESS", "none").lower()
IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
CLASSES_FILE = Path(os.getenv("CLASSES_FILE", "classes.json"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))
TEST_DIR = os.getenv("TEST_DIR", "data/test")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.5))

DEFAULT_CLASSES = [
    "Calculus",
    "Data caries",
    "Gingivitis",
    "Mouth Ulcer",
    "Tooth Discoloration",
    "hypodontia",
]

HISTORY_FILE = RESULTS_DIR / "history.json"
HISTORY_COLUMNS = ["Time", "Prediction", "Confidence", "Status", "Preprocessing"]

MODES = ["none", "rescale", "resnetv2"]
if PREPROCESS not in MODES:
  PREPROCESS = "none"


# ===================================================================== history
def _load_history_records():
  if not HISTORY_FILE.exists():
    return []
  try:
    data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []
  except Exception:
    return []


def _history_df():
  records = _load_history_records()
  if not records:
    return pd.DataFrame(columns=HISTORY_COLUMNS)
  df = pd.DataFrame(records)
  for col in HISTORY_COLUMNS:
    if col not in df.columns:
      df[col] = ""
  return df[HISTORY_COLUMNS]


def _append_history(prediction: str, confidence: float, status: str, mode: str):
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  records = _load_history_records()
  records.insert(
      0,
      {
          "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          "Prediction": prediction,
          "Confidence": f"{confidence:.1%}",
          "Status": status,
          "Preprocessing": mode,
      },
  )
  HISTORY_FILE.write_text(
      json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
  )


def clear_history():
  try:
    if HISTORY_FILE.exists():
      HISTORY_FILE.unlink()
  except OSError:
    pass


# ===================================================================== model
def _find_classes(n: int):
  if CLASSES_FILE.exists():
    try:
      names = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
      if isinstance(names, list) and len(names) == n:
        return names
    except Exception:
      pass
  if os.path.isdir(TEST_DIR):
    names = sorted(
        d
        for d in os.listdir(TEST_DIR)
        if os.path.isdir(os.path.join(TEST_DIR, d))
    )
    if len(names) == n:
      return names
  if n == len(DEFAULT_CLASSES):
    return DEFAULT_CLASSES
  return [f"Class {i}" for i in range(n)]


def _strip_quantization(path: str) -> str:
  """Write a copy of a .keras file whose config.json has no `quantization_config` keys."""

  def clean(obj):
    if isinstance(obj, dict):
      return {
          k: clean(v) for k, v in obj.items() if k != "quantization_config"
      }
    if isinstance(obj, list):
      return [clean(v) for v in obj]
    return obj

  out = os.path.join(tempfile.gettempdir(), "oral_model_compat.keras")
  with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(
      out, "w", zipfile.ZIP_DEFLATED
  ) as zout:
    for item in zin.infolist():
      data = zin.read(item.filename)
      if item.filename == "config.json":
        data = json.dumps(clean(json.loads(data))).encode("utf-8")
      zout.writestr(item, data)
  return out


def _resolve_model_file(
    model_path: str, hf_repo: str, hf_filename: str
) -> tuple[str | None, str | None]:
  if model_path and os.path.exists(model_path):
    return model_path, None

  try:
    downloaded = hf_hub_download(
        repo_id=hf_repo,
        filename=hf_filename,
    )
    return downloaded, None
  except Exception as e:
    return (
        None,
        f"Model not found locally (`{model_path}`) and failed to download from `{hf_repo}/{hf_filename}`: {e}",
    )


@st.cache_resource(show_spinner="⏳ Loading ResNet50V2 oral disease model...")
def load_model_cached(
    model_path: str, hf_repo: str, hf_filename: str, default_img_size: int
):
  resolved_path, resolve_err = _resolve_model_file(
      model_path, hf_repo, hf_filename
  )
  if not resolved_path or not os.path.exists(resolved_path):
    return None, [], (0, 0), resolve_err

  try:
    import tensorflow as tf

    try:
      net = tf.keras.models.load_model(resolved_path, compile=False)
    except Exception as first_err:
      if "quantization_config" not in str(first_err):
        raise
      net = tf.keras.models.load_model(
          _strip_quantization(resolved_path), compile=False
      )

    shp = (
        net.input_shape[0]
        if isinstance(net.input_shape, list)
        else net.input_shape
    )
    h, w = shp[1], shp[2]
    size = (int(w), int(h)) if h and w else (default_img_size, default_img_size)
    n = int(net.output_shape[-1])
    return net, _find_classes(n), size, None
  except Exception as e:
    return None, [], (0, 0), f"Could not load model: {e}"


MODEL, CLASSES, SIZE, MODEL_ERROR = load_model_cached(
    MODEL_PATH, HF_MODEL_REPO, HF_MODEL_FILENAME, IMG_SIZE
)


def _prep(img: Image.Image, mode: str, target_size: tuple[int, int]) -> np.ndarray:
  arr = np.asarray(
      img.convert("RGB").resize(target_size, Image.BILINEAR), dtype=np.float32
  )
  if mode == "resnetv2":
    arr = arr / 127.5 - 1.0
  elif mode == "rescale":
    arr = arr / 255.0
  return arr


def predict_probs(
    imgs: list[Image.Image], mode: str | None = None
) -> np.ndarray:
  active_mode = mode or PREPROCESS
  x = np.stack([_prep(i, active_mode, SIZE) for i in imgs])
  out_tensor = MODEL(x, training=False)
  p = np.asarray(
      out_tensor.numpy() if hasattr(out_tensor, "numpy") else out_tensor,
      dtype=np.float64,
  )
  if p.min() < 0 or p.max() > 1 or not np.allclose(p.sum(1), 1, atol=1e-3):
    e = np.exp(p - p.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
  return p


# ===================================================================== saved results helpers
def _img_path(name: str) -> str | None:
  p = RESULTS_DIR / name
  return str(p) if p.exists() else None


def _comparison_df() -> pd.DataFrame:
  p = RESULTS_DIR / "comparison.csv"
  if p.exists():
    try:
      return pd.read_csv(p)
    except Exception:
      pass
  return pd.DataFrame(columns=["model", "accuracy", "f1"])


def _saved_metrics_md() -> str:
  p = RESULTS_DIR / "metrics.json"
  if not p.exists():
    return "_No saved `results/metrics.json` found._"
  try:
    m = json.loads(p.read_text(encoding="utf-8"))
  except Exception:
    return "_Could not parse `results/metrics.json`._"
  parts = [
      f"**{k.replace('_', ' ').title()}:** {v:.2%}"
      if isinstance(v, float) and v <= 1
      else f"**{k}:** {v}"
      for k, v
