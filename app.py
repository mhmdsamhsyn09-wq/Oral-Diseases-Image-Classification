from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import zipfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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


@st.cache_resource(show_spinner="⏳ Loading ResNet50V2 oral disease model...")
def load_model_cached(model_path: str, default_img_size: int):
  if not os.path.exists(model_path):
    return (
        None,
        [],
        (0, 0),
        f"Model file not found: `{model_path}`. Put it next to `app.py` (or set `MODEL_PATH`).",
    )
  try:
    import tensorflow as tf

    try:
      net = tf.keras.models.load_model(model_path, compile=False)
    except Exception as first_err:
      if "quantization_config" not in str(first_err):
        raise
      net = tf.keras.models.load_model(
          _strip_quantization(model_path), compile=False
      )

    shp = net.input_shape[0] if isinstance(net.input_shape, list) else net.input_shape
    h, w = shp[1], shp[2]
    size = (int(w), int(h)) if h and w else (default_img_size, default_img_size)
    n = int(net.output_shape[-1])
    return net, _find_classes(n), size, None
  except Exception as e:
    return None, [], (0, 0), f"Could not load model: {e}"


MODEL, CLASSES, SIZE, MODEL_ERROR = load_model_cached(MODEL_PATH, IMG_SIZE)


def _prep(img: Image.Image, mode: str, target_size: tuple[int, int]) -> np.ndarray:
  arr = np.asarray(
      img.convert("RGB").resize(target_size, Image.BILINEAR), dtype=np.float32
  )
  if mode == "resnetv2":
    arr = arr / 127.5 - 1.0
  elif mode == "rescale":
    arr = arr / 255.0
  return arr


def predict_probs(imgs: list[Image.Image], mode: str | None = None) -> np.ndarray:
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
      for k, v in m.items()
  ]
  return " &nbsp;|&nbsp; ".join(parts) if parts else "_Empty `metrics.json`._"


# ===================================================================== UI
st.markdown(
    """
    <div id='hero'>
        <h1>🦷 Oral Disease Classifier</h1>
        <p>Deep-learning diagnosis support powered by ResNet50V2</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if MODEL is None:
  st.warning(f"⚠️ {MODEL_ERROR}")

if "last_prediction" not in st.session_state:
  st.session_state.last_prediction = None
if "uploader_key" not in st.session_state:
  st.session_state.uploader_key = 0


def reset_image():
  st.session_state.uploader_key += 1
  st.session_state.last_prediction = None


tab_diag, tab_results, tab_about = st.tabs(
    ["🔍 Diagnosis", "📊 Model Results", "ℹ️ About"]
)

# --------------------------------------------------------------------- Tab 1: Diagnosis
with tab_diag:
  col_left, col_right = st.columns([1, 1], gap="large")

  with col_left:
    uploaded_file = st.file_uploader(
        "Upload an oral image",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"oral_uploader_{st.session_state.uploader_key}",
    )

    pil_img = None
    if uploaded_file is not None:
      pil_img = Image.open(uploaded_file)
      st.image(pil_img, caption="Uploaded Oral Image", use_container_width=True)

    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
      st.button("Clear", on_click=reset_image, use_container_width=True)
    with b_col2:
      classify_clicked = st.button(
          "Classify", type="primary", use_container_width=True
      )

    if classify_clicked:
      if MODEL is None:
        st.error(MODEL_ERROR)
      elif pil_img is None:
        st.error("Please upload an image first.")
      else:
        with st.spinner("Analyzing oral image..."):
          probs = predict_probs([pil_img], PREPROCESS)[0]
          top_idx = int(probs.argmax())
          conf = float(probs[top_idx])
          pred_class = CLASSES[top_idx]

          if conf >= CONF_THRESHOLD:
            status = "Confident"
            headline = f"Prediction: **{pred_class}**"
            detail = f"Confidence: **{conf:.1%}**"
            is_low_conf = False
          else:
            status = "Low confidence"
            headline = f"Low confidence ({conf:.1%})"
            detail = (
                f"Best guess: **{pred_class}**. Try a clearer, well-lit"
                " close-up image."
            )
            is_low_conf = True

          _append_history(pred_class, conf, status, PREPROCESS)
          st.session_state.last_prediction = {
              "headline": headline,
              "detail": detail,
              "is_low_conf": is_low_conf,
              "probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))},
          }

  with col_right:
    st.subheader("Diagnosis Output")
    pred_state = st.session_state.last_prediction
    if pred_state is None:
      st.info("Upload an image and click **Classify** to see the prediction.")
    else:
      card_cls = "result-card-warn" if pred_state["is_low_conf"] else "result-card-ok"
      st.markdown(
          f"""
          <div class="{card_cls}">
              <div style="font-size: 1.15rem; margin-bottom: 4px;">{pred_state["headline"]}</div>
              <div>{pred_state["detail"]}</div>
          </div>
          """,
          unsafe_allow_html=True,
      )
      st.markdown("**Class probabilities**")
      sorted_probs = sorted(
          pred_state["probs"].items(), key=lambda x: x[1], reverse=True
      )
      for cls_name, prob_val in sorted_probs:
        pct = prob_val * 100.0
        st.markdown(f"**{cls_name}** — `{pct:.1f}%`")
        st.progress(min(max(float(prob_val), 0.0), 1.0))

# --------------------------------------------------------------------- Tab 2: Model Results
with tab_results:
  st.markdown("## ResNet50V2 - saved results")
  st.markdown(_saved_metrics_md(), unsafe_allow_html=True)

  curves_img = _img_path("training_curves.png")
  cm_img = _img_path("confusion_matrix.png")

  if curves_img or cm_img:
    p_col1, p_col2 = st.columns(2, gap="medium")
    with p_col1:
      if curves_img:
        st.image(curves_img, caption="Training curves", use_container_width=True)
    with p_col2:
      if cm_img:
        st.image(cm_img, caption="Confusion matrix", use_container_width=True)
  else:
    st.caption(
        "_No saved plots yet. Put `confusion_matrix.png` or `training_curves.png` inside `results/`._"
    )

  st.markdown("### Model comparison")
  st.dataframe(_comparison_df(), use_container_width=True, hide_index=True)

  st.markdown("## 🕒 Prediction History")
  st.caption(
      "Saved predictions appear here automatically and stay available across sessions."
  )
  st.dataframe(
      _history_df(),
      column_config={col: col for col in HISTORY_COLUMNS},
      use_container_width=True,
      hide_index=True,
  )
  if st.button("Clear history", type="secondary"):
    clear_history()
    st.rerun()

# --------------------------------------------------------------------- Tab 3: About
with tab_about:
  st.markdown(
      f"""
      **Task:** classify oral-disease images into their categories.

      **Model:** ResNet50V2 fine-tuned (best of the compared models: custom CNN vs pretrained networks).

      **Classes:** {", ".join(CLASSES) if CLASSES else "n/a"}

      **Input size:** {SIZE[0]}x{SIZE[1]}

      > ⚠️ This tool is for educational purposes only and is **not** a medical diagnosis. Consult a qualified dentist or physician.
      """
  )