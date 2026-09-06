"""Lazy loaders for heavy vision models (YOLO, CLIP, NIMA, optional DeepFace)."""

import importlib.util
import logging
import os
from functools import wraps
from threading import RLock

from .runtime_paths import MODEL_DIR

logger = logging.getLogger(__name__)
_model_init_lock = RLock()


def _serialized_init(loader):
    @wraps(loader)
    def load(*args, **kwargs):
        with _model_init_lock:
            return loader(*args, **kwargs)
    return load


# YOLO
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", os.path.join(MODEL_DIR, "yolov8n.pt"))
_yolo_model = None


@_serialized_init
def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        weights_dir = os.path.dirname(YOLO_WEIGHTS)
        os.makedirs(weights_dir, exist_ok=True)
        _yolo_model = YOLO(YOLO_WEIGHTS)
    return _yolo_model


# CLIP
_clip_model = None
_clip_processor = None
_device = None


@_serialized_init
def get_clip_model():
    global _clip_model, _clip_processor, _device
    if _clip_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model, _clip_processor = model, processor
    return _clip_model, _clip_processor, _device


def warm_public_models():
    """Initialize once in the serving process, before accepting demo traffic."""
    get_yolo_model()
    get_clip_model()


def public_models_ready():
    return _yolo_model is not None and _clip_model is not None and _clip_processor is not None


# NIMA (optional TF)
NIMA_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
if NIMA_AVAILABLE:

    def _import_tf_keras():
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
        from tensorflow.keras.layers import Dense, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.preprocessing import image as keras_image

        return Model, Dense, Dropout, MobileNet, preprocess_input, keras_image

else:
    _import_tf_keras = None

_nima_model = None


def get_nima_model():
    global _nima_model
    if _nima_model is None and NIMA_AVAILABLE:
        try:
            model_path = os.path.join(MODEL_DIR, "nima_mobilenet.h5")
            Model, Dense, Dropout, MobileNet, _, _ = _import_tf_keras()
            base_model = MobileNet((None, None, 3), include_top=False, pooling="avg", weights=None)
            x = Dropout(0.75)(base_model.output)
            x = Dense(10, activation="softmax")(x)
            _nima_model = Model(base_model.input, x)
            _nima_model.load_weights(model_path)
        except Exception as e:
            logger.warning("NIMA disabled (weights missing or load failed): %s", e)
            _nima_model = None
    return _nima_model


# DeepFace (optional)
DEEPFACE_ENABLE = os.environ.get("DEEPFACE_ENABLE", "false").lower() == "true"
if DEEPFACE_ENABLE:
    try:
        from deepface import DeepFace
    except Exception:
        DeepFace = None
else:
    DeepFace = None

