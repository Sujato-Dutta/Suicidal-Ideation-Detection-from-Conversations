# Module imports and logger setup
from pathlib import Path
import os
from typing import Tuple, Dict, Any
from dotenv import load_dotenv
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Use relative imports when running as a package; fallback to absolute for direct execution
try:
    from .utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__)

# Force CPU-only usage for TensorFlow
tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('ERROR')  # Suppress TF logs

def get_default_model_dir() -> Path:
    load_dotenv()
    env_dir = os.getenv("MODEL_DIR")

    # Resolve project root as the parent of the 'src' directory
    project_root = Path(__file__).resolve().parents[1]

    if env_dir:
        p = Path(env_dir)
        # If env path is relative, make it relative to project root
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.expanduser().resolve()
        logger.info(f"Using model dir from .env MODEL_DIR={p}")
        return p

    # Default to artifacts folder inside the project
    default_dir = (
        project_root
        / "artifacts"
        / "fine_tuned_tinybert_suicide_detection"
        / "fine_tuned_tinybert_suicide_detection"
    ).resolve()
    logger.info(f"Using default model dir: {default_dir}")
    return default_dir


def load_model(model_dir: str = None) -> Tuple[Any, Any, Dict[int, str]]:
    """
    Load TensorFlow model and tokenizer from the specified directory.
    Returns: (tokenizer, model, id2label)
    """
    if model_dir is None:
        model_dir = get_default_model_dir()
    else:
        model_dir = Path(model_dir)
    
    logger.info(f"Loading model from: {model_dir}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        
        # Try to load TensorFlow model first, fallback to PyTorch if needed
        try:
            model = TFAutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                from_tf=True  # Try loading from TensorFlow format first
            )
            logger.info("Loaded TensorFlow model directly")
        except (OSError, ValueError, TypeError):
            # If TF model doesn't exist or fails, try PyTorch and convert
            logger.info("TensorFlow model not found, attempting to load PyTorch model...")
            try:
                from transformers import AutoModelForSequenceClassification
                # Load PyTorch model first
                pt_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
                # Convert to TensorFlow
                model = TFAutoModelForSequenceClassification.from_pretrained(
                    str(model_dir), 
                    from_pt=True
                )
                logger.info("Successfully converted PyTorch model to TensorFlow")
            except Exception as convert_error:
                logger.error(f"Failed to convert PyTorch model: {convert_error}")
                # Last resort: use PyTorch model with TF wrapper
                logger.info("Using PyTorch model as fallback...")
                from transformers import AutoModelForSequenceClassification
                pt_model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
                # We'll need to modify the predict functions to handle PyTorch
                model = pt_model
        
        # Extract label mapping
        config = model.config
        id2label = getattr(config, 'id2label', {0: 'non_suicidal', 1: 'suicidal'})
        
        # Ensure id2label has integer keys
        if isinstance(list(id2label.keys())[0], str):
            id2label = {int(k): v for k, v in id2label.items()}
        
        logger.info(f"Model loaded successfully. Labels: {id2label}")
        return tokenizer, model, id2label
        
    except Exception as e:
        logger.error(f"Failed to load model from {model_dir}: {e}")
        raise