from typing import Callable, List, Dict, Any, Optional
import numpy as np
from lime.lime_text import LimeTextExplainer

# Use relative imports when running as a package; fallback to absolute for direct execution
try:
    from ..utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__)


class LimeTextExplanationResult:
    """
    A serializable container for LIME explanation results.
    """
    def __init__(self, text: str, class_names: List[str], top_labels: List[int], contributions: Dict[int, List[Dict[str, float]]]):
        self.text = text
        self.class_names = class_names
        self.top_labels = top_labels
        self.contributions = contributions  # {label_id: [{token: weight}, ...]}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "class_names": self.class_names,
            "top_labels": self.top_labels,
            "contributions": self.contributions,
        }


class LimeExplainer:
    """
    Wraps LimeTextExplainer for text classification.
    Provide:
      - class_names (e.g., ["non_suicidal", "suicidal"])
      - predict_proba: Callable[[List[str]], np.ndarray] -> (n_samples, n_classes)
    """
    def __init__(self, class_names: List[str], kernel_width: float = 25.0, random_state: int = 42):
        self.class_names = class_names
        self.explainer = LimeTextExplainer(class_names=class_names, kernel_width=kernel_width, random_state=random_state)
        logger.info(f"LIME explainer initialized with class_names={class_names}")

    def explain(
        self,
        text: str,
        predict_proba: Callable[[List[str]], np.ndarray],
        num_features: int = 10,
        top_labels: int = 1,
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single text.
        Returns a dict containing:
          - summary (top labels and weights)
          - raw explanation in a serialized form (token weights per label)
        """
        logger.info("Generating LIME explanation...")
        exp = self.explainer.explain_instance(
            text_instance=text,
            classifier_fn=predict_proba,
            num_features=num_features,
            top_labels=top_labels,
        )

        top_label_ids = exp.top_labels
        contributions: Dict[int, List[Dict[str, float]]] = {}
        for label_id in top_label_ids:
            # List of tuples (token, weight)
            weights = exp.as_list(label=label_id)
            contributions[label_id] = [{"token": tok, "weight": float(w)} for tok, w in weights]

        result = LimeTextExplanationResult(
            text=text,
            class_names=self.class_names,
            top_labels=top_label_ids,
            contributions=contributions,
        )

        return {
            "summary": {
                "top_labels": [
                    {"label_id": int(lid), "label_name": self.class_names[int(lid)] if int(lid) < len(self.class_names) else str(lid)}
                    for lid in top_label_ids
                ],
                "num_features": num_features,
            },
            "explanation": result.to_dict(),
        }