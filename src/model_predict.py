from __future__ import annotations
import time
from typing import List, Dict, Any, Union, Optional

import numpy as np
import tensorflow as tf

from model_loader import load_model
from utils.logger import get_logger
from monitoring.mlflow_helper import setup_mlflow, MlflowHelper
from explainability.lime_explainer import LimeExplainer

logger = get_logger(__name__)


class SuicideDetector:
    """
    TensorFlow-based text classifier wrapper with optional MLflow logging and LIME explanations.
    """
    def __init__(
        self,
        model_dir: str | None = None,
        max_length: int = 128,
        enable_mlflow: bool = False,
        mlflow_experiment: str = "SuicidalIdeationDetection",
        mlflow_tags: Optional[Dict[str, str]] = None,
    ):
        # Load tokenizer/model; keep max_length small for CPU-friendly tokenization
        self.tokenizer, self.model, self.id2label = load_model(model_dir)
        self.max_length = max_length

        # Build consistent class names and mapping
        try:
            cls_ids = sorted(int(k) if not isinstance(k, int) else k for k in self.id2label.keys())
        except Exception:
            # Fallback if keys are not coercible
            cls_ids = list(range(len(self.id2label)))
        self.class_names = [self.id2label[i] for i in cls_ids]
        self.label2id = {v: k for k, v in self.id2label.items()}

        # MLflow config
        self.mlflow_enabled = enable_mlflow
        self.mlflow_tags = mlflow_tags or {}
        if self.mlflow_enabled:
            try:
                setup_mlflow(mlflow_experiment)
                logger.info(f"MLflow enabled for experiment '{mlflow_experiment}'.")
            except Exception as e:
                # Do not fail detector construction due to MLflow config issues
                logger.warning(f"MLflow setup failed: {e}. Continuing without MLflow.")
                self.mlflow_enabled = False

        logger.info(
            f"Initialized SuicideDetector (TF) with max_length={max_length}, num_classes={len(self.class_names)}"
        )

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Returns probability array of shape (n_samples, n_classes).
        This function is compatible with LIME's classifier_fn signature.
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )

        # Check if model is TensorFlow or PyTorch
        if hasattr(self.model, 'predict') or 'TF' in str(type(self.model)):
            # TensorFlow model
            outputs = self.model(enc, training=False)
            logits = outputs.logits  # [B, C]
            probs = tf.nn.softmax(logits, axis=-1).numpy()
        else:
            # PyTorch model fallback
            import torch
            enc_pt = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**enc_pt)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1).numpy()
        
        return probs

    def _predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        start = time.perf_counter()
        probs = self.predict_proba(texts)
        latency_ms = (time.perf_counter() - start) * 1000

        preds = probs.argmax(axis=-1).tolist()
        results = []
        for i, pred_id in enumerate(preds):
            label = self.id2label.get(pred_id, str(pred_id))
            scores = {self.id2label[j]: float(probs[i][j]) for j in range(probs.shape[1])}
            results.append(
                {"label": label, "score": float(probs[i][pred_id]), "scores": scores}
            )

        # Lightweight MLflow logging per batch
        if self.mlflow_enabled:
            try:
                with MlflowHelper(run_name="inference", tags=self.mlflow_tags) as mlf:
                    mlf.log_params(
                        {
                            "max_length": self.max_length,
                            "num_classes": probs.shape[1],
                            "model_type": getattr(self.model.config, "model_type", "unknown"),
                            "model_name_or_path": getattr(self.model, "name_or_path", "local"),
                            "batch_size": len(texts),
                            "avg_input_len": float(np.mean([len(t) for t in texts])) if texts else 0.0,
                        }
                    )
                    # Log summary metrics
                    avg_conf = float(np.mean([r["score"] for r in results])) if results else 0.0
                    mlf.log_metrics({"latency_ms": latency_ms, "avg_confidence": avg_conf})
                    # Log only aggregate results to avoid storing sensitive text by default
                    mlf.log_dict({"results": results}, artifact_file="outputs/inference_results.json")
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        return results

    def predict(
        self,
        text_or_texts: Union[str, List[str]],
        log_to_mlflow: bool = False,
        log_input_text: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Predict for a single text or a batch of texts.
        Optionally log the request/response to MLflow. Input text logging is disabled by default.
        """
        if isinstance(text_or_texts, str):
            texts = [text_or_texts]
            results = self._predict_batch(texts)
            result = results[0]

            if log_to_mlflow and self.mlflow_enabled:
                try:
                    with MlflowHelper(run_name="inference_single", tags=self.mlflow_tags) as mlf:
                        mlf.log_params({"max_length": self.max_length})
                        mlf.log_metrics({"confidence": float(result["score"])})
                        mlf.log_dict(result, artifact_file="outputs/single_result.json")
                        if log_input_text:
                            mlf.log_text("inputs/single_input.txt", text_or_texts)
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")
            return result

            # List input
        if isinstance(text_or_texts, list):
            results = self._predict_batch(text_or_texts)

            if log_to_mlflow and self.mlflow_enabled:
                try:
                    with MlflowHelper(run_name="inference_batch", tags=self.mlflow_tags) as mlf:
                        mlf.log_params({"max_length": self.max_length, "batch_size": len(text_or_texts)})
                        avg_conf = float(np.mean([r["score"] for r in results])) if results else 0.0
                        mlf.log_metrics({"avg_confidence": avg_conf})
                        mlf.log_dict({"results": results}, artifact_file="outputs/batch_results.json")
                        if log_input_text:
                            # Redact or limit logging of raw inputs if needed
                            joined = "\n".join(text_or_texts)
                            mlf.log_text("inputs/batch_inputs.txt", joined)
                except Exception as e:
                    logger.warning(f"MLflow logging failed: {e}")
            return results

        raise TypeError("Input must be a string or list of strings.")
    #LIME explanations

    def explain(
        self,
        text: str,
        num_features: int = 10,
        top_labels: int = 1,
        log_to_mlflow: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a LIME explanation for a single text.
        Returns a dict with 'summary' and 'explanation' keys.
        """
        # Initialize LIME with our class names
        explainer = LimeExplainer(class_names=self.class_names)
        explanation = explainer.explain(
            text=text,
            predict_proba=self.predict_proba,
            num_features=num_features,
            top_labels=top_labels,
        )

        if log_to_mlflow and self.mlflow_enabled:
            try:
                with MlflowHelper(run_name="explain", tags=self.mlflow_tags) as mlf:
                    mlf.log_params({"num_features": num_features, "top_labels": top_labels})
                    mlf.log_dict(explanation, artifact_file="explanations/lime_explanation.json")
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        return explanation


if __name__ == "__main__":
    # Test the SuicideDetector
    print("Initializing SuicideDetector...")
    detector = SuicideDetector(enable_mlflow=True)  # Enable MLflow for testing
    
    # Test single prediction
    test_text = "I'm feeling really sad and hopeless today"
    print(f"\nTesting single prediction with: '{test_text}'")
    result = detector.predict(test_text, log_to_mlflow=True)
    print(f"Result: {result}")
    
    # Test batch prediction
    test_texts = [
        "I'm having a great day!",
        "I don't want to live anymore",
        "Just finished a good workout"
    ]
    print(f"\nTesting batch prediction with {len(test_texts)} texts...")
    batch_results = detector.predict(test_texts, log_to_mlflow=True)
    for i, res in enumerate(batch_results):
        print(f"Text {i+1}: {res}")
    
    # Test LIME explanation
    print(f"\nGenerating LIME explanation for: '{test_text}'")
    explanation = detector.explain(test_text, num_features=5, log_to_mlflow=True)
    print(f"Explanation summary: {explanation['summary']}")
    
    print("\nAll tests completed successfully!")