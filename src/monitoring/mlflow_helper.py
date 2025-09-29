import os
from typing import Optional, Dict, Any

import mlflow
from dotenv import load_dotenv

# Use relative imports when running as a package; fallback to absolute for direct execution
try:
    from ..utils.logger import get_logger
except ImportError:
    from utils.logger import get_logger

logger = get_logger(__name__)


def setup_mlflow(experiment_name: str) -> None:
    """
    Configure MLflow from .env and set the experiment.
    Expects:
      - MLFLOW_TRACKING_URI
      - MLFLOW_TRACKING_USERNAME (optional)
      - MLFLOW_TRACKING_PASSWORD (optional)
    """
    load_dotenv()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI not set in .env")

    mlflow.set_tracking_uri(tracking_uri)

    # MLflow supports basic auth via these env vars; used by DagsHub
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password
        logger.info("MLflow auth set via environment variables.")

    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow configured. Tracking URI: {tracking_uri}, Experiment: {experiment_name}")


class MlflowHelper:
    """
    Lightweight helper for MLflow logging.
    Usage:
        setup_mlflow("my-experiment")
        with MlflowHelper(run_name="inference") as mlf:
            mlf.log_params({...})
            mlf.log_metrics({...})
            mlf.log_text("input.txt", "some text")
    """

    def __init__(self, run_name: Optional[str] = None, nested: bool = False, tags: Optional[Dict[str, str]] = None):
        self.run_name = run_name
        self.nested = nested
        self.tags = tags or {}
        self.active_run = None

    def __enter__(self):
        self.active_run = mlflow.start_run(run_name=self.run_name, nested=self.nested)
        if self.tags:
            mlflow.set_tags(self.tags)
        logger.info(f"MLflow run started: run_id={self.active_run.info.run_id}, run_name={self.run_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.exception("Exception during MLflow run.", exc_info=(exc_type, exc_val, exc_tb))
        mlflow.end_run()
        logger.info("MLflow run ended.")

    # Logging helpers

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def set_tags(self, tags: Dict[str, str]) -> None:
        mlflow.set_tags(tags)

    def log_text(self, artifact_path: str, text: str) -> None:
        """
        Log a text artifact by writing to a temp file and uploading.
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, os.path.basename(artifact_path))
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            mlflow.log_artifact(path, artifact_path=os.path.dirname(artifact_path) or None)

    def log_dict(self, d: Dict[str, Any], artifact_file: str) -> None:
        mlflow.log_dict(d, artifact_file)

    def log_figure(self, fig, artifact_path: str) -> None:
        """
        Log a matplotlib figure.
        """
        mlflow.log_figure(fig, artifact_path)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)