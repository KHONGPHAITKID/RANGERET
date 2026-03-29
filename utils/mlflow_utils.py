import os
from pathlib import Path


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8")

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        primary = self.streams[0]
        return getattr(primary, "isatty", lambda: False)()

    def fileno(self):
        primary = self.streams[0]
        return primary.fileno()


class MLflowTracker:
    def __init__(self, config, logdir, run_name_default=None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enable", False))
        self.logdir = Path(logdir)
        self.run_name_default = run_name_default or self.logdir.name
        self.mlflow = None
        self.started = False

    def start(self, params=None):
        if not self.enabled:
            return

        try:
            import mlflow
        except ImportError as exc:
            raise ImportError("MLflow tracking is enabled in config, but the 'mlflow' package is not installed.") from exc

        self.mlflow = mlflow
        tracking_uri = self.config.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        experiment_name = self.config.get("experiment_name")
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        run_name = self.config.get("run_name") or self.run_name_default
        mlflow.start_run(run_name=run_name, nested=self.config.get("nested", False))
        self.started = True

        if params:
            sanitized = {key: str(value) for key, value in params.items() if value is not None}
            mlflow.log_params(sanitized)

    def log_metrics(self, metrics, step=None):
        if not self.started:
            return

        for key, value in metrics.items():
            if value is None:
                continue
            self.mlflow.log_metric(key, float(value), step=step)

    def log_metric(self, key, value, step=None):
        if not self.started:
            return
        self.mlflow.log_metric(key, float(value), step=step)

    def log_artifact(self, path, artifact_path=None):
        if not self.started:
            return
        if os.path.exists(path):
            self.mlflow.log_artifact(path, artifact_path=artifact_path)

    def log_artifacts(self, path, artifact_path=None):
        if not self.started:
            return
        if os.path.exists(path):
            self.mlflow.log_artifacts(path, artifact_path=artifact_path)

    def log_run_artifacts(self):
        if not self.started:
            return

        console_log = self.logdir / "console.log"
        if console_log.exists():
            self.log_artifact(str(console_log), artifact_path="logs")

        training_log = self.logdir / "training_log.txt"
        if training_log.exists():
            self.log_artifact(str(training_log), artifact_path="logs")

        if self.config.get("log_checkpoints", False):
            for checkpoint in self.logdir.glob("*.pt"):
                self.log_artifact(str(checkpoint), artifact_path="checkpoints")

        if self.config.get("log_code_snapshot", False):
            for name in ["train.py", "infer.py"]:
                path = Path(name)
                if path.exists():
                    self.log_artifact(str(path), artifact_path="code")
            for directory in ["modules", "network", "utils", "dataloader", "losses", "postprocess", "config"]:
                path = Path(directory)
                if path.exists():
                    self.log_artifacts(str(path), artifact_path=f"code/{directory}")

    def finish(self, status="FINISHED"):
        if not self.started:
            return
        self.log_run_artifacts()
        self.mlflow.end_run(status=status)
        self.started = False
