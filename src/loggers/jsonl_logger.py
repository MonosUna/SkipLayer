import json
import os
import time
from typing import Any


class JsonlFileLogger:
    def __init__(
        self,
        log_dir: str | None = None,
        run_name: str | None = None,
        **kwargs,
    ):
        self.log_dir = os.path.abspath(log_dir or os.getcwd())
        self.run_name = run_name
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "html"), exist_ok=True)

        self._metrics_path = os.path.join(self.log_dir, "metrics.jsonl")
        self._summary_path = os.path.join(self.log_dir, "metrics_summary.json")
        self._params_path = os.path.join(self.log_dir, "params.json")
        self._texts_path = os.path.join(self.log_dir, "texts.jsonl")

        self._summary: dict[str, Any] = {}
        if run_name:
            self._summary["_run_name"] = run_name

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
    ) -> None:
        scalar_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }
        if scalar_metrics:
            record = {
                "step": step,
                "ts": time.time(),
                "metrics": scalar_metrics,
            }
            with open(self._metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._summary.update(scalar_metrics)
            if step is not None:
                self._summary["_last_step"] = step
            self._summary["_last_ts"] = record["ts"]
            with open(self._summary_path, "w", encoding="utf-8") as f:
                json.dump(self._summary, f, ensure_ascii=False, indent=2)

        for k, v in metrics.items():
            if "html" in k and v:
                self.log_html(v, step=step, key=k)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for t in v:
                    self.log_text(
                        t.get("text", ""),
                        metadata=t.get("metadata"),
                        step=step,
                    )

    def log_params(self, params: dict[str, Any]) -> None:
        with open(self._params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2, default=str)

    def log_text(
        self,
        text: str,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "step": step,
            "ts": time.time(),
            "metadata": metadata or {},
            "text": text,
        }
        with open(self._texts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_model(self, model_path: str, model_name: str) -> None:
        return

    def log_html(
        self, html: str, step: int | None = None, key: str = "html"
    ) -> None:
        safe_key = key.replace("/", "__").replace(os.sep, "__")
        suffix = f"__step{step}" if step is not None else ""
        path = os.path.join(self.log_dir, "html", f"{safe_key}{suffix}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def finish(self) -> None:
        with open(self._summary_path, "w", encoding="utf-8") as f:
            json.dump(self._summary, f, ensure_ascii=False, indent=2)


class MultiLogger:
    def __init__(self, loggers: list[Any], **kwargs):
        self.loggers = [lg for lg in loggers if lg is not None]

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for lg in self.loggers:
            lg.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        for lg in self.loggers:
            lg.log_params(params)

    def log_text(
        self,
        text: str,
        step: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        for lg in self.loggers:
            if hasattr(lg, "log_text"):
                lg.log_text(text, step=step, metadata=metadata)

    def log_model(self, model_path: str, model_name: str) -> None:
        for lg in self.loggers:
            if hasattr(lg, "log_model"):
                lg.log_model(model_path, model_name)

    def log_html(
        self, html: str, step: int | None = None, key: str = "html"
    ) -> None:
        for lg in self.loggers:
            if hasattr(lg, "log_html"):
                lg.log_html(html, step=step, key=key)

    def finish(self) -> None:
        for lg in self.loggers:
            if hasattr(lg, "finish"):
                lg.finish()
