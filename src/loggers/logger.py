from typing import Any

from src.loggers.utils import build_html_dashboard


class CometMLLogger:
    def __init__(
        self,
        api_key: str,
        project_name: str,
        workspace: str | None = None,
        run_name: str | None = None,
        **kwargs,
    ):
        import comet_ml

        if run_name:
            api = comet_ml.API(api_key=api_key)
            existing_experiments = api.get_experiments(
                workspace=workspace, project_name=project_name
            )
            for exp in existing_experiments:
                if exp.get_name() == run_name:
                    raise RuntimeError(
                        f"CometML run with name '{run_name}' already exists in "
                        f"project '{project_name}'. Please choose a different logger.run_name."
                    )

        self.experiment = comet_ml.Experiment(
            api_key=api_key, project_name=project_name, workspace=workspace, **kwargs
        )

        if run_name:
            self.experiment.set_name(run_name)

        self._html_by_step: dict[str, dict[int, str]] = {}

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
    ) -> None:
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int | float))}
        if scalar_metrics:
            self.experiment.log_metrics(scalar_metrics, step=step)

        for k, v in metrics.items():
            if "html" in k and v:
                self.log_html(v, step=step, key=k)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for t in v:
                    self.log_text(t["text"], metadata=t["metadata"], step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        flat_params = self._flatten_dict(params)
        self.experiment.log_parameters(flat_params)

    def log_text(
        self, text: str, step: int | None = None, metadata: dict[str, Any] | None = None
    ) -> None:
        self.experiment.log_text(text, step=step, metadata=metadata)

    def log_model(self, model_path: str, model_name: str) -> None:
        self.experiment.log_model(model_name, model_path)

    def log_html(self, html: str, step: int | None = None, key: str = "html") -> None:
        step_val = step if step is not None else 0
        self._html_by_step.setdefault(key, {})[step_val] = html
        combined = build_html_dashboard(self._html_by_step)
        self.experiment.log_html(combined, clear=True)

    def finish(self) -> None:
        self.experiment.end()

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(CometMLLogger._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
