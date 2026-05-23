from __future__ import annotations

from .mcp_tools import (
    compare_retrain_policies,
    find_train_history_window,
    monitor_decay,
    plan_walk_forward,
    preview_dataset,
    run_walk_forward,
    run_walk_forward_baseline,
)


def build_server():
    """Build the Jano MCP server lazily so importing jano does not require MCP."""

    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:  # pragma: no cover - exercised through runtime usage
        raise RuntimeError(
            "The Jano MCP server requires the optional MCP dependency. "
            "Install it with `pip install \"jano[mcp]\"` in a Python 3.10+ environment."
        ) from exc

    mcp = FastMCP(
        "Jano",
        instructions=(
            "Jano exposes temporal planning and simulation tools for time-aware "
            "machine learning evaluation. Prefer planning before materializing folds "
            "when the user wants to inspect iteration geometry or exclude date windows."
        ),
    )

    @mcp.tool()
    def preview_local_dataset(
        dataset_path: str,
        dataset_format: str = "auto",
        sample_rows: int = 5,
    ) -> dict:
        """Preview a local tabular dataset before building temporal policies.

        Args:
            dataset_path: Local path to a CSV, Parquet or ZIP-with-CSV dataset.
            dataset_format: Explicit format or ``"auto"``.
            sample_rows: Number of rows to include in the preview.
        """
        return preview_dataset(
            dataset_path,
            dataset_format=dataset_format,
            sample_rows=sample_rows,
        )

    @mcp.tool()
    def plan_walk_forward_simulation(
        dataset_path: str,
        partition: dict,
        step: str,
        time_col: str,
        strategy: str = "rolling",
        allow_partial: bool = False,
        engine: str = "auto",
        start_at: str | None = None,
        end_at: str | None = None,
        max_folds: int | None = None,
        dataset_format: str = "auto",
        order_col: str | None = None,
        train_time_col: str | None = None,
        validation_time_col: str | None = None,
        test_time_col: str | None = None,
        title: str | None = None,
        preview_rows: int = 20,
    ) -> dict:
        """Precompute a walk-forward plan and return fold boundaries plus row counts.

        Args:
            dataset_path: Local path to a CSV, Parquet or ZIP-with-CSV dataset.
            partition: Object accepted by ``TemporalPartitionSpec``. Example:
                ``{"layout": "train_test", "train_size": "7D", "test_size": "1D"}``.
            step: Step size such as ``"1D"``.
            time_col: Timeline column used to anchor the simulation.
            strategy: Movement strategy: ``"single"``, ``"rolling"`` or ``"expanding"``.
            allow_partial: Whether to keep a final partial fold.
            engine: Internal partition engine preference: ``"auto"``, ``"pandas"``,
                ``"polars"`` or ``"numpy"``.
            start_at: Optional lower timestamp bound.
            end_at: Optional upper timestamp bound.
            max_folds: Optional maximum number of folds.
            dataset_format: Explicit format or ``"auto"``.
            order_col: Optional column used to sort the dataset.
            train_time_col: Optional timestamp column used to assign train rows.
            validation_time_col: Optional timestamp column used to assign validation rows.
            test_time_col: Optional timestamp column used to assign test rows.
            title: Optional report title.
            preview_rows: Number of planned folds returned in the preview.
        """
        return plan_walk_forward(
            dataset_path,
            partition=partition,
            step=step,
            time_col=time_col,
            strategy=strategy,
            allow_partial=allow_partial,
            engine=engine,
            start_at=start_at,
            end_at=end_at,
            max_folds=max_folds,
            dataset_format=dataset_format,
            order_col=order_col,
            train_time_col=train_time_col,
            validation_time_col=validation_time_col,
            test_time_col=test_time_col,
            title=title,
            preview_rows=preview_rows,
        )

    @mcp.tool()
    def run_walk_forward_simulation(
        dataset_path: str,
        partition: dict,
        step: str,
        time_col: str,
        strategy: str = "rolling",
        allow_partial: bool = False,
        engine: str = "auto",
        start_at: str | None = None,
        end_at: str | None = None,
        max_folds: int | None = None,
        dataset_format: str = "auto",
        order_col: str | None = None,
        train_time_col: str | None = None,
        validation_time_col: str | None = None,
        test_time_col: str | None = None,
        title: str | None = None,
        preview_rows: int = 20,
    ) -> dict:
        """Run a walk-forward simulation and return a compact summary plus HTML.

        Args:
            dataset_path: Local path to a CSV, Parquet or ZIP-with-CSV dataset.
            partition: Object accepted by ``TemporalPartitionSpec``.
            step: Step size such as ``"1D"``.
            time_col: Timeline column used to anchor the simulation.
            strategy: Movement strategy: ``"single"``, ``"rolling"`` or ``"expanding"``.
            allow_partial: Whether to keep a final partial fold.
            engine: Internal partition engine preference: ``"auto"``, ``"pandas"``,
                ``"polars"`` or ``"numpy"``.
            start_at: Optional lower timestamp bound.
            end_at: Optional upper timestamp bound.
            max_folds: Optional maximum number of folds.
            dataset_format: Explicit format or ``"auto"``.
            order_col: Optional column used to sort the dataset.
            train_time_col: Optional timestamp column used to assign train rows.
            validation_time_col: Optional timestamp column used to assign validation rows.
            test_time_col: Optional timestamp column used to assign test rows.
            title: Optional report title.
            preview_rows: Number of summary rows returned in the preview.
        """
        return run_walk_forward(
            dataset_path,
            partition=partition,
            step=step,
            time_col=time_col,
            strategy=strategy,
            allow_partial=allow_partial,
            engine=engine,
            start_at=start_at,
            end_at=end_at,
            max_folds=max_folds,
            dataset_format=dataset_format,
            order_col=order_col,
            train_time_col=train_time_col,
            validation_time_col=validation_time_col,
            test_time_col=test_time_col,
            title=title,
            preview_rows=preview_rows,
        )

    @mcp.tool()
    def run_walk_forward_baseline_model(
        dataset_path: str,
        partition: dict,
        step: str,
        time_col: str,
        target_col: str,
        feature_cols: list[str] | None = None,
        model: str = "mean",
        metrics: dict[str, Any] | None = None,
        retrain: bool | str = "always",
        retrain_interval: int | None = None,
        drift_metric: str = "rmse",
        drift_threshold: float = 0.05,
        drift_baseline: str = "last_retrain",
        drift_relative: bool = True,
        strategy: str = "rolling",
        allow_partial: bool = False,
        engine: str = "auto",
        start_at: str | None = None,
        end_at: str | None = None,
        max_folds: int | None = None,
        dataset_format: str = "auto",
        order_col: str | None = None,
        train_time_col: str | None = None,
        validation_time_col: str | None = None,
        test_time_col: str | None = None,
        include_predictions: bool = False,
        preview_rows: int = 20,
        prediction_preview_rows: int = 20,
    ) -> dict:
        """Run a simple baseline model over walk-forward folds.

        Args:
            dataset_path: Local path to a CSV, Parquet or ZIP-with-CSV dataset.
            partition: Object accepted by ``TemporalPartitionSpec``.
            step: Step size such as ``"1D"``.
            time_col: Timeline column used to anchor the simulation.
            target_col: Target column to evaluate.
            feature_cols: Optional feature columns.
            model: ``"mean"`` for numeric regression or ``"majority_class"`` for
                classification.
            metrics: Python-only mapping of metric names to callables; MCP JSON clients cannot pass callables.
            retrain: ``"always"``, ``"never"``, ``"periodic"``, ``"on_drift"``,
                ``True`` or ``False``.
            retrain_interval: Fold interval required by ``retrain="periodic"``.
            drift_metric: Metric monitored by ``retrain="on_drift"``.
            drift_threshold: Drift threshold.
            drift_baseline: Drift baseline reference.
            drift_relative: Whether drift threshold is relative or absolute.
            strategy: Movement strategy: ``"single"``, ``"rolling"`` or ``"expanding"``.
            allow_partial: Whether to keep a final partial fold.
            engine: Internal partition engine preference.
            start_at: Optional lower timestamp bound.
            end_at: Optional upper timestamp bound.
            max_folds: Optional maximum number of folds.
            dataset_format: Explicit format or ``"auto"``.
            order_col: Optional column used to sort the dataset.
            train_time_col: Optional timestamp column used to assign train rows.
            validation_time_col: Optional timestamp column used to assign validation rows.
            test_time_col: Optional timestamp column used to assign test rows.
            include_predictions: Whether to return a bounded prediction preview.
            preview_rows: Number of fold/metric rows returned in previews.
            prediction_preview_rows: Number of prediction rows returned when requested.
        """
        return run_walk_forward_baseline(
            dataset_path,
            partition=partition,
            step=step,
            time_col=time_col,
            target_col=target_col,
            feature_cols=feature_cols,
            model=model,
            metrics=metrics,
            retrain=retrain,
            retrain_interval=retrain_interval,
            drift_metric=drift_metric,
            drift_threshold=drift_threshold,
            drift_baseline=drift_baseline,
            drift_relative=drift_relative,
            strategy=strategy,
            allow_partial=allow_partial,
            engine=engine,
            start_at=start_at,
            end_at=end_at,
            max_folds=max_folds,
            dataset_format=dataset_format,
            order_col=order_col,
            train_time_col=train_time_col,
            validation_time_col=validation_time_col,
            test_time_col=test_time_col,
            include_predictions=include_predictions,
            preview_rows=preview_rows,
            prediction_preview_rows=prediction_preview_rows,
        )

    @mcp.tool()
    def compare_retrain_policy_baselines(
        dataset_path: str,
        partition: dict,
        step: str,
        time_col: str,
        target_col: str,
        feature_cols: list[str] | None = None,
        model: str = "mean",
        metrics: dict[str, Any] | None = None,
        policies: list[dict] | None = None,
        strategy: str = "rolling",
        allow_partial: bool = False,
        engine: str = "auto",
        start_at: str | None = None,
        end_at: str | None = None,
        max_folds: int | None = None,
        dataset_format: str = "auto",
        order_col: str | None = None,
        train_time_col: str | None = None,
        validation_time_col: str | None = None,
        test_time_col: str | None = None,
        preview_rows: int = 20,
    ) -> dict:
        """Compare built-in baseline performance across retraining policies."""
        return compare_retrain_policies(
            dataset_path,
            partition=partition,
            step=step,
            time_col=time_col,
            target_col=target_col,
            feature_cols=feature_cols,
            model=model,
            metrics=metrics,
            policies=policies,
            strategy=strategy,
            allow_partial=allow_partial,
            engine=engine,
            start_at=start_at,
            end_at=end_at,
            max_folds=max_folds,
            dataset_format=dataset_format,
            order_col=order_col,
            train_time_col=train_time_col,
            validation_time_col=validation_time_col,
            test_time_col=test_time_col,
            preview_rows=preview_rows,
        )

    @mcp.tool()
    def find_train_history_window_baseline(
        dataset_path: str,
        time_col: str,
        cutoff: str,
        train_sizes: list[str],
        test_size: str,
        target_col: str,
        feature_cols: list[str] | None = None,
        model: str = "mean",
        metrics: dict[str, Any] | None = None,
        metric: str = "rmse",
        tolerance: float = 0.0,
        relative: bool = True,
        gap_before_test: str | None = None,
        dataset_format: str = "auto",
        order_col: str | None = None,
        train_time_col: str | None = None,
        validation_time_col: str | None = None,
        test_time_col: str | None = None,
        preview_rows: int = 20,
    ) -> dict:
        """Find the smallest train-history window close to the best baseline score."""
        return find_train_history_window(
            dataset_path,
            time_col=time_col,
            cutoff=cutoff,
            train_sizes=train_sizes,
            test_size=test_size,
            target_col=target_col,
            feature_cols=feature_cols,
            model=model,
            metrics=metrics,
            metric=metric,
            tolerance=tolerance,
            relative=relative,
            gap_before_test=gap_before_test,
            dataset_format=dataset_format,
            order_col=order_col,
            train_time_col=train_time_col,
            validation_time_col=validation_time_col,
            test_time_col=test_time_col,
            preview_rows=preview_rows,
        )

    @mcp.tool()
    def monitor_decay_baseline(
        dataset_path: str,
        time_col: str,
        cutoff: str,
        train_size: str,
        test_size: str,
        step: str,
        target_col: str,
        feature_cols: list[str] | None = None,
        model: str = "mean",
        metrics: dict[str, Any] | None = None,
        metric: str = "rmse",
        threshold: float = 0.1,
        baseline: str | float = "first",
        relative: bool = True,
        gap_before_test: str | None = None,
        max_windows: int | None = None,
        dataset_format: str = "auto",
        order_col: str | None = None,
        train_time_col: str | None = None,
        validation_time_col: str | None = None,
        test_time_col: str | None = None,
        preview_rows: int = 20,
    ) -> dict:
        """Monitor when a fixed train window starts degrading on moving test windows."""
        return monitor_decay(
            dataset_path,
            time_col=time_col,
            cutoff=cutoff,
            train_size=train_size,
            test_size=test_size,
            step=step,
            target_col=target_col,
            feature_cols=feature_cols,
            model=model,
            metrics=metrics,
            metric=metric,
            threshold=threshold,
            baseline=baseline,
            relative=relative,
            gap_before_test=gap_before_test,
            max_windows=max_windows,
            dataset_format=dataset_format,
            order_col=order_col,
            train_time_col=train_time_col,
            validation_time_col=validation_time_col,
            test_time_col=test_time_col,
            preview_rows=preview_rows,
        )

    return mcp


def main() -> None:
    """Run Jano as a local stdio MCP server."""

    build_server().run()


if __name__ == "__main__":
    main()
