from __future__ import annotations

from .mcp_tools import plan_walk_forward, preview_dataset, run_walk_forward


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

    return mcp


def main() -> None:
    """Run Jano as a local stdio MCP server."""

    build_server().run()


if __name__ == "__main__":
    main()
