from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from lamindb_setup import logger

from lamindb.core._compat import (
    is_polars_dataframe,
    with_package_obj,
)

if TYPE_CHECKING:
    from pandas import DataFrame
    from polars import DataFrame as PolarsDataFrame
    from polars import LazyFrame as PolarsLazyFrame

    from .types import ScverseDataStructures

    SupportedDataTypes: TypeAlias = (
        DataFrame | PolarsDataFrame | PolarsLazyFrame | ScverseDataStructures
    )
else:
    SupportedDataTypes: TypeAlias = Any


def infer_suffix(
    dmem: SupportedDataTypes, format: str | dict[str, Any] | None = None
) -> str:
    """Infer LaminDB storage file suffix from a data object."""
    if is_polars_dataframe(dmem):
        return _infer_dataframe_suffix(format)
    has_anndata, anndata_suffix = with_package_obj(
        dmem,
        "AnnData",
        "anndata",
        lambda obj: _infer_anndata_suffix(format),
    )
    if has_anndata:
        return anndata_suffix

    has_dataframe, dataframe_suffix = with_package_obj(
        dmem,
        "DataFrame",
        "pandas",
        lambda obj: _infer_dataframe_suffix(format),
    )
    if has_dataframe:
        return dataframe_suffix

    if with_package_obj(
        dmem,
        "MuData",
        "mudata",
        lambda obj: True,  # Just checking type, not calling any method
    )[0]:
        return ".h5mu"

    has_spatialdata, spatialdata_suffix = with_package_obj(
        dmem,
        "SpatialData",
        "spatialdata",
        lambda obj: _infer_spatialdata_suffix(format),
    )
    if has_spatialdata:
        return spatialdata_suffix
    else:
        raise NotImplementedError


def _infer_anndata_suffix(format: str | dict[str, Any] | None) -> str:
    assert not isinstance(format, dict)  # noqa: S101
    if format is not None:
        # should be `.h5ad`, `.`zarr`, or `.anndata.zarr`
        if format not in {"h5ad", "zarr", "anndata.zarr"}:
            raise ValueError(
                "Error when specifying AnnData storage format, it should be"
                f" 'h5ad', 'zarr', not '{format}'. Check 'format'"
                " or the suffix of 'key'."
            )
        return "." + format
    return ".h5ad"


def _infer_dataframe_suffix(format: str | dict[str, Any] | None) -> str:
    if isinstance(format, str):
        if format == ".csv":
            return ".csv"
    elif isinstance(format, dict):
        if format.get("suffix") == ".csv":
            return ".csv"
    return ".parquet"


def _infer_spatialdata_suffix(format: str | dict[str, Any] | None) -> str:
    if format is None:
        return ".zarr"
    if isinstance(format, str) and format in {"spatialdata.zarr", "zarr"}:
        return f".{format}"
    raise ValueError(
        "Error when specifying SpatialData storage format, it should be"
        f" 'zarr', 'spatialdata.zarr', not '{format}'. Check 'format'"
        " or the suffix of 'key'."
    )


# for types below note that local UPaths are subclasses of Path
# Path(UPath(...)) properly coerces local UPaths and throws an error for cloud UPaths


def write_to_disk(dmem: SupportedDataTypes, filepath: Path | str, **kwargs) -> None:
    """Writes the passed in memory data to disk to a specified path."""
    sorting_keys = kwargs.pop("_lamindb_sorting_columns", None)
    if is_polars_dataframe(dmem):
        import polars as pl

        if sorting_keys and Path(filepath).suffix != ".csv":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                logger.warning(
                    "PyArrow is not installed; writing Parquet without sorting_columns metadata"
                )
                if isinstance(dmem, pl.LazyFrame):
                    dmem.sink_parquet(filepath, **kwargs)
                else:
                    dmem.write_parquet(filepath, **kwargs)
                return

            ordering = [
                (name, "ascending" if ascending else "descending")
                for name, ascending in sorting_keys
            ]
            if isinstance(dmem, pl.LazyFrame):
                with tempfile.NamedTemporaryFile(
                    dir=Path(filepath).parent, suffix=".parquet", delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                row_group_size = kwargs.get("row_group_size") or 65536
                try:
                    dmem.sink_parquet(temporary_path, **kwargs)
                    parquet_file = pq.ParquetFile(temporary_path)
                    sorting_columns = pq.SortingColumn.from_ordering(
                        parquet_file.schema_arrow,
                        ordering,
                        null_placement="at_end",
                    )
                    writer_kwargs = _pyarrow_writer_kwargs(kwargs)
                    writer_kwargs.setdefault("compression", "zstd")
                    with pq.ParquetWriter(
                        filepath,
                        parquet_file.schema_arrow,
                        sorting_columns=sorting_columns,
                        **writer_kwargs,
                    ) as writer:
                        for batch in parquet_file.iter_batches(
                            batch_size=row_group_size
                        ):
                            writer.write_batch(batch)
                finally:
                    temporary_path.unlink(missing_ok=True)
            else:
                table = dmem.to_arrow()
                sorting_columns = pq.SortingColumn.from_ordering(
                    table.schema, ordering, null_placement="at_end"
                )
                writer_kwargs = _pyarrow_writer_kwargs(kwargs)
                writer_kwargs.setdefault("compression", "zstd")
                pq.write_table(
                    table,
                    filepath,
                    sorting_columns=sorting_columns,
                    **writer_kwargs,
                )
            return
        if isinstance(dmem, pl.LazyFrame):
            if Path(filepath).suffix == ".csv":
                dmem.sink_csv(filepath, **kwargs)
            else:
                dmem.sink_parquet(filepath, **kwargs)
            return
        if Path(filepath).suffix == ".csv":
            dmem.write_csv(filepath, **kwargs)
        else:
            dmem.write_parquet(filepath, **kwargs)
        return
    if with_package_obj(
        dmem,
        "AnnData",
        "anndata",
        lambda obj: _write_anndata(obj, filepath, **kwargs),
    )[0]:
        return

    if with_package_obj(
        dmem,
        "DataFrame",
        "pandas",
        lambda obj: _write_dataframe(
            obj,
            filepath,
            sorting_keys=sorting_keys if Path(filepath).suffix != ".csv" else None,
            **kwargs,
        ),
    )[0]:
        return

    if with_package_obj(dmem, "MuData", "mudata", lambda obj: obj.write(filepath))[0]:
        return

    if with_package_obj(
        dmem,
        "SpatialData",
        "spatialdata",
        lambda obj: obj.write(filepath, overwrite=True),
    )[0]:
        return

    raise NotImplementedError


def _pyarrow_writer_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate the common Polars parquet writer options to PyArrow options."""
    options = dict(kwargs.pop("pyarrow_options", {}) or {})
    kwargs.pop("use_pyarrow", None)
    kwargs.pop("row_group_size", None)
    kwargs.pop("maintain_order", None)
    if "statistics" in kwargs:
        options["write_statistics"] = kwargs.pop("statistics")
    options.update(kwargs)
    return options


def _write_anndata(dmem: Any, filepath: Path | str, **kwargs) -> None:
    suffix = Path(filepath).suffix
    if suffix == ".h5ad":
        dmem.write_h5ad(filepath, **kwargs)
        return
    elif suffix == ".zarr":
        dmem.write_zarr(filepath, **kwargs)
        return
    else:
        raise NotImplementedError


def _write_dataframe(
    dmem: Any,
    filepath: Path | str,
    *,
    sorting_keys: list[tuple[str, bool]] | None = None,
    **kwargs,
) -> None:
    suffix = Path(filepath).suffix
    if suffix == ".csv":
        dmem.to_csv(filepath, **kwargs)
        return
    if sorting_keys:
        import pyarrow as pa
        import pyarrow.parquet as pq

        arrow_schema = pa.Schema.from_pandas(dmem, preserve_index=kwargs.get("index"))
        sorting_columns = pq.SortingColumn.from_ordering(
            arrow_schema,
            [
                (name, "ascending" if ascending else "descending")
                for name, ascending in sorting_keys
            ],
            null_placement="at_end",
        )
        kwargs["sorting_columns"] = sorting_columns
    dmem.to_parquet(filepath, **kwargs)
