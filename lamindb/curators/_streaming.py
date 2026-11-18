from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lamindb.errors import ValidationError

if TYPE_CHECKING:
    from lamindb.models.schema import Schema


class StreamChecks:
    """Validate rules that a dataframe curator cannot check within one batch."""

    def __init__(self, schema: Schema, *, is_polars: bool):
        self.is_polars = is_polars
        self.sort_keys = schema.sorted_by
        self.previous_end: Any = None
        index = schema.index
        index_uid = index.uid if index is not None else None
        self.unique = [
            [feature.name]
            for feature in schema.members
            if feature.unique and feature.uid != index_uid
        ]
        if schema.unique:
            self.unique.append(schema.unique)
        self.unique_columns = list(dict.fromkeys(c for key in self.unique for c in key))
        self.projections: list[Any] = []
        self.index_projections: list[Any] = []
        self.unique_index = not is_polars and index is not None and index.unique

    def _concat(self, frames: list[Any]) -> Any:
        if self.is_polars:
            import polars as pl

            return pl.concat(frames)
        import pandas as pd

        return pd.concat(frames, ignore_index=True)

    def add(self, frame: Any) -> None:
        from .core import _is_sorted

        if self.sort_keys:
            columns = [name for name, _ in self.sort_keys]
            keys = frame.select(columns) if self.is_polars else frame[columns]
            start = keys.head(1)
            if self.is_polars:
                import polars as pl

                end = pl.DataFrame(
                    [pl.Series(s.name, [s[-1]], dtype=s.dtype) for s in keys]
                )
                if (
                    self.previous_end is not None
                    and start.schema != self.previous_end.schema
                ):
                    raise ValidationError("Sort key types differ between batches")
            else:
                start = start.copy(deep=True)
                end = keys.tail(1).copy(deep=True)
                if self.previous_end is not None and not start.dtypes.equals(
                    self.previous_end.dtypes
                ):
                    raise ValidationError("Sort key types differ between batches")
            if self.previous_end is not None and not _is_sorted(
                self._concat([self.previous_end, start]), self.sort_keys, self.is_polars
            ):
                raise ValidationError(
                    "Dataframe is not sorted across the batch boundary"
                )
            self.previous_end = end
        if self.unique_columns:
            columns = [c for c in self.unique_columns if c in frame.columns]
            if columns:
                self.projections.append(
                    frame.select(columns)
                    if self.is_polars
                    else frame[columns].reset_index(drop=True).copy(deep=True)
                )
        if self.unique_index:
            self.index_projections.append(frame.index.to_series().copy(deep=True))

    def finish(self) -> None:
        if self.projections:
            combined = self._concat(self.projections)
            for columns in self.unique:
                if not all(c in combined.columns for c in columns):
                    continue
                duplicate = (
                    combined.select(columns).is_duplicated().any()
                    if self.is_polars
                    else combined.duplicated(subset=columns).any()
                )
                if duplicate:
                    raise ValidationError(
                        f"Duplicate values across batches for {columns}"
                    )
        if (
            self.index_projections
            and self._concat(self.index_projections).duplicated().any()
        ):
            raise ValidationError("Duplicate index values across batches")
