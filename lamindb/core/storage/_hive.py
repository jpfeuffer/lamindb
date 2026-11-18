from __future__ import annotations

from typing import TYPE_CHECKING

from lamindb.errors import ValidationError

if TYPE_CHECKING:
    from pyarrow import Scalar


def hive_directory(column: str, value: Scalar) -> str:
    import pyarrow as pa
    import pyarrow.dataset as ds

    if column.startswith((".", "_")) or any(c in column for c in "/\\=\x00%?#"):
        raise ValidationError(f"Invalid Hive partition column name: {column!r}")
    if pa.types.is_dictionary(value.type):
        value = pa.scalar(value.as_py(), type=value.type.value_type)
    partitioning = ds.HivePartitioning(pa.schema([(column, value.type)]))
    field = ds.field(column)
    expression = (field == value) if value.is_valid else field.is_null()
    try:
        directory, _ = partitioning.format(expression)
        keys = ds.get_partition_keys(partitioning.parse(f"{directory}/part.parquet"))
        round_tripped = pa.scalar(keys[column], type=value.type)
    except (pa.ArrowException, KeyError) as error:
        raise ValidationError(
            f"Partition value for {column!r} cannot be represented in a Hive path"
        ) from error
    if not directory or not value.equals(round_tripped):
        raise ValidationError(
            f"Partition value for {column!r} cannot round-trip in a Hive path"
        )
    return directory
