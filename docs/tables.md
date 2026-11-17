---
execute_via: python
---

# Query tables in storage

This guide walks through querying tabular datasets stored in parquet and related file formats.
Queries stream directly from disk or cloud storage with PyArrow, Polars or DuckDB.

For an in-depth example and a complex dataset, read this blog post: [blog.lamin.ai/1000genomes](https://blog.lamin.ai/1000genomes).

```python
import lamindb as ln

db = ln.DB("laminlabs/lamindata")
```

## Open a dataset for queries

To open an **artifact** as a [pyarrow dataset](https://arrow.apache.org/docs/python/dataset.html) call `.open()` with default arguments:

```python
artifact = db.Artifact.filter(
    key__startswith="example_datasets/small", suffix=".parquet"
).first()
dataset = artifact.open()
dataset
```

To open a **collection** (a sharded dataset), call:

```python
collection = db.Collection.get(key="sharded_parquet_collection")
dataset = collection.open()
dataset
```

We recommended to use {class}`~lamindb.Collection` to manage large-scale datasets as this gives you typical lakehouse guarantees. For more background, read {doc}`architecture`.

### Create a schema-sharded collection

Declare the sharding column once, in the shared schema. It must be a scalar
column feature, included in the schema's required features (`minimal_set=True`,
the default), not an optional feature or a dataframe index. Nullable partition
features are supported: nulls form their own group.

```{code-block} python
import pandas as pd
import lamindb as ln

batch = ln.Feature(name="batch", dtype=str, nullable=True).save()
measurement = ln.Feature(name="measurement", dtype=float).save()
schema = ln.Schema(
    [batch, measurement],
    otype="DataFrame",
    partition_by="batch",
).save()

df = pd.DataFrame({
    "batch": ["a", "b", "a", None],
    "measurement": [1.0, 2.0, 3.0, 4.0],
})
collection = ln.Collection.from_dataframe(
    df,
    key="measurements/by-batch",
    schema=schema,
    max_rows_per_shard=1_000_000,
).save()
```

`Collection.from_dataframe` validates the whole input, including uniqueness and
ordering rules, before grouping. It uses pandas `groupby` or Polars `partition_by`
and writes ordinary single-file Parquet artifacts, each validated against the
common schema. Every shard retains the partition column and contains only one
distinct value of it. Observed categorical values (including Polars Enum) are
supported; unused categories do not produce empty shards. Nulls are not dropped.
Values are never encoded in paths: shard keys are
`measurements/by-batch/part-00000.parquet`, etc., not Hive partitions.

A group larger than `max_rows_per_shard` is sliced into multiple artifacts, each
with the same partition value. The default limit is one million rows per file;
it is not a byte-size or peak-memory guarantee. Group order follows first
appearance, and row order within each group is preserved, but the original
interleaving of groups is not. No sorting is performed. Existing
`schema.sorted_by` validation remains active; optional
`write_parquet_sorting_metadata=True` writes verified per-shard metadata, not
a global sorting guarantee for the collection.

Pandas DataFrames, Polars DataFrames, and Polars LazyFrames are accepted.
LazyFrames are collected once before validation and grouping: this constructor
is **not** an out-of-core ingestion API. Empty inputs raise an error rather
than creating an empty collection.

The constructor saves member artifacts and returns an unsaved collection.
Calling `.save()` publishes a versioned collection; repeating the key with
changed content creates a new version, following normal Collection conventions.
Storage writes are not transactional: a failure may leave saved member artifacts,
but no partial collection is published. Partition configuration persists in
the schema's existing auxiliary metadata and contributes to its hash; no
database migration is needed.

Open or filter the collection as one logical table using the existing APIs:

```{code-block} python
import pyarrow.compute as pc

table = collection.open().to_table(filter=pc.field("batch") == "a")

import polars as pl

with collection.open(engine="polars") as lazy_df:
    table = lazy_df.filter(pl.col("batch").is_null()).collect()
```

<!-- #region -->

:::{dropdown} Open sets of artifacts ad hoc

You can also filter artifacts by a prefix and open the resulting `ArtifactSet`:

```python
dataset = db.Artifact.filter(
    key__startswith="example_datasets/small", suffix=".parquet", is_latest=True
).open()
dataset
```

:::

<!-- #endregion -->

## Queries

Filters push down into Parquet row groups — only matching data is read from storage, not the whole file.

```python
import pyarrow.compute as pc

dataset.to_table(filter=pc.field("cell_type").is_valid()).to_pandas().head(5)
```

You can build up from there. Materialize the filtered result once, then compute against it in memory — for example, count rows per cell type without re-reading storage:

```python
counts = (
    dataset.to_table(filter=pc.field("cell_type").is_valid())
    .group_by("cell_type")
    .aggregate([("cell_type", "count")])
    .to_pandas()
)
counts.head()
```

These filters run the same way on every engine below. For heavier patterns — schema evolution, time travel, appends, and concurrent writers — the [LaminDB lakehouse benchmark](https://lamin.ai/blog/lakehouse-benchmark) walks through all six operations end to end across the five engines on a shared genomics dataset.

## Choosing a query engine

By default `Artifact.open()` and `Collection.open()` use `pyarrow` to lazily open dataframes. `polars` can also be used by passing `engine="polars"`. Note also that `.open(engine="polars")` returns a context manager with [LazyFrame](https://docs.pola.rs/api/python/stable/reference/lazyframe/index.html).

```python
with collection.open(engine="polars") as lazy_df:
    display(lazy_df.collect().to_pandas())
```

LaminDB artifacts are plain files or folders on S3, so any engine that reads the format works. DuckDB and Iceberg read Parquet. A `.lance` artifact opens with `artifact.lance.open()` as a LanceDB table. You pick the engine based on your query pattern.

::::::{tab-set}
:::::{tab-item} PyArrow
`.open()` returns a lazy PyArrow dataset backed by S3. Filters push down into Parquet row groups.

```{code-block} python
import pyarrow.compute as pc

dataset = collection.open()
result = dataset.to_table(filter=pc.field("cell_type").is_valid()).to_pandas()
```

:::::

:::::{tab-item} Polars
`.open(engine="polars")` returns a context manager yielding a Polars LazyFrame backed by S3. No data is read until `.collect()` is called.

```{code-block} python
import polars as pl

with collection.open(engine="polars") as lazy_df:
    result = lazy_df.filter(pl.col("cell_type").is_not_null()).collect()
```

:::::

:::::{tab-item} DuckDB
DuckDB reads Parquet files directly via `read_parquet()`. Use `.cache()` to get a local path, or register a view over S3 paths via `httpfs` for a collection:

```{code-block} python
import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("CREATE OR REPLACE SECRET s3 (TYPE s3, PROVIDER credential_chain);")

s3_paths = [str(a.path) for a in collection.ordered_artifacts.all()]
con.execute(f"CREATE OR REPLACE VIEW data AS SELECT * FROM read_parquet({s3_paths})")

result = con.execute("SELECT * FROM data WHERE cell_type IS NOT NULL").df()
```

:::::

:::::{tab-item} Iceberg
Iceberg requires a one-time ingestion into a catalog-managed table on S3. After ingestion you get native partition pruning, metadata-only schema evolution, and snapshot-based time travel.

```{code-block} python
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.expressions import NotNull

arrow = collection.open().to_table()

catalog = SqlCatalog(
    "local", uri="sqlite:///iceberg_catalog.db", warehouse="/tmp/iceberg_warehouse"
)
if not catalog.namespace_exists("demo"):
    catalog.create_namespace("demo")
table = catalog.create_table("demo.data", schema=arrow.schema)
table.append(arrow)

result = table.scan(row_filter=NotNull("cell_type")).to_arrow().to_pandas()
```

:::::

:::::{tab-item} LanceDB
A `.lance` artifact is one LanceDB table. Register an existing directory with `from_lance()`. `artifact.lance.open()` returns that table, and filters run in Lance.

```{code-block} python
import pandas as pd

df = pd.DataFrame(
    {
        "cell_type": ["T cell", None, "B cell"],
        "n_genes": [1000, 800, 1200],
    }
)
artifact = ln.Artifact.from_dataframe(df, key="example_datasets/small.lance").save()
table = artifact.lance.open()

result = table.search().where("cell_type IS NOT NULL", prefilter=True).limit(5).to_pandas()
```

:::::
::::::
