import lamindb as ln
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from django.db import transaction
from lamindb.errors import InvalidArgument, ValidationError


@pytest.fixture(autouse=True)
def isolate_records():
    with transaction.atomic():
        yield
        transaction.set_rollback(True)


def make_schema(**kwargs):
    features = [
        ln.Feature(name="shard_group", dtype=str).save(),
        ln.Feature(name="shard_value", dtype=int).save(),
    ]
    return ln.Schema(features, partition_by="shard_group", **kwargs).save()


@pytest.fixture(params=["pandas", "polars", "lazyframe"])
def backend(request):
    def convert(df):
        if request.param == "pandas":
            return df
        pl = pytest.importorskip("polars")
        result = pl.from_pandas(df)
        return result.lazy() if request.param == "lazyframe" else result

    return convert


@pytest.mark.parametrize("categorical", [False, True])
def test_grouping_and_open(backend, categorical):
    schema = make_schema(maximal_set=True)
    df = pd.DataFrame(
        {
            "shard_group": ["b", None, "a", "b", "a", "b", None],
            "shard_value": list(range(7)),
        }
    )
    if categorical:
        df["shard_group"] = pd.Categorical(
            df.shard_group, categories=["unused", "a", "b"]
        )
    collection = ln.Collection.from_dataframe(
        backend(df), key="sharded/example", schema=schema, max_rows_per_shard=2
    )
    assert collection._state.adding
    assert all(not artifact._state.adding for artifact in collection._artifacts)
    collection.save()
    collection = ln.Collection.get(collection.uid)
    assert collection.schema == schema
    artifacts = list(collection.ordered_artifacts)
    assert len(artifacts) == 4
    assert [a.n_observations for a in artifacts] == [2, 1, 2, 2]
    assert all(a.schema == schema and a.suffix == ".parquet" for a in artifacts)
    for i, artifact in enumerate(artifacts):
        assert artifact.path.is_file()
        assert artifact.key == f"sharded/example/part-{i:05d}.parquet"
        shard = artifact.load()
        assert shard.shard_group.nunique(dropna=False) == 1
        assert set(shard.columns) == {"shard_group", "shard_value"}
        assert pq.ParquetFile(artifact.path).metadata.row_group(0).sorting_columns == ()
    result = collection.open().to_table().to_pandas()
    assert result.shard_value.tolist() == [0, 3, 5, 1, 6, 2, 4]
    pd.testing.assert_frame_equal(
        result.sort_values("shard_value")
        .reset_index(drop=True)
        .astype({"shard_group": "object"}),
        df.astype({"shard_group": "object"}),
    )
    assert collection.artifacts.all().open().count_rows() == len(df)
    if find_spec("polars") is not None:
        import polars as pl

        with collection.open(engine="polars") as lazy:
            assert lazy.filter(pl.col("shard_group").is_null()).collect().height == 2
    collection.verify_schema()


def test_polars_enum():
    pl = pytest.importorskip("polars")
    schema = make_schema()
    df = pl.DataFrame(
        {
            "shard_group": pl.Series(
                ["b", None, "a", "b"], dtype=pl.Enum(["unused", "a", "b"])
            ),
            "shard_value": [1, 2, 3, 4],
        }
    )
    collection = ln.Collection.from_dataframe(
        df, key="sharded/enum", schema=schema
    ).save()
    assert collection.artifacts.count() == 3
    assert collection.open().count_rows() == 4
    with collection.open(engine="polars") as lazy:
        assert lazy.collect().schema["shard_group"] == df.schema["shard_group"]


@pytest.mark.parametrize("layout", ["plain", "hive"])
def test_streaming_layout(backend, layout):
    schema = make_schema(unique=["shard_value"], sorted_by=["shard_value"])
    frame = pd.DataFrame(
        {"shard_group": ["a/b %", None, "a/b %", "b"], "shard_value": range(4)}
    )
    if type(backend(frame)).__name__ == "LazyFrame":
        pytest.skip("Batches are eager")
    key = f"streaming/{layout}"

    def batches():
        yield backend(frame.iloc[:2])
        assert ln.Artifact.filter(key__startswith=f"{key}/").count() == 2
        yield backend(frame.iloc[2:])

    collection = ln.Collection.from_dataframe(
        batches(), key=key, schema=schema, layout=layout
    ).save()
    paths = [a.path for a in collection.ordered_artifacts]
    assert len(set(paths)) == 4
    result = collection.open().to_table().sort_by("shard_value")
    assert result["shard_group"].to_pylist() == ["a/b %", None, "a/b %", "b"]
    if layout == "hive":
        root = paths[0].parent.parent
        assert set(root.rglob("*.parquet")) == set(paths)
        assert paths[0].parent.name == "shard_group=a%2Fb%20%25"
        assert paths[1].parent.name == "shard_group=__HIVE_DEFAULT_PARTITION__"
        field = pq.read_schema(paths[0]).field("shard_group")
        hive = ds.dataset(
            root,
            format="parquet",
            partitioning=ds.partitioning(pa.schema([field]), flavor="hive"),
        )
        assert hive.to_table().sort_by("shard_value").equals(result)
        assert (
            len(list(hive.get_fragments(filter=ds.field("shard_group") == "a/b %")))
            == 2
        )


@pytest.mark.parametrize(
    "rule,match",
    [("unique", "Duplicate values across"), ("sorted_by", "not sorted across")],
)
def test_streaming_cross_batch_validation(backend, rule, match):
    schema = make_schema(**{rule: ["shard_value"]})
    first = backend(pd.DataFrame({"shard_group": ["a"], "shard_value": [2]}))
    second = backend(
        pd.DataFrame(
            {"shard_group": ["b"], "shard_value": [2 if rule == "unique" else 1]}
        )
    )
    if type(first).__name__ == "LazyFrame":
        pytest.skip("Batches are eager")
    with pytest.raises(ValidationError, match=match):
        ln.Collection.from_dataframe(iter([first, second]), key="bad", schema=schema)
    assert not ln.Collection.filter(key="bad").exists()


def test_hive_versions_and_layout_identity():
    schema = make_schema()
    frame = pd.DataFrame({"shard_group": ["a"], "shard_value": [1]})
    plain = ln.Collection.from_dataframe(frame, key="managed", schema=schema).save()
    hive = ln.Collection.from_dataframe(
        frame, key="managed", schema=schema, layout="hive"
    ).save()
    updated = ln.Collection.from_dataframe(
        frame.assign(shard_value=2), key="managed", schema=schema, layout="hive"
    ).save()
    assert len({plain.uid, hive.uid, updated.uid}) == 3
    root = hive.ordered_artifacts.all()[0].path.parent.parent
    assert updated.ordered_artifacts.all()[0].path.parent.parent != root
    assert hive.open().to_table()["shard_value"].to_pylist() == [1]
    assert updated.open().to_table()["shard_value"].to_pylist() == [2]


@pytest.mark.parametrize("dtype", ["Int64", "category", "enum"])
def test_hive_partition_types(dtype):
    values = [2**40, None, 2**40] if dtype == "Int64" else ["a", None, "a"]
    group = ln.Feature(
        name="shard_group", dtype=int if dtype == "Int64" else str
    ).save()
    value = ln.Feature(name="shard_value", dtype=int).save()
    schema = ln.Schema([group, value], partition_by=group.name).save()
    if dtype == "enum":
        pl = pytest.importorskip("polars")
        frame = pl.DataFrame(
            {
                group.name: pl.Series(values, dtype=pl.Enum(["a", "unused"])),
                value.name: range(3),
            }
        )
    else:
        frame = pd.DataFrame(
            {group.name: pd.Series(values, dtype=dtype), value.name: range(3)}
        )
    collection = ln.Collection.from_dataframe(
        frame, key=f"managed/{dtype}", schema=schema, layout="hive"
    ).save()
    shard = collection.ordered_artifacts.all()[0].path
    field = pq.read_schema(shard).field(group.name)
    if dtype == "Int64":
        assert field.type == pa.int64()
    partitioning = ds.partitioning(
        pa.schema([field]), flavor="hive", dictionaries="infer"
    )
    table = ds.dataset(
        shard.parent.parent, format="parquet", partitioning=partitioning
    ).to_table()
    assert dict(
        zip(table[value.name].to_pylist(), table[group.name].to_pylist())
    ) == dict(enumerate(values))
    assert (
        collection.open().to_table().sort_by(value.name)[group.name].to_pylist()
        == values
    )


def test_hive_reserved_null_value():
    schema = make_schema()
    frame = pd.DataFrame(
        {"shard_group": ["__HIVE_DEFAULT_PARTITION__"], "shard_value": [1]}
    )
    with pytest.raises(ValidationError, match="cannot round-trip"):
        ln.Collection.from_dataframe(
            frame, key="managed/null", schema=schema, layout="hive"
        )
    assert not ln.Artifact.filter(key__startswith="managed/null/").exists()


def test_streaming_feature_uniqueness():
    group = ln.Feature(name="shard_group", dtype=str).save()
    value = ln.Feature(name="shard_value", dtype=int, unique=True).save()
    schema = ln.Schema([group, value], partition_by=group.name).save()
    frames = [
        pd.DataFrame({"shard_group": [name], "shard_value": [1]}) for name in ("a", "b")
    ]
    with pytest.raises(ValidationError, match="Duplicate values across"):
        ln.Collection.from_dataframe(frames, key="managed/unique", schema=schema)


def test_schema_partition_config():
    schema = make_schema(unique=["shard_value"], sorted_by=["shard_value"])
    schema.save()
    reloaded = ln.Schema.get(schema.uid)
    assert reloaded.partition_by == "shard_group"
    assert reloaded.unique == ["shard_value"]
    assert reloaded.sorted_by == [("shard_value", True)]
    assert (
        ln.Schema(
            list(schema.members),
            partition_by="shard_group",
            unique=["shard_value"],
            sorted_by=["shard_value"],
        )
        == schema
    )
    assert ln.Schema(list(schema.members)).hash != schema.hash
    assert (
        ln.Schema(list(schema.members), partition_by="shard_value").hash != schema.hash
    )
    reloaded.optionals.add(ln.Feature.get(name="shard_group"))
    with pytest.raises(InvalidArgument, match="required"):
        reloaded.save()


@pytest.mark.parametrize("partition", ["", " ", ["shard_group"], 1])
def test_invalid_partition_config(partition):
    feature = ln.Feature(name="shard_group", dtype=str).save()
    with pytest.raises(InvalidArgument, match="one non-empty"):
        ln.Schema([feature], partition_by=partition)


@pytest.mark.parametrize(
    "config",
    [
        {"partition_by": "absent"},
        {"partition_by": "shard_group", "minimal_set": False},
        {"partition_by": "shard_group", "otype": "AnnData"},
    ],
)
def test_partition_requires_column(config):
    feature = ln.Feature(name="shard_group", dtype=str).save()
    with pytest.raises(InvalidArgument):
        ln.Schema([feature], **config)


def test_partition_rejects_optional_index_and_nested():
    feature = ln.Feature(name="shard_group", dtype=str).save()
    with pytest.raises(InvalidArgument, match="required"):
        ln.Schema([feature.with_config(optional=True)], partition_by="shard_group")
    with pytest.raises(InvalidArgument, match="required"):
        ln.Schema(index=feature, partition_by="shard_group")
    nested = ln.Feature(name="shard_nested", dtype=list[str]).save()
    with pytest.raises(InvalidArgument, match="scalar"):
        ln.Schema([nested], partition_by="shard_nested")
    with pytest.raises(InvalidArgument, match="required"):
        ln.Schema(itype="Feature", partition_by="shard_group")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_invalid_size(limit):
    schema = make_schema()
    with pytest.raises(InvalidArgument, match="positive integer"):
        ln.Collection.from_dataframe(
            pd.DataFrame({"shard_group": ["a"], "shard_value": [1]}),
            key="sharded/invalid",
            schema=schema,
            max_rows_per_shard=limit,
        )
    assert ln.Artifact.filter(key__startswith="sharded/").count() == 0


def test_invalid_constructor_input():
    schema = make_schema()
    df = pd.DataFrame({"shard_group": ["a"], "shard_value": [1]})
    for invalid_schema in (None, ln.Schema(list(schema.members)), "valid_features"):
        with pytest.raises(InvalidArgument, match="saved Schema"):
            ln.Collection.from_dataframe(
                df, key="sharded/invalid", schema=invalid_schema
            )
    unpartitioned = ln.Schema(list(schema.members)).save()
    with pytest.raises(InvalidArgument, match="Schema.partition_by"):
        ln.Collection.from_dataframe(df, key="sharded/invalid", schema=unpartitioned)
    with pytest.raises(TypeError, match="df must"):
        ln.Collection.from_dataframe(
            "data.parquet", key="sharded/invalid", schema=schema
        )
    with pytest.raises(InvalidArgument, match="non-empty"):
        ln.Collection.from_dataframe(df, key="", schema=schema)
    for option in ("partition_by", "partition_cols", "partitioning"):
        with pytest.raises(InvalidArgument, match="Directory partitioning"):
            ln.Collection.from_dataframe(
                df,
                key="sharded/invalid",
                schema=schema,
                parquet_kwargs={option: ["shard_group"]},
            )


def test_empty_input(backend):
    schema = make_schema()
    df = pd.DataFrame(
        {"shard_group": pd.Series(dtype=str), "shard_value": pd.Series(dtype=int)}
    )
    with pytest.raises(InvalidArgument, match="empty dataframe"):
        ln.Collection.from_dataframe(backend(df), key="sharded/empty", schema=schema)
    assert not ln.Artifact.filter(key__startswith="sharded/").exists()


@pytest.mark.parametrize("invalid", ["missing", "null", "dtype", "unique", "regex"])
def test_validation_before_writing(backend, invalid):
    group = ln.Feature(
        name="shard_group", dtype=str, nullable=False, regex="[ab]"
    ).save()
    value = ln.Feature(name="shard_value", dtype=int, unique=True).save()
    schema = ln.Schema([group, value], partition_by="shard_group").save()
    df = pd.DataFrame({"shard_group": ["a", "b"], "shard_value": [1, 2]})
    if invalid == "missing":
        df = df.drop(columns="shard_group")
    elif invalid == "null":
        df.loc[1, "shard_group"] = None
    elif invalid == "dtype":
        df["shard_value"] = ["bad", "type"]
    elif invalid == "regex":
        df["shard_group"] = ["a", "invalid"]
    else:
        df["shard_value"] = [1, 1]
    with pytest.raises(ValidationError):
        ln.Collection.from_dataframe(backend(df), key="sharded/invalid", schema=schema)
    assert not ln.Artifact.filter(key__startswith="sharded/").exists()


def test_coercion_before_grouping(backend):
    group = ln.Feature(name="shard_group", dtype=int, coerce=True).save()
    value = ln.Feature(name="shard_value", dtype=int).save()
    schema = ln.Schema([group, value], partition_by="shard_group").save()
    df = pd.DataFrame({"shard_group": ["01", "1", "2"], "shard_value": [1, 2, 3]})
    collection = ln.Collection.from_dataframe(
        backend(df), key="sharded/coerced", schema=schema
    ).save()
    assert collection.artifacts.count() == 2
    assert collection.open().to_table()["shard_group"].to_pylist() == [1, 1, 2]


def test_null_only(backend):
    schema = make_schema()
    df = pd.DataFrame({"shard_group": [None, None], "shard_value": [1, 2]})
    collection = ln.Collection.from_dataframe(
        backend(df), key="sharded/null", schema=schema
    ).save()
    assert collection.artifacts.count() == 1
    assert collection.open().to_table()["shard_group"].null_count == 2


@pytest.mark.parametrize(
    "dtype,values",
    [
        (int, pd.Series([1, None, 1, 2], dtype="Int64")),
        (bool, pd.Series([True, None, False, True], dtype="boolean")),
        (float, pd.Series([1.0, None, 1.0, 2.0])),
    ],
)
def test_nullable_scalar_partitions(backend, dtype, values):
    group = ln.Feature(name="shard_group", dtype=dtype).save()
    value = ln.Feature(name="shard_value", dtype=int).save()
    schema = ln.Schema([group, value], partition_by="shard_group").save()
    df = pd.DataFrame({"shard_group": values, "shard_value": [1, 2, 3, 4]})
    collection = ln.Collection.from_dataframe(
        backend(df), key="sharded/scalar", schema=schema
    ).save()
    assert collection.artifacts.count() == 3
    assert collection.open().count_rows() == 4
    for artifact in collection.ordered_artifacts:
        assert artifact.load().shard_group.nunique(dropna=False) == 1


def test_schema_unique_across_shards(backend):
    schema = make_schema(unique=["shard_value"])
    df = pd.DataFrame({"shard_group": ["a", "b"], "shard_value": [1, 1]})
    with pytest.raises(ValidationError):
        ln.Collection.from_dataframe(backend(df), key="sharded/unique", schema=schema)
    assert not ln.Artifact.filter(key__startswith="sharded/unique").exists()


def test_duplicate_chunks_and_versions(backend):
    schema = make_schema()
    df = pd.DataFrame({"shard_group": ["a"] * 4, "shard_value": [1] * 4})
    first = ln.Collection.from_dataframe(
        backend(df),
        key="sharded/versioned",
        schema=schema,
        max_rows_per_shard=2,
        parquet_kwargs={"index": False}
        if isinstance(backend(df), pd.DataFrame)
        else {},
    ).save()
    assert first.artifacts.count() == 2
    assert first.open().count_rows() == 4
    repeated = ln.Collection.from_dataframe(
        backend(df),
        key="sharded/versioned",
        schema=schema,
        max_rows_per_shard=2,
        parquet_kwargs={"index": False}
        if isinstance(backend(df), pd.DataFrame)
        else {},
    ).save()
    assert repeated == first
    assert repeated.artifacts.count() == 2
    assert repeated.open().count_rows() == 4
    second = ln.Collection.from_dataframe(
        backend(df.iloc[:2]),
        key="sharded/versioned",
        schema=schema,
        max_rows_per_shard=2,
        parquet_kwargs={"index": False}
        if isinstance(backend(df), pd.DataFrame)
        else {},
    ).save()
    first.refresh_from_db()
    assert second.stem_uid == first.stem_uid
    assert second.hash != first.hash
    assert not first.is_latest and second.is_latest
    assert first.open().count_rows() == 4
    assert second.open().count_rows() == 2


@pytest.mark.parametrize("metadata", [False, True])
def test_sorting_metadata(backend, metadata):
    schema = make_schema(sorted_by=["shard_value"])
    df = pd.DataFrame(
        {"shard_group": ["b", "a", "b", "a"], "shard_value": [1, 2, 3, 4]}
    )
    collection = ln.Collection.from_dataframe(
        backend(df),
        key="sharded/sorted",
        schema=schema,
        write_parquet_sorting_metadata=metadata,
    ).save()
    for artifact in collection.ordered_artifacts:
        actual = pq.ParquetFile(artifact.path).metadata.row_group(0).sorting_columns
        assert bool(actual) == metadata
        if metadata:
            assert actual == (pq.SortingColumn(1),)
    with pytest.raises(ValidationError):
        ln.Collection.from_dataframe(
            backend(df.iloc[::-1]), key="sharded/unsorted", schema=schema
        )
    assert not ln.Artifact.filter(key__startswith="sharded/unsorted").exists()


def test_registry_categories(backend):
    ln.ULabel(name="shard_a").save()
    ln.ULabel(name="shard_b").save()
    group = ln.Feature(name="shard_group", dtype=ln.ULabel.name).save()
    value = ln.Feature(name="shard_value", dtype=int).save()
    schema = ln.Schema([group, value], partition_by="shard_group").save()
    df = pd.DataFrame(
        {"shard_group": ["shard_b", None, "shard_a"], "shard_value": [1, 2, 3]}
    )
    df["shard_group"] = df.shard_group.astype("category")
    collection = ln.Collection.from_dataframe(
        backend(df), key="sharded/registry", schema=schema
    ).save()
    assert collection.artifacts.count() == 3
    assert collection.open().count_rows() == 3
    df["shard_group"] = df.shard_group.cat.add_categories(["unknown"])
    df.loc[0, "shard_group"] = "unknown"
    with pytest.raises(ValidationError):
        ln.Collection.from_dataframe(backend(df), key="sharded/unknown", schema=schema)
    assert not ln.Artifact.filter(key__startswith="sharded/unknown").exists()
    other_schema = ln.Schema([group, value]).save()
    with pytest.raises(ValidationError, match="schemas do not match"):
        ln.Collection(
            list(collection.artifacts.all()),
            key="sharded/mismatch",
            schema=other_schema,
        )


def test_write_failure_does_not_publish_collection(monkeypatch):
    schema = make_schema()
    df = pd.DataFrame({"shard_group": ["a", "b"], "shard_value": [1, 2]})
    original = ln.Artifact.from_dataframe

    def fail_second_shard(*args, **kwargs):
        if kwargs["key"].endswith("00001.parquet"):
            raise OSError("shard write failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(ln.Artifact, "from_dataframe", fail_second_shard)
    with pytest.raises(OSError, match="shard write failed"):
        ln.Collection.from_dataframe(df, key="sharded/failure", schema=schema)
    assert not ln.Collection.filter(key="sharded/failure").exists()
    assert ln.Artifact.filter(key__startswith="sharded/failure/").count() == 1


from importlib.util import find_spec
