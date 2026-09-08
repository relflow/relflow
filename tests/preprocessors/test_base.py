from enum import StrEnum

import polars as pl
import pyarrow as pa
import pytest

import relflow as rf
from relflow.data import processors
from relflow.structs.enums import Strata


def make_frame(values: list[int]) -> pl.DataFrame:
    return pl.DataFrame({"value": values})


def test_preprocessor_providers_are_string_enums():
    assert issubclass(rf.PreprocessorProvider, StrEnum)
    assert rf.PreprocessorProvider.strata == "strata"
    assert rf.PreprocessorProvider.schema == "schema"
    assert rf.PreprocessorProvider.encoding_context == "encoding_context"


def test_preprocess_returns_callable_processor_object():
    @processors.preprocess
    def increment(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col("value") + 1)

    source = make_frame([1, 2])
    assert isinstance(increment, processors.Preprocessor)
    assert increment(source).to_dict(as_series=False) == {"value": [2, 3]}
    [result] = increment.run(source, strata=Strata.train, schema=None, encoding_context={})
    assert result.equals(pl.DataFrame({"value": [2, 3]}))


def test_preprocess_scope_is_the_only_decorator_option():
    @processors.preprocess(scope="dataset")
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    assert prepare.scope == "dataset"
    assert not hasattr(prepare, "requires")
    assert not hasattr(prepare, "produces")

    with pytest.raises(ValueError, match="scope must be"):
        processors.preprocess(scope="global")(lambda frame: frame)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="unexpected keyword argument 'requires'"):
        processors.preprocess(requires="value")  # type: ignore[call-overload]


def test_processor_decorators_accept_direct_calls():
    prepare = processors.preprocess(lambda frame: frame, scope="dataset")
    finish = processors.postprocess(lambda frame: frame)

    assert prepare.scope == "dataset"
    assert isinstance(finish, processors.Postprocessor)


def test_processor_pipelines_normalize_to_immutable_tuples():
    @rf.preprocess
    def first(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    @rf.preprocess
    def second(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    configured = [first, second, first]

    assert rf.Preprocessor.normalize(None) == ()
    assert rf.Preprocessor.normalize(first) == (first,)
    assert rf.Preprocessor.normalize(configured) == (first, second, first)
    assert rf.Preprocessor.normalize(tuple(configured)) == (first, second, first)

    normalized = rf.Preprocessor.normalize(configured)
    configured.pop()
    assert normalized == (first, second, first)


def test_processor_pipeline_rejects_wrong_and_nested_members_by_index():
    @rf.preprocess
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    @rf.postprocess
    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    with pytest.raises(TypeError, match="preprocessor at index 1.*Postprocessor"):
        rf.Preprocessor.normalize([prepare, finish])
    with pytest.raises(TypeError, match="preprocessor at index 1.*list"):
        rf.Preprocessor.normalize([prepare, [prepare]])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="preprocessor must be a Preprocessor, list, tuple, or None"):
        rf.Preprocessor.normalize(processor for processor in (prepare,))  # type: ignore[arg-type]


def test_preprocessor_discards_none_and_expands_frames():
    @processors.preprocess
    def discard(frame: pl.DataFrame):
        return None

    @processors.preprocess
    def split(frame: pl.DataFrame):
        yield frame.head(1)
        yield frame.slice(1)

    source = make_frame([1, 2, 3])
    assert list(discard.run(source, strata=Strata.train, schema=None, encoding_context={})) == []
    assert [
        item["value"].to_list() for item in split.run(source, strata=Strata.train, schema=None, encoding_context={})
    ] == [[1], [2, 3]]


@pytest.mark.parametrize(
    "value",
    [pa.table({"value": [1]}), {"value": [1]}, [1], pl.Series("value", [1]), pl.LazyFrame({"value": [1]})],
)
def test_preprocessor_rejects_non_dataframe_results(value):
    @processors.preprocess
    def invalid(frame: pl.DataFrame):
        return value

    with pytest.raises(TypeError, match="DataFrame"):
        list(invalid.run(make_frame([1]), strata=Strata.train, schema=None, encoding_context={}))


def test_preprocessor_accepts_arbitrary_canonical_row_changes():
    @rf.preprocess
    def canonicalize(frame: pl.DataFrame) -> pl.DataFrame:
        return (
            frame.filter(pl.col("value") > 1)
            .sort("value", descending=True)
            .select(pl.col("value").repeat_by(2).explode())
        )

    [result] = canonicalize.run(make_frame([1, 2, 3]), strata=Strata.train, schema=None, encoding_context={})
    assert result["value"].to_list() == [3, 3, 2, 2]


def test_preprocessor_receives_only_named_pipeline_providers():
    @processors.preprocess
    def inspect(frame: pl.DataFrame, *, strata, schema, encoding_context):
        assert strata is Strata.validate
        assert schema == "schema"
        assert encoding_context == {"marker": "seen"}
        return frame

    source = make_frame([1])
    assert list(
        inspect.run(
            source,
            strata=Strata.validate,
            schema="schema",
            encoding_context={"marker": "seen"},
        )
    ) == [source]


def test_required_user_parameters_are_bound_immutably():
    @processors.preprocess
    def offset(frame: pl.DataFrame, *, amount: int) -> pl.DataFrame:
        return frame.with_columns(pl.col("value") + amount)

    with pytest.raises(ValueError, match="requires unbound parameter"):
        list(offset.run(make_frame([1]), strata=Strata.train, schema=None, encoding_context={}))

    configured = offset.partial(amount=4)
    assert configured.bound == {"amount": 4}
    assert offset.bound == {}
    [result] = configured.run(make_frame([1]), strata=Strata.train, schema=None, encoding_context={})
    assert result["value"].to_list() == [5]

    with pytest.raises(ValueError, match="already bound"):
        configured.partial(amount=5)


def test_pipeline_parameters_cannot_be_bound():
    @processors.preprocess
    def inspect(frame: pl.DataFrame, *, strata):
        return frame

    with pytest.raises(ValueError, match="provided by the pipeline"):
        inspect.partial(strata=Strata.train)


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (lambda value: value, "first parameter must be 'frame'"),
        (lambda frame, value: frame, "must be keyword-only"),
    ],
)
def test_processor_signatures_are_explicit(function, message):
    with pytest.raises(TypeError, match=message):
        processors.preprocess(function)


def test_preprocessor_normalize_rejects_raw_callable():
    with pytest.raises(TypeError, match="preprocessor must be a Preprocessor"):
        processors.Preprocessor.normalize(lambda frame: frame)


def test_postprocessor_may_change_rows_and_columns():
    @rf.postprocess
    def compact(frame: pl.DataFrame, *, threshold: int) -> pl.DataFrame:
        return frame.filter(pl.col("value") >= threshold).select(large=pl.lit(True))

    output = compact.partial(threshold=2).run(make_frame([1, 3]))
    assert output.to_dict(as_series=False) == {"large": [True]}


def test_postprocessor_pipeline_runs_in_order():
    calls: list[str] = []

    @rf.postprocess
    def double(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("double")
        return frame.select(doubled=pl.col("value") * 2)

    @rf.postprocess
    def classify(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("classify")
        return frame.select(large=pl.col("doubled") > 3)

    result = processors.apply(pa.table({"value": [1, 2]}), [double, classify])

    assert calls == ["double", "classify"]
    assert result.to_pydict() == {"large": [False, True]}


def test_processor_boundaries_convert_arrow_once(monkeypatch):
    counts = {"from": 0, "to": 0}
    from_arrow = pl.from_arrow
    to_arrow = pl.DataFrame.to_arrow

    def convert_from(*args, **kwargs):
        counts["from"] += 1
        return from_arrow(*args, **kwargs)

    def convert_to(self, *args, **kwargs):
        counts["to"] += 1
        return to_arrow(self, *args, **kwargs)

    monkeypatch.setattr(pl, "from_arrow", convert_from)
    monkeypatch.setattr(pl.DataFrame, "to_arrow", convert_to)

    @rf.postprocess
    def first(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col("value") + 1)

    @rf.postprocess
    def second(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(pl.col("value") * 2)

    result = processors.apply(pa.table({"value": [1]}), [first, second])

    assert result.to_pydict() == {"value": [4]}
    assert counts == {"from": 1, "to": 1}


def test_processors_reject_zero_column_results():
    @rf.preprocess
    def empty(frame: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame()

    @rf.postprocess
    def also_empty(frame: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame()

    with pytest.raises(ValueError, match="at least one Polars column"):
        list(empty.run(make_frame([1]), strata=Strata.train, schema=None, encoding_context={}))
    with pytest.raises(ValueError, match="at least one Polars column"):
        also_empty.run(make_frame([1]))


def test_postprocessor_normalize_rejects_raw_callable():
    with pytest.raises(TypeError, match="postprocessor must be a Postprocessor"):
        rf.Postprocessor.normalize(lambda frame: frame)
