"""Portable schema tooling preserves data and uses live extension contracts."""

import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import skills

DIRECTORY = Path(__file__).resolve().parents[1] / "skills/relflow-schema/scripts"


@pytest.fixture
def tooling(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    modules = {}
    for name in ("observations", "profile", "validate"):
        filename = "profile_data" if name == "profile" else name
        spec = importlib.util.spec_from_file_location(f"skill_{name}", DIRECTORY / f"{filename}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
        if name == "observations":
            monkeypatch.setitem(sys.modules, name, module)
    return modules


def model(directory, fields, **options):
    import yaml

    path = directory / "model.yaml"
    path.write_text(yaml.safe_dump({"preset": "xs", **options, "fields": fields}), encoding="utf-8")
    return path


def test_jsonl_union_of_root_fields_and_prefix_scope(tooling, tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id":"0012"}\n{"amount":4}\n{"unread":true}\n')
    table, source = tooling["observations"].read(path, rows=2)
    assert table.to_pylist() == [{"id": "0012", "amount": None}, {"id": None, "amount": 4}]
    assert source["sample_rows"] == 2 and "total_rows" not in source
    assert len(source["sample_sha256"]) == 64


def test_csv_keeps_identifiers_and_literal_null_labels(tooling, tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text("id,label,amount\n0012,NULL,1.25\n0013,NA,3\n")
    table, _ = tooling["observations"].read(path)
    assert table.to_pylist() == [
        {"id": "0012", "label": "NULL", "amount": "1.25"},
        {"id": "0013", "label": "NA", "amount": "3"},
    ]


def test_parquet_and_ipc_prefixes_preserve_schema(tooling, tmp_path):
    table = pa.table({"id": range(100), "history": [[1, 2]] * 100})
    for suffix in ("parquet", "arrow", "ipc"):
        path = tmp_path / f"rows.{suffix}"
        if suffix == "parquet":
            pq.write_table(table, path, row_group_size=10)
        else:
            with path.open("wb") as sink:
                factory = pa.ipc.new_stream if suffix == "ipc" else pa.ipc.new_file
                with factory(sink, table.schema) as writer:
                    writer.write_table(table, max_chunksize=10)
        sampled, source = tooling["observations"].read(path, rows=3)
        assert sampled.equals(table.slice(0, 3))
        assert source["sample_rows"] == 3
        if suffix == "parquet":
            assert source["total_rows"] == 100


def test_parser_and_retained_arrow_budgets_fail_with_remedies(tooling, tmp_path):
    path = tmp_path / "rows.json"
    path.write_text('[{"text":"long value"}]')
    with pytest.raises(ValueError, match="use JSONL or Arrow"):
        tooling["observations"].read(path, max_bytes=5)
    pq.write_table(pa.table({"text": ["long value"]}), tmp_path / "rows.parquet")
    with pytest.raises(ValueError, match="lower --rows"):
        tooling["observations"].read(tmp_path / "rows.parquet", max_bytes=5)
    with pytest.raises(ValueError, match="positive"):
        tooling["observations"].read(path, rows=0)


def test_nested_profile_preserves_null_empty_and_null_item_distinctions(tooling):
    table = pa.Table.from_pylist(
        [{"events": None}, {"events": []}, {"events": [None, {"code": "0012"}, {"code": None}]}]
    )
    result = tooling["profile"].profile(table)
    events = result["fields"]["events"]
    assert events["nulls"] == 1
    assert events["lengths"]["min"] == 0 and events["lengths"]["max"] == 3
    assert events["element"]["nulls"] == 1
    code = events["element"]["fields"]["code"]
    assert code["nulls"] == 2
    assert code["patterns"]["leading_zero_integer_text"] == 1


def test_profile_handles_sliced_lists_all_null_and_nonfinite_values(tooling):
    table = pa.table(
        {
            "events": pa.array([[9], [], None, [1, 2]]).slice(1),
            "unknown": pa.nulls(3),
            "number": [float("nan"), float("inf"), 3.0],
        }
    )
    result = tooling["profile"].profile(table, max_values=2)
    events = result["fields"]["events"]
    assert events["truncated"] and events["lengths"]["max"] == 0
    assert events["element"]["sampled_values"] == 0
    assert result["fields"]["number"]["nonfinite"] == 2
    json.dumps(result, allow_nan=False)
    assert "examples" not in tooling["profile"].profile(table, examples=0)["fields"]["number"]


def test_profile_enforces_schema_traversal_budget(tooling):
    with pytest.raises(ValueError, match="max-fields"):
        tooling["profile"].profile(pa.table({"a": [1], "b": [2]}), max_fields=1)


def test_validator_checks_hidden_supervised_target_types_without_model_construction(tooling, tmp_path, monkeypatch):
    import relflow as rf

    def construct(*args, **kwargs):
        raise AssertionError("schema checks must not instantiate neural modules")

    monkeypatch.setattr(rf.Model, "__init__", construct)
    path = model(tmp_path, {"amount": {"type": "number"}, "label": {"type": "number", "mask": True}})
    data = tmp_path / "data.json"
    data.write_text('[{"amount":2,"label":"invalid number"}]')
    with pytest.raises(TypeError, match="/label"):
        tooling["validate"].validate(path, data=data)
    data.write_text('[{"amount":2,"label":1}]')
    result = tooling["validate"].validate(path, data=data)
    assert result["checks"] == {"schema": "passed", "arrow": "passed", "encoding": "not_run", "training": "not_run"}
    assert result["objectives"] == ["/label"]
    assert all(
        any(field["address"] == "/label" and field["values"] for field in fields)
        for fields in result["fields"].values()
    )


def test_validator_uses_late_registered_extension_contracts(tooling, tmp_path):
    import relflow as rf
    from relflow.tensorfields.base import TENSORFIELDS

    class Request(rf.RequestBase):
        type: Literal["skillcustom"] = "skillcustom"

    assert "skillcustom" not in TENSORFIELDS
    extension = rf.Extension("skillcustom", types=(int,))
    try:
        extension.register(rf.Component.Request, Request)
        path = model(tmp_path, {"value": {"type": "skillcustom"}, "label": {"type": "number", "mask": True}})
        data = tmp_path / "data.json"
        data.write_text('[{"value":4,"label":1}]')
        result = tooling["validate"].validate(path, data=data)
        assert any(field["type"] == "skillcustom" for field in result["fields"]["train"])
        data.write_text('[{"value":"wrong family","label":1}]')
        with pytest.raises(TypeError, match="skillcustom.*does not accept Arrow type"):
            tooling["validate"].validate(path, data=data)
    finally:
        del TENSORFIELDS["skillcustom"]


def test_absent_visible_fields_fail_the_live_sdk_contract(tooling, tmp_path):
    path = model(tmp_path, {"value": {"type": "number"}, "label": {"type": "number", "mask": True}})
    data = tmp_path / "data.json"
    data.write_text('[{"label":1}]')
    with pytest.raises(ValueError, match="/value.*absent"):
        tooling["validate"].validate(path, data=data)


def test_empty_contexts_are_reported_as_partial_compatibility(tooling, tmp_path):
    path = model(
        tmp_path,
        {
            "events": {"type": "branch", "length": 2, "fields": {"value": {"type": "number"}}},
            "label": {"type": "number", "mask": True},
        },
    )
    data = tmp_path / "data.json"
    data.write_text('[{"events":[],"label":1}]')
    result = tooling["validate"].validate(path, data=data)
    assert result["checks"]["arrow"] == "partial"
    assert any("/events/value" in limit and "unchecked" in limit for limit in result["limits"])


def test_validator_rejects_unbounded_and_multiplicative_geometry(tooling, tmp_path):
    path = model(tmp_path, {"events": {"type": "branch", "fields": {"value": {"type": "number"}}}})
    with pytest.raises(ValueError, match="unbounded branch geometry"):
        tooling["validate"].validate(path)
    assert tooling["validate"].validate(path, allow_unbounded=True)["declared_slots_per_row_bound"] is None
    path = model(
        tmp_path,
        {
            "events": {
                "type": "branch",
                "length": 1024,
                "fields": {"child": {"type": "branch", "length": 1024, "fields": {"value": {"type": "number"}}}},
            }
        },
    )
    with pytest.raises(ValueError, match="above max_slots"):
        tooling["validate"].validate(path)


def test_explicit_encoding_checks_extension_semantics(tooling, tmp_path):
    path = model(
        tmp_path, {"value": {"type": "number"}, "label": {"type": "number", "mask": True}}, d_model=8, n_heads=2
    )
    data = tmp_path / "data.json"
    data.write_text('[{"value":2,"label":1}]')
    assert tooling["validate"].validate(path, data=data, encode=True)["checks"]["encoding"] == "passed"
    data.write_text(json.dumps([{"value": 2**53 + 1, "label": 1}]))
    with pytest.raises(ValueError, match=r"2\^53|9007199254740992|exact"):
        tooling["validate"].validate(path, data=data, encode=True)


def test_cli_reports_machine_readable_failure_and_nonzero_status(tooling, tmp_path, monkeypatch, capsys):
    path = tmp_path / "wrong.txt"
    path.write_text("not observations")
    monkeypatch.setattr(sys, "argv", ["profile.py", str(path)])
    assert tooling["profile"].main() == 1
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["ok"] is False


def test_publishing_bundles_are_deterministic_and_self_contained(tmp_path):
    first = skills.build(tmp_path / "first")
    second = skills.build(tmp_path / "second")
    assert first == second
    source = skills.ROOT / "skills/relflow-schema/SKILL.md"
    with zipfile.ZipFile(tmp_path / "first/relflow-schema-0.1.0-skill.zip") as bundle:
        assert bundle.read("relflow-schema/SKILL.md") == source.read_bytes()
        assert "relflow-schema/LICENSE" in bundle.namelist()
    with zipfile.ZipFile(tmp_path / "first/relflow-schema-0.1.0-plugin.zip") as bundle:
        portable = json.loads(bundle.read("plugin.json"))
        claude = json.loads(bundle.read(".claude-plugin/plugin.json"))
        assert portable["name"] == claude["name"] == "relflow-schema"
        assert bundle.read("skills/relflow-schema/SKILL.md") == source.read_bytes()
        assert all("__pycache__" not in name and not name.startswith("/") for name in bundle.namelist())
    with pytest.raises(FileExistsError):
        skills.build(tmp_path / "first")
