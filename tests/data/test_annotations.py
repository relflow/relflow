"""Public processor annotations remain available to runtime introspection."""

from collections.abc import Iterator
from typing import get_overloads, get_type_hints

import polars as pl

import relflow as rf


def test_processor_annotations_resolve_after_a_normal_package_import():
    hints = get_type_hints(rf.Preprocessor.run)
    assert hints["strata"] is rf.Strata
    assert hints["schema"] is rf.Schema
    assert hints["encoding_context"] == dict[rf.Address, object]
    assert hints["return"] == Iterator[pl.DataFrame]

    for decorator in (rf.preprocess, rf.postprocess):
        for variant in get_overloads(decorator):
            assert "return" in get_type_hints(variant)
