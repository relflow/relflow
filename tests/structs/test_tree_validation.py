import pydantic
import pytest

import relflow as rf
from relflow.structs.tree import Address, Leaf, Node, selections


class AddressPayload(pydantic.BaseModel):
    address: Address


def test_address_can_be_initialized_from_path_parts():
    address = Address("transactions", "amount")

    assert address == "/transactions/amount"
    assert isinstance(address, str)


def test_address_accepts_slash_delimited_or_path_parts():
    assert Address("/label") == Address("label") == Address("/", "label")
    assert Address("/metrics/sepal_length") == Address("metrics", "sepal_length")
    assert Address("metrics/sepal_length") == "/metrics/sepal_length"
    assert Address(Address("metrics", "sepal_length")) == "/metrics/sepal_length"


def test_empty_components_select_the_root_but_explicit_empty_paths_are_unbound():
    assert Address() == Address("/") == "/"
    assert Address("") == ""
    assert Address() != Address("")


@pytest.mark.parametrize("parts", [(None,), (1,), (b"amount",), ("transactions", None), (1, "amount")])
def test_address_rejects_non_string_components(parts):
    with pytest.raises(TypeError, match="Address parts must be strings"):
        Address(*parts)


def test_address_can_be_pydantic_coerced_from_string():
    payload = AddressPayload.model_validate({"address": "/label"})

    assert payload.address == Address("label")
    assert isinstance(payload.address, Address)
    assert payload.model_dump(mode="json") == {"address": "/label"}
    assert AddressPayload.model_validate({"address": "label"}).address == payload.address


def test_node_rejects_invalid_name_characters():
    with pytest.raises(ValueError, match="name may contain only letters"):
        Node.model_validate({"name": "bad name", "type": "node", "n_heads": 4})


def test_node_requires_even_n_heads():
    with pytest.raises(ValueError, match="n_heads must be even"):
        Node.model_validate({"name": "ok_name", "type": "node", "n_heads": 3})


def test_leaf_query_is_optional():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4})

    assert leaf.query is None


def test_leaf_accepts_explicit_request_level_query():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "payload.amount"})

    assert leaf.query == ("payload.amount",)


def test_leaf_requires_non_empty_explicit_query():
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "   "})


def test_leaf_rejects_invalid_explicit_query():
    with pytest.raises(ValueError, match="invalid query"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "["})


def test_leaf_query_rejects_a_batch_axis_prefix():
    with pytest.raises(ValueError, match="must not begin with"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "[*].payload.amount"})


def test_leaf_defaults_to_not_embedded():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4})

    assert leaf.embed is False


def test_leaf_hide_defaults_to_an_empty_tuple():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4})
    assert leaf.hide == ()
    assert leaf.predict is False
    assert leaf.task is None


@pytest.mark.parametrize("value", [False, True, None, 1, 1.0, -0.1, float("nan"), float("inf"), "[", ("flag", True)])
def test_leaf_hide_rejects_invalid_selection(value):
    with pytest.raises((TypeError, ValueError)):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "hide": value})


@pytest.mark.parametrize("value", [None, 0, 1, "true", (True,)])
def test_prediction_requires_a_boolean(value):
    with pytest.raises((TypeError, ValueError)):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "predict": value})


def test_leaf_serializes_hide_selectors():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "hide": (0.25, "flag")})
    assert leaf.hide == (0.25, "flag")
    assert Leaf.model_validate_json(leaf.model_dump_json()).hide == leaf.hide


def test_node_rejects_extra_fields():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Node.model_validate({"name": "ok_name", "type": "node", "n_heads": 4, "unexpected": 0.0})


def test_node_description_trims_and_accepts_optional_metadata():
    node = Node.model_validate({"name": "ok_name", "type": "node", "description": "  docs here  ", "n_heads": 4})
    assert node.description == "docs here"


def test_node_description_empty_string_becomes_none():
    node = Node.model_validate({"name": "ok_name", "type": "node", "description": "   ", "n_heads": 4})
    assert node.description is None


def test_duplicate_and_zero_hide_rules_compile_to_one_selection():
    selectors = [0.0, "redacted", "redacted", 0.4, 0.4]
    leaf = Leaf.model_validate({"type": "number", "hide": selectors})
    selectors.clear()
    assert leaf.hide == (0.0, "redacted", "redacted", 0.4, 0.4)
    assert [(rule.query, rule.rate) for rule in selections(leaf)] == [("redacted", None), (None, 0.4)]


def test_reconstruction_is_immutable_and_rejects_unknown_options():
    policy = rf.reconstruct(rate=0.25)
    with pytest.raises(ValueError, match="frozen"):
        policy.rate = 0.75
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        rf.Reconstruction.model_validate({"rate": 0.25, "dropout": True})


def test_hide_rejects_mapping_selectors_and_mask_is_not_a_schema_option():
    with pytest.raises(TypeError, match="hide expects a float probability or a Boolean query path"):
        Leaf.model_validate({"type": "number", "hide": [{"skip": True, "reconstruct": True}]})
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Leaf.model_validate({"type": "number", "mask": 0.2})
