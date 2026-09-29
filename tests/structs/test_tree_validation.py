import pydantic
import pytest

from relflow.structs.tree import Address, Leaf, Mask, Node


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

    assert leaf.query == "payload.amount"


def test_leaf_requires_non_empty_explicit_query():
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "   "})


def test_leaf_rejects_invalid_explicit_query():
    with pytest.raises(ValueError, match="invalid query"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "["})


def test_leaf_query_is_observation_relative():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "payload.amount"})

    assert leaf.query == "payload.amount"


def test_leaf_query_rejects_a_batch_axis_prefix():
    with pytest.raises(ValueError, match="must not begin with"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "query": "[*].payload.amount"})


def test_leaf_defaults_to_not_embedded():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4})

    assert leaf.embed is False


def test_leaf_mask_defaults_to_an_empty_tuple():
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4})

    assert Leaf.model_fields["mask"].default is False
    assert leaf.mask == ()


@pytest.mark.parametrize("value", [False, [], ()])
def test_leaf_empty_mask_forms_normalize(value):
    assert Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "mask": value}).mask == ()


def test_leaf_mask_singletons_normalize():
    assert Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "mask": 0.25}).mask == (
        Mask(rate=0.25),
    )
    assert Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "mask": True}).mask == (
        Mask(skip=True, dropout=False, reconstruct=True),
    )


def test_leaf_mask_collection_is_copied_normalized_and_deduplicated():
    values = [Mask(rate=0.0), Mask(rate=1.0), Mask(), Mask(query="selected")]
    leaf = Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "mask": values})

    values.clear()

    assert leaf.mask == (Mask(), Mask(query="selected"))


def test_leaf_mask_rejects_none_and_non_mask_collection_entries():
    with pytest.raises(TypeError, match="mask cannot be None"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "mask": None})

    with pytest.raises(TypeError, match="entries must be Mask"):
        Leaf.model_validate({"name": "leaf", "type": "number", "n_heads": 4, "mask": [True]})


def test_mask_normalizes_dropout_and_rejects_invalid_combinations():
    assert Mask().dropout is True
    assert Mask(reconstruct=True).dropout is False

    with pytest.raises(ValueError, match="dropout=True"):
        Mask(dropout=True, reconstruct=True)

    assert Mask(skip=True, rate=0.5) == Mask(skip=True, rate=0.5, dropout=True)


def test_mask_is_frozen_and_rejects_unknown_fields():
    policy = Mask()

    with pytest.raises(pydantic.ValidationError, match="Instance is frozen"):
        policy.skip = True

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Mask.model_validate({"unknown": True})


def test_node_rejects_extra_fields():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Node.model_validate({"name": "ok_name", "type": "node", "n_heads": 4, "unexpected": 0.0})


def test_node_description_trims_and_accepts_optional_metadata():
    node = Node.model_validate({"name": "ok_name", "type": "node", "description": "  docs here  ", "n_heads": 4})
    assert node.description == "docs here"


def test_node_description_empty_string_becomes_none():
    node = Node.model_validate({"name": "ok_name", "type": "node", "description": "   ", "n_heads": 4})
    assert node.description is None
