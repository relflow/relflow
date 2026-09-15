import pytest

from relflow.structs.experiment import Schema


def _structure_payload() -> dict:
    field: dict = {
        "name": "identifier",
        "type": "hash",
    }
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "dropout": 0.1,
            "fields": [
                {
                    "name": "items",
                    "type": "branch",
                    "length": 2,
                    "fields": [field],
                }
            ],
        },
    }


def _schema_payload() -> dict:
    return _structure_payload()


def test_schema_rejects_undeclared_options():
    payload = _schema_payload()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Schema.model_validate(payload)


def test_schema_derives_effective_reconstruct_roles_from_masks():
    payload = _schema_payload()
    payload["fields"]["fields"][0]["mask"] = {"reconstruct": True, "dropout": False}

    schema = Schema.model_validate(payload)

    assert schema.reconstruct == ["root/items/identifier"]
    assert schema.objectives == ["root/items/identifier"]
    assert schema.decodes == ["root/items/identifier"]
    assert schema.forward_for("train") == ["root/items/identifier"]
    assert schema.forward_for("predict") == ["root/items/identifier"]


def test_schema_derives_embed_from_node_attribute():
    payload = _schema_payload()
    payload["fields"]["embed"] = True

    schema = Schema.model_validate(payload)

    assert schema.embed == ["root"]


def test_schema_excludes_dynamic_reconstruct_from_prediction_decodes():
    payload = _schema_payload()
    payload["fields"]["fields"][0]["fields"][0]["mask"] = {
        "rate": 0.5,
        "reconstruct": True,
        "dropout": False,
    }

    schema = Schema.model_validate(payload)

    assert schema.objectives == ["root/items/identifier"]
    assert schema.decodes == []
    assert schema.forward_for("train") == ["root/items/identifier"]
    assert schema.forward_for("predict") == []


def test_schema_mask_round_trip_preserves_skip_policy():
    payload = _schema_payload()
    payload["fields"]["mask"] = {"skip": True, "reconstruct": True, "dropout": False}
    schema = Schema.model_validate(payload)

    restored = Schema.model_validate(schema.model_dump(mode="python", round_trip=True))

    assert restored.fields.mask == schema.fields.mask
    assert restored.reconstruct == ["root/items/identifier"]

    restored_json = Schema.model_validate_json(schema.model_dump_json())

    assert restored_json.fields.mask == schema.fields.mask
    assert restored_json.reconstruct == ["root/items/identifier"]
