from relflow.structs.experiment import Schema


def test_schema_supports_programmatic_instantiation():
    schema = Schema.model_validate(
        {
            "d_model": 16,
            "fields": {
                "type": "branch",
                "dropout": 0.1,
                "length": 1,
                "fields": [
                    {
                        "name": "identifier",
                        "type": "category",
                    }
                ],
            },
        }
    )

    assert schema.d_model == 16
    assert "/identifier" in schema.requests
