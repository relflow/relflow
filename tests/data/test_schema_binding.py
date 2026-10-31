import pyarrow as pa

import relflow as rf


def test_schema_binding_preserves_sparse_coordinates() -> None:
    observation = pa.Table.from_pylist(
        [
            {
                "events": [
                    {"ip_country": "US", "amount": None},
                    {"ip_country": None, "amount": 950.0},
                    {"ip_country": "CA", "amount": None},
                ]
            }
        ]
    )

    model = rf.Model.xs(
        d_model=8,
        events=rf.Branch(
            length=3,
            ip_country=rf.Category,
            amount=rf.Number,
        ),
    )
    encoded = model.encode(observation)
    assert encoded["/events/ip_country"].state.tolist() == [[[rf.Tokens.valued, rf.Tokens.null, rf.Tokens.valued]]]
    assert encoded["/events/amount"].state.tolist() == [[[rf.Tokens.null, rf.Tokens.valued, rf.Tokens.null]]]
