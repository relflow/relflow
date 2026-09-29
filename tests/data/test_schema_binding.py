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

    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=3,
            ip_country=rf.Category(),
            amount=rf.Number,
        ),
    )
    encoded = model.encode(observation)
    assert encoded[rf.Address("/", "events", "ip_country")].state.tolist() == [[[0, 1, 0]]]
    assert encoded[rf.Address("/", "events", "amount")].state.tolist() == [[[1, 0, 1]]]

    source = [
        {
            "events": [
                {"event_type": "login", "device_id": "a", "risk_score": None},
                {"event_type": "other", "device_id": "b", "risk_score": 1.0},
                {"event_type": "login", "device_id": "c", "risk_score": 2.0},
            ]
        }
    ]
    filtered = [
        {
            "login_events": [event for event in record["events"] if event["event_type"] == "login"],
        }
        for record in source
    ]
    filtered_model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        login_events=rf.Branch(
            length=2,
            device_id=rf.Category(),
            risk_score=rf.Number,
        ),
    )
    filtered_encoded = filtered_model.encode(pa.Table.from_pylist(filtered))
    assert filtered_encoded[rf.Address("/", "login_events", "device_id")].state.tolist() == [[[0, 0]]]
    assert filtered_encoded[rf.Address("/", "login_events", "risk_score")].state.tolist() == [[[1, 0]]]
