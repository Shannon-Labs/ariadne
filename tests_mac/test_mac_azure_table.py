from __future__ import annotations


def test_azure_table_returns_codes() -> None:
    from ariadne_mac.ft.resource_estimator import azure_estimate_table

    table = azure_estimate_table("some_program")
    assert set(table.keys()) == {"surface", "floquet"}
    # Values are structured records; either unavailable or estimate dicts
    for est in table.values():
        assert isinstance(est, dict)
