from __future__ import annotations

import pytest


@pytest.mark.azure
def test_azure_resources_wrapper() -> None:
    from ariadne_mac.ft.resource_estimator import azure_estimate_table

    table = azure_estimate_table("qsp_demo")
    assert set(table.keys()) == {"surface", "floquet"}
    # Either unavailable record or dict with numbers
    for rec in table.values():
        assert isinstance(rec, dict)

