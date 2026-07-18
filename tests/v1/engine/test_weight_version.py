# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.v1.engine.core import EngineCore


def _make_engine_core() -> EngineCore:
    engine_core = object.__new__(EngineCore)
    engine_core.model_executor = MagicMock()
    engine_core._weight_version = 0
    engine_core._weight_update_is_draft = None
    engine_core.scheduler = SimpleNamespace(
        add_request=MagicMock(),
        get_kv_connector=MagicMock(return_value=None),
        get_ec_connector=MagicMock(return_value=None),
    )
    return engine_core


def _make_request(request_id: str):
    return SimpleNamespace(
        request_id=request_id,
        pooling_params=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        abort_immediately=False,
        weight_version=None,
    )


@pytest.mark.parametrize(
    ("start_method", "expected_version"),
    [("start_weight_update", 1), ("start_draft_weight_update", 0)],
)
def test_successful_finish_updates_policy_weight_version(
    start_method: str, expected_version: int
):
    engine_core = _make_engine_core()
    engine_core.model_executor.collective_rpc.return_value = [None]

    engine_core.collective_rpc(start_method)
    assert engine_core.collective_rpc("finish_weight_update") == [None]
    assert engine_core.get_weight_version() == expected_version


def test_failed_finish_does_not_advance_weight_version():
    engine_core = _make_engine_core()
    engine_core.model_executor.collective_rpc.return_value = [None]
    engine_core.collective_rpc("start_weight_update")
    engine_core.model_executor.collective_rpc.side_effect = RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        engine_core.collective_rpc("finish_weight_update")

    assert engine_core.get_weight_version() == 0


def test_request_version_is_bound_at_scheduler_admission():
    engine_core = _make_engine_core()

    request = _make_request("request-1")
    engine_core._weight_version = 3

    def assert_bound_before_scheduler_add(admitted_request):
        assert admitted_request.weight_version == 3

    engine_core.scheduler.add_request.side_effect = assert_bound_before_scheduler_add
    engine_core.add_request(request)

    # The request remains at version 3 even if an update finishes before its
    # first scheduler step.
    engine_core._weight_version = 4
    assert request.weight_version == 3
