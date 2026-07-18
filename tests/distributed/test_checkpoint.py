# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.distributed.device_communicators import flashinfer_all_reduce


@pytest.fixture
def reset_allreduce_state(monkeypatch):
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_workspace", None)
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_quant_workspace", None)
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_workspace_groups", {})


@pytest.mark.parametrize("distinct_workspaces", [False, True])
def test_allreduce_checkpoint_restores_each_workspace_with_its_group(
    monkeypatch, reset_allreduce_state, distinct_workspaces
):
    workspace_count = 2 if distinct_workspaces else 1
    workspaces = [Mock() for _ in range(workspace_count)]
    groups = [object() for _ in range(workspace_count)]
    backends = {group: object() for group in groups}
    create_workspace = Mock(side_effect=workspaces)
    monkeypatch.setattr(
        flashinfer_all_reduce,
        "flashinfer_comm",
        SimpleNamespace(create_allreduce_fusion_workspace=create_workspace),
        raising=False,
    )
    torch_dist_backend = Mock(side_effect=lambda *, group: backends[group])
    monkeypatch.setattr(
        flashinfer_all_reduce, "TorchDistBackend", torch_dist_backend, raising=False
    )

    created = [
        flashinfer_all_reduce._create_workspace(
            "mnnvl", 2, rank, 128, 256, torch.float16, group
        )
        for rank, group in enumerate(groups)
    ]
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_workspace", created[0])
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_quant_workspace", created[-1])

    flashinfer_all_reduce.checkpoint_prepare_fi_ar_workspaces()
    flashinfer_all_reduce.checkpoint_restore_fi_ar_workspaces()

    for workspace, group in zip(workspaces, groups):
        workspace.checkpoint_prepare.assert_called_once_with()
        workspace.checkpoint_restore.assert_called_once_with(backends[group])
    expected_backend_calls = [call(group=group) for group in groups]
    assert torch_dist_backend.call_args_list == expected_backend_calls * 2


@pytest.mark.parametrize(
    "checkpoint_fn",
    [
        flashinfer_all_reduce.checkpoint_prepare_fi_ar_workspaces,
        flashinfer_all_reduce.checkpoint_restore_fi_ar_workspaces,
    ],
)
def test_allreduce_checkpoint_rejects_missing_api(
    monkeypatch, reset_allreduce_state, checkpoint_fn
):
    workspace = object()
    monkeypatch.setattr(flashinfer_all_reduce, "_fi_ar_workspace", workspace)
    monkeypatch.setattr(
        flashinfer_all_reduce,
        "_fi_ar_workspace_groups",
        {id(workspace): object()},
    )

    with pytest.raises(NotImplementedError, match="^Checkpointing not supported$"):
        checkpoint_fn()


def test_distributed_checkpoint_order_and_fences(monkeypatch):
    events = []

    def group(name):
        return SimpleNamespace(
            device_communicator=SimpleNamespace(
                checkpoint_prepare=lambda: events.append(f"prepare-{name}"),
                checkpoint_restore=lambda: events.append(f"restore-{name}"),
            )
        )

    for name in ("_WORLD", "_TP", "_DCP", "_PCP", "_PP", "_DP", "_EP", "_EPLB"):
        monkeypatch.setattr(parallel_state, name, None)
    group_a = group("a")
    monkeypatch.setattr(parallel_state, "_TP", group_a)
    monkeypatch.setattr(parallel_state, "_DCP", group_a)
    monkeypatch.setattr(parallel_state, "_EP", group("b"))
    monkeypatch.setattr(
        parallel_state.torch.accelerator,
        "synchronize",
        lambda: events.append("fence"),
    )
    monkeypatch.setattr(
        flashinfer_all_reduce,
        "checkpoint_prepare_fi_ar_workspaces",
        lambda: events.append("prepare-allreduce"),
    )
    monkeypatch.setattr(
        flashinfer_all_reduce,
        "checkpoint_restore_fi_ar_workspaces",
        lambda: events.append("restore-allreduce"),
    )

    parallel_state.checkpoint_prepare_distributed_state()
    parallel_state.checkpoint_restore_distributed_state()

    assert events == [
        "fence",
        "prepare-a",
        "prepare-b",
        "prepare-allreduce",
        "fence",
        "fence",
        "restore-allreduce",
        "restore-a",
        "restore-b",
        "fence",
    ]
