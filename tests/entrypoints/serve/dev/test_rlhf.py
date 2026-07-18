# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.serve.dev.rlhf.api_router import attach_router


def test_weight_info():
    app = FastAPI()
    engine = MagicMock(spec=EngineClient)
    engine.get_weight_version = AsyncMock(return_value=3)
    app.state.engine_client = engine
    attach_router(app)

    response = TestClient(app).get("/weight_info")

    assert response.status_code == 200
    assert response.json() == {"weight_version": 3}
    engine.get_weight_version.assert_awaited_once_with()
