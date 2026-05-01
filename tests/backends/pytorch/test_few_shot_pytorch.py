import sys
from unittest import mock

import gemma_4_sql.backends.pytorch.few_shot as fs


def test_build_few_shot_prompt_pytorch():
    res = fs.build_few_shot_prompt("foo", "prompt", [{"input": "in", "output": "out"}])
    assert res["backend"] == "pytorch"
    assert res["model"] == "foo"
    assert "Input: in" in res["few_shot_prompt"]


def test_build_few_shot_prompt_pytorch_missing():
    with mock.patch.dict(sys.modules, {"torch": None}):
        import importlib

        importlib.reload(fs)
        res = fs.build_few_shot_prompt("foo", "prompt", [])
        assert res["status"] == "mocked_missing_pytorch"
