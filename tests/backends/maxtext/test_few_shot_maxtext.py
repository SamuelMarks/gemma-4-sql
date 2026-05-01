import sys
from unittest import mock

import gemma_4_sql.backends.maxtext.few_shot as fs


def test_build_few_shot_prompt_maxtext():
    res = fs.build_few_shot_prompt("foo", "prompt", [{"input": "in", "output": "out"}])
    assert res["backend"] == "maxtext"
    assert res["model"] == "foo"


def test_build_few_shot_prompt_maxtext_missing():
    with mock.patch.dict(
        sys.modules,
        {"maxtext.models.gemma4": None, "maxtext.models": None, "maxtext": None},
    ):
        import importlib

        importlib.reload(fs)
        res = fs.build_few_shot_prompt("foo", "prompt", [])
        assert res["status"] == "mocked_missing_maxtext"
