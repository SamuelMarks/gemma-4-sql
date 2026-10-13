import argparse

from gemma_4_sql.cli_misc import log_metrics_cmd


def test_log_metrics_cmd_with_metrics(monkeypatch, capsys):
    args = argparse.Namespace(metrics="loss=1.0,acc=0.9", step=1, log_dir="logs", backend="dummy")

    def mock_log_metrics(**kwargs):
        assert kwargs["metrics"] == {"loss": 1.0, "acc": 0.9}
        return {"status": "logged"}

    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", mock_log_metrics)
    log_metrics_cmd(args)
    captured = capsys.readouterr()
    assert "logged" in captured.out


def test_log_metrics_cmd_without_metrics(monkeypatch, capsys):
    args = argparse.Namespace(metrics="", step=1, log_dir="logs", backend="dummy")

    def mock_log_metrics(**kwargs):
        assert kwargs["metrics"] == {}
        return {"status": "logged_empty"}

    monkeypatch.setattr("gemma_4_sql.cli_misc.log_metrics", mock_log_metrics)
    log_metrics_cmd(args)
    captured = capsys.readouterr()
    assert "logged_empty" in captured.out


def test_quantize_cmd(monkeypatch, capsys):
    args = argparse.Namespace(model="dummy", method="dummy", backend="dummy")
    monkeypatch.setattr("gemma_4_sql.cli_misc.quantize_model", lambda *a, **k: {"status": "q"})
    from gemma_4_sql.cli_misc import quantize_cmd

    quantize_cmd(args)
    assert '"status": "q"' in capsys.readouterr().out


def test_export_cmd(monkeypatch, capsys):
    args = argparse.Namespace(model="dummy", path="dummy", backend="dummy")
    monkeypatch.setattr("gemma_4_sql.cli_misc.export_model", lambda *a, **k: {"status": "e"})
    from gemma_4_sql.cli_misc import export_cmd

    export_cmd(args)
    assert '"status": "e"' in capsys.readouterr().out


def test_tokenize_cmd_encode(monkeypatch, capsys):
    args = argparse.Namespace(hf_model=None, vocab_size=256, encode="test", decode=None)
    monkeypatch.setattr("gemma_4_sql.cli_misc.SQLTokenizer.encode", lambda self, text: [1, 2, 3])
    from gemma_4_sql.cli_misc import tokenize_cmd

    tokenize_cmd(args)
    assert "[1, 2, 3]" in capsys.readouterr().out


def test_tokenize_cmd_decode(monkeypatch, capsys):
    args = argparse.Namespace(hf_model=None, vocab_size=256, encode=None, decode="1, 2, 3")
    monkeypatch.setattr("gemma_4_sql.cli_misc.SQLTokenizer.decode", lambda self, tokens: "test")
    from gemma_4_sql.cli_misc import tokenize_cmd

    tokenize_cmd(args)
    assert "test" in capsys.readouterr().out


def test_tokenize_cmd_decode_invalid(capsys):
    args = argparse.Namespace(hf_model=None, vocab_size=256, encode=None, decode="invalid")
    from gemma_4_sql.cli_misc import tokenize_cmd

    tokenize_cmd(args)


def test_rag_cmd_extract(capsys):
    args = argparse.Namespace(action="extract", ddl="test")
    from gemma_4_sql.cli_misc import rag_cmd

    rag_cmd(args)


def test_rag_cmd_retrieve(capsys):
    args = argparse.Namespace(action="retrieve", ddl="test", prompt="p")
    from gemma_4_sql.cli_misc import rag_cmd

    rag_cmd(args)


def test_rag_cmd_with_images_and_audio(monkeypatch, capsys):
    import argparse

    from gemma_4_sql.cli_misc import rag_cmd

    args = argparse.Namespace(action="build", prompt="Test prompt", ddl="Test DDL", image_path="test.jpg", audio_path="test.mp3")

    def mock_build_rag_prompt(prompt, ddl):
        return f"{prompt} with {ddl}"

    monkeypatch.setattr("gemma_4_sql.cli_misc.build_rag_prompt", mock_build_rag_prompt)
    rag_cmd(args)
    captured = capsys.readouterr()
    assert "Test prompt with Test DDL" in captured.out
