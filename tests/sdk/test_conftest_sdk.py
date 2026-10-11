def test_mock_tokenizer(mock_tokenizer):
    assert mock_tokenizer.encode("foo") == [1, 2, 3]
    assert mock_tokenizer.decode([1, 2, 3]) == "SELECT 1;"


def test_mock_model_evaluator(mock_model_evaluator):
    assert mock_model_evaluator.evaluate("m", "d") == {"exact_match": 1.0, "execution_accuracy": 1.0}


def test_in_memory_batch_generator(in_memory_batch_generator):
    assert in_memory_batch_generator == {"inputs": [1, 2], "targets": [2, 3]}
