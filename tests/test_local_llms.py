from unittest.mock import patch

from abstract_ranker.local_llms import query_hugging_face


def test_hf():
    # Mock out the call to the transformers library pipeline call:
    with patch("transformers.pipeline") as mock_pipeline:
        call_message = None

        def call_back(msg: str) -> str:
            nonlocal call_message
            assert isinstance(msg, list)
            assert len(msg) == 1
            call_message = msg[0]
            return [{'generated_text': }]

        mock_pipeline.return_value = call_back

        result = query_hugging_face(
            "What is the summary?",
            {"title": "Title", "abstract": "Abstract"},
            "microsoft/Phi-3-mini-4k-instruct",
        )
        # Check the result
        assert result == "forking fork"
        # Check the call
        mock_pipeline.assert_called_once_with(
            "text-generation",
            model="microsoft/Phi-3-mini-4k-instruct",
            trust_remote_code=True,
        )

        assert call_message is not None
        assert isinstance(call_message, dict)
        assert (
            call_message["content"]
            == """What is the summary?
Title: Title
Abstract: Abstract"""
        )
        assert (
            call_message["role"]
            == "All of your answers will be parsed by a yaml parser. Format answer as yaml "
            "(all output must be yaml). If the user gives you a template for the yaml, please "
            "use that (it will be in between ```yaml <template> ```."
        )


def test_hf_twice():
    "Make sure we do not create the same model twice"
    with patch("transformers.pipeline") as mock_pipeline:
        pipe_call_count = 0

        def call_back(msg: str) -> str:
            nonlocal pipe_call_count
            pipe_call_count += 1
            return "forking fork"

        mock_pipeline.return_value = call_back

        # Call twice!!
        _ = query_hugging_face(
            "What is the summary?",
            {"title": "Title", "abstract": "Abstract"},
            "microsoft/Phi-3-mini-4k-instruct",
        )
        _ = query_hugging_face(
            "What is the summary?",
            {"title": "Title", "abstract": "Abstract"},
            "microsoft/Phi-3-mini-4k-instruct",
        )
        mock_pipeline.assert_called_once()
        assert pipe_call_count == 2


def test_hf_two_models():
    "Make sure we do not create the same model twice"
    with patch("transformers.pipeline") as mock_pipeline:
        pipe_call_count = 0

        def call_back(msg: str) -> str:
            nonlocal pipe_call_count
            pipe_call_count += 1
            return "forking fork"

        mock_pipeline.return_value = call_back

        # Call twice!!
        _ = query_hugging_face(
            "What is the summary?",
            {"title": "Title", "abstract": "Abstract"},
            "microsoft/Phi-3-mini-4k-instruct",
        )
        _ = query_hugging_face(
            "What is the summary?",
            {"title": "Title", "abstract": "Abstract"},
            "microsoft/Phi-3-mini-4k-instruct_t",
        )
        assert mock_pipeline.call_count == 2
        assert pipe_call_count == 2
