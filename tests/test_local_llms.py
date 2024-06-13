from typing import Dict, List, Optional
from unittest.mock import patch

from abstract_ranker.local_llms import query_hugging_face


def test_hf():
    # Mock out the call to the transformers library pipeline call:
    with patch("abstract_ranker.local_llms.create_pipeline") as mock_pipeline:
        call_message: Optional[List[Dict[str, str]]] = None

        def call_back(msg: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
            nonlocal call_message
            assert isinstance(msg, list)
            assert len(msg) == 2
            call_message = msg
            return [{"generated_text": "forking fork"}]

        mock_pipeline.return_value = call_back

        result = query_hugging_face(
            "What is the summary?",
            {"title": "Title", "abstract": "Abstract"},
            "microsoft/Phi-3-mini-4k-instruct",
        )
        # Check the result
        assert result == "forking fork"
        # Check the call
        mock_pipeline.assert_called_once_with("microsoft/Phi-3-mini-4k-instruct")

        assert call_message is not None
        for item in call_message:  # type: ignore
            if item["role"] == "system":
                assert item["content"].startswith("All of your answers will")
            elif item["role"] == "user":
                assert (
                    item["content"]
                    == """What is the summary?
Title: Title
Abstract: Abstract"""
                )
            else:
                assert False, f"Unknown role: {item['role']}"


def test_hf_twice():
    "Make sure we do not create the same model twice"
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as auto_causal_mock:
        with patch("transformers.AutoTokenizer.from_pretrained") as auto_token_mock:
            with patch("transformers.pipeline") as pipeline_mock:

                auto_causal_mock.return_value = "model"
                auto_token_mock.return_value = "tokenizer"

                pipe_call_count = 0

                def call_back(msg: str, **_) -> List[Dict[str, str]]:
                    nonlocal pipe_call_count
                    pipe_call_count += 1
                    return [{"generated_text": "forking fork"}]

                pipeline_mock.return_value = call_back

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
                pipeline_mock.assert_called_once()
                assert pipe_call_count == 2


def test_hf_two_models():
    "Make sure we create different models"
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as auto_causal_mock:
        with patch("transformers.AutoTokenizer.from_pretrained") as auto_token_mock:
            with patch("transformers.pipeline") as pipeline_mock:

                auto_causal_mock.return_value = "model"
                auto_token_mock.return_value = "tokenizer"

                pipe_call_count = 0

                def call_back(msg: str, **_) -> List[Dict[str, str]]:
                    nonlocal pipe_call_count
                    pipe_call_count += 1
                    return [{"generated_text": "forking fork"}]

                pipeline_mock.return_value = call_back

                # Call twice with different model names
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

        assert pipeline_mock.call_count == 2
        assert pipe_call_count == 2
