from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def pipeline_callback():
    with patch("abstract_ranker.local_llms.create_pipeline") as mock_pipeline:
        with patch(
            "lmformatenforcer.integrations.transformers."
            "build_transformers_prefix_allowed_tokens_fn"
        ) as build_trans:
            call_message: Optional[List[Dict[str, str]]] = None

            class pipe_callback:

                def __init__(self):
                    self.tokenizer = MagicMock()

                def __call__(
                    self, msg: List[Dict[str, str]], **kwargs
                ) -> List[Dict[str, str]]:
                    nonlocal call_message
                    assert isinstance(msg, list)
                    assert len(msg) == 2
                    call_message = msg
                    return [
                        {
                            "generated_text": '{"summary": "forking fork", "experiment": "", '
                            '"keywords": [], "interest": "high", "explanation": "because"}'
                        }
                    ]

            mock_pipeline.return_value = pipe_callback()

            build_trans.return_value = None

            # Yield back so we keep the patch
            yield mock_pipeline


# setup_before_test
def test_hf(pipeline_callback, setup_before_test):
    # Mock out the call to the transformers library pipeline call:
    from abstract_ranker.local_llms import query_hugging_face

    result = query_hugging_face(
        "What is the summary?",
        {"title": "Title", "abstract": "Abstract"},
        "microsoft/Phi-3-mini-4k-instruct",
    )
    # Check the result
    assert result.summary == "forking fork"
    # Check the call
    pipeline_callback.assert_called_once_with("microsoft/Phi-3-mini-4k-instruct")


# Disabled b.c. the caching mechanism doesn't seem to be testing right.
# def test_hf_twice(pipeline_callback, setup_before_test):
#     "Make sure we do not create the same model twice"
#     # Call twice!!
#     from abstract_ranker.local_llms import query_hugging_face

#     _ = query_hugging_face(
#         "What is the summary?",
#         {"title": "Title", "abstract": "Abstract"},
#         "microsoft/Phi-3-mini-4k-instruct",
#     )
#     _ = query_hugging_face(
#         "What is the summary?",
#         {"title": "Title", "abstract": "Abstract"},
#         "microsoft/Phi-3-mini-4k-instruct",
#     )

#     pipeline_callback.assert_called_once()


# def test_hf_two_models(pipeline_callback, setup_before_test):
#     "Make sure we create different models"
#     from abstract_ranker.local_llms import query_hugging_face

#     # Call twice with different model names
#     _ = query_hugging_face(
#         "What is the summary?",
#         {"title": "Title", "abstract": "Abstract"},
#         "microsoft/Phi-3-mini-4k-instruct",
#     )
#     _ = query_hugging_face(
#         "What is the summary?",
#         {"title": "Title", "abstract": "Abstract"},
#         "microsoft/Phi-3-mini-4k-instruct_t",
#     )

#     assert pipeline_callback.call_count == 2
