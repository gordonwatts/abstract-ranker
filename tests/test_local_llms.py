from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from abstract_ranker.config import abstract_ranking_prompt
from abstract_ranker.local_llms import query_hugging_face


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


@pytest.mark.skip("This test uses phi-3 and is too expensive to run all the time")
def test_CaloDiT_phi3():
    "This abstract summary was failing in the wild"

    title = "CaloDiT: Diffusion with transformers for fast shower simulation"
    abstract = """
Recently, transformers have proven to be a generalised architecture for various data modalities,
i.e., ranging from text (BERT, GPT3), time series (PatchTST) to images (ViT) and even a
combination of them (Dall-E 2, OpenAI Whisper). Additionally, when given enough data, transformers
can learn better representations than other deep learning models thanks to the absence of
inductive bias, better modelling of long-range dependencies, and interpolation and extrapolation
capabilities. On the other hand, diffusion models are the state-of-the-art approach for image
generation, which still use conventional U-net models for generation, mostly consisting of
convolution layers making little use of the advantages of transformers. While these models show
good generation performance it lacks the generalisation capabilities obtained from the transformer
model. Standard diffusion models with an Unet architecture have already proven to be able to
generate calorimeter showers, while transformer-based models, like those based on a VQ-VAE
architecture, also show promising results. A combination of a diffusion model with a transformer
architecture should bridge the quality of the generation sample obtained from diffusion with the
generalisation capabilities of the transformer architecture. In this paper, we propose CaloDiT, to
model our problem as a diffusion process with transformer blocks. Furthermore, we show the ability
of the model to generalise to different calorimeter geometries, bringing us closer to a foundation
model for calorimeter shower generation."""
    abstract = abstract.replace("\n", " ")

    r = query_hugging_face(
        abstract_ranking_prompt,
        {"title": title, "abstract": abstract},
        "microsoft/Phi-3-mini-4k-instruct",
    )

    assert r.experiment.count(" ") == 0
