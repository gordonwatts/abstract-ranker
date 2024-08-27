from dataclasses import dataclass
from typing import List
from unittest.mock import patch

from pydantic import ValidationError
import pytest


@dataclass
class message:
    content: str


@dataclass
class choice:
    message: message


@dataclass
class response:
    choices: List[choice]


def test_openai_simple_call():
    with patch("openai.OpenAI") as mock_openai:
        with patch("abstract_ranker.openai_utils.get_key") as mock_get_key:
            mock_get_key.return_value = "bogus_key"

            mock_openai.return_value.chat.completions.create.return_value = response(
                choices=[
                    choice(
                        message=message(
                            content='{"summary": "hi", "experiment": "", "keywords": [], '
                            '"interest": "", "explanation": "", "confidence": 0.5, '
                            '"unknown_terms": []}'
                        )
                    )
                ]
            )

            from abstract_ranker.openai_utils import query_gpt
            from abstract_ranker.llm_utils import AbstractLLMResponse

            r = query_gpt(
                "hi",
                {
                    "title": "hi",
                    "abstract": "hi",
                    "interested_topics": ["hi", "there"],
                    "not_interested_topics": ["no", "thanks"],
                },
                "gpt-4-turbo-bogus",
            )
            assert isinstance(r, AbstractLLMResponse)
            assert r.summary == "hi"
            assert r.experiment == ""
            assert r.keywords == []
            assert r.interest == ""
            assert r.explanation == ""

            mock_openai.assert_called_once()
            mock_openai.return_value.chat.completions.create.assert_called_once()
            assert (
                mock_openai.return_value.chat.completions.create.call_args[1]["model"]
                == "gpt-4-turbo-bogus"
            )


def test_openai_json_header_trailer_removal():
    with patch("openai.OpenAI") as mock_openai:
        with patch("abstract_ranker.openai_utils.get_key") as mock_get_key:
            mock_get_key.return_value = "bogus_key"

            mock_openai.return_value.chat.completions.create.return_value = response(
                choices=[
                    choice(
                        message=message(
                            content='json\n{"summary": "hi", "experiment": "", "keywords": [], '
                            '"interest": "", "explanation": "", "confidence": 0.5, '
                            '"unknown_terms": []}```'
                        )
                    )
                ]
            )

            from abstract_ranker.openai_utils import query_gpt
            from abstract_ranker.llm_utils import AbstractLLMResponse

            r = query_gpt(
                "hi",
                {
                    "title": "hi",
                    "abstract": "hi",
                    "interested_topics": ["hi", "there"],
                    "not_interested_topics": ["no", "thanks"],
                },
                "gpt-4-turbo-bogus",
            )
            assert isinstance(r, AbstractLLMResponse)
            assert r.summary == "hi"
            assert r.experiment == ""
            assert r.keywords == []
            assert r.interest == ""
            assert r.explanation == ""

            mock_openai.assert_called_once()
            mock_openai.return_value.chat.completions.create.assert_called_once()
            assert (
                mock_openai.return_value.chat.completions.create.call_args[1]["model"]
                == "gpt-4-turbo-bogus"
            )


def test_openai_bad_response():
    with patch("openai.OpenAI") as mock_openai:
        with patch("abstract_ranker.openai_utils.get_key") as mock_get_key:
            mock_get_key.return_value = "bogus_key"

            mock_openai.return_value.chat.completions.create.return_value = response(
                choices=[choice(message=message(content="fork it"))]
            )

            from abstract_ranker.openai_utils import query_gpt

            with pytest.raises(ValidationError) as e:
                query_gpt(
                    "hi",
                    {
                        "title": "hi",
                        "abstract": "hi",
                        "interested_topics": ["hi", "there"],
                        "not_interested_topics": ["no", "thanks"],
                    },
                    "gpt-4-turbo-bogus",
                )

            assert "validation error" in str(e.value)


def test_openai_latex_response():
    with patch("openai.OpenAI") as mock_openai:
        with patch("abstract_ranker.openai_utils.get_key") as mock_get_key:
            mock_get_key.return_value = "bogus_key"

            mock_openai.return_value.chat.completions.create.return_value = response(
                choices=[
                    choice(
                        message=message(
                            content="""{
    "summary": "Not given",
    "experiment": "",
    "keywords": ["b to s transitions", "flavor physics", "lepton interactions"],
    "interest": "low",
    "explanation": "The topic of $b \\to s \\ell \\ell$ fits does not align with my interests \
in hidden sector physics or long-lived particles.",
    "confidence": 0.3,
    "unknown_terms": []
}
"""
                        )
                    )
                ]
            )

            from abstract_ranker.openai_utils import query_gpt

            r = query_gpt(
                "hi",
                {
                    "title": "hi",
                    "abstract": "hi",
                    "interested_topics": ["hi", "there"],
                    "not_interested_topics": ["no", "thanks"],
                },
                "gpt-4-turbo-bogus",
            )

            assert (
                r.explanation
                == "The topic of $b \\to s \\ell \\ell$ fits does not align with my interests "
                "in hidden sector physics or long-lived particles."
            )
