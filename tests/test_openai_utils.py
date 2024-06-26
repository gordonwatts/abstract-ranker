from dataclasses import dataclass
from typing import List
from unittest.mock import patch


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
                            '"interest": "", "explanation": ""}'
                        )
                    )
                ]
            )

            from abstract_ranker.openai_utils import query_gpt
            from abstract_ranker.llm_utils import AbstractLLMResponse

            r = query_gpt("hi", {"title": "hi", "abstract": "hi"}, "gpt-4-turbo-bogus")
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
