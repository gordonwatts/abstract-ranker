from unittest.mock import patch


def test_dispatch():
    with patch("abstract_ranker.openai_utils.query_gpt") as mock_query_gpt:
        from abstract_ranker.llm_utils import AbstractLLMResponse

        mock_query_gpt.return_value = AbstractLLMResponse(
            summary="yes or no, you'll have to find out",
            experiment="hi",
            keywords=["hi"],
            interest="high",
            explanation="hi",
        )

        context = {"title": "hi"}
        prompt = "hi"

        from abstract_ranker.llm_utils import query_llm

        r = query_llm(prompt, context, "GPT4Turbo")
        assert isinstance(r, AbstractLLMResponse)
        assert r.interest == "high"
        assert r.summary == "yes or no, you'll have to find out"

        assert mock_query_gpt.call_count == 1
        mock_query_gpt.assert_called_with(prompt, context, "gpt-4-turbo")
