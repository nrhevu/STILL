from types import SimpleNamespace

from neural_kv.eval.ruler_coverage import row_coverage, summarize_coverage


class WhitespaceTokenizer:
    chat_template = "chat"

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[self._id(token) for token in text.split()])

    def decode(self, ids):
        return " ".join(f"tok{item}" for item in ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs,
    ):
        del tokenize, add_generation_prompt, kwargs
        return messages[0]["content"]

    @staticmethod
    def _id(token: str) -> int:
        return abs(hash(token)) % 1000003


def test_row_coverage_passes_visible_short_target_line() -> None:
    row = {
        "id": "ok",
        "context": "filler words\nkey alpha value ANSWER_TOKEN\nmore filler",
        "target_line": "key alpha value ANSWER_TOKEN",
        "answer": "ANSWER_TOKEN",
        "ruler_task": "niah_single",
    }

    result = row_coverage(
        WhitespaceTokenizer(),
        row,
        context_length=64,
        exact_tokens=8,
    )

    assert result["passes"] is True
    assert result["target_token_count"] == 4
    assert summarize_coverage([result])["failed"] == 0


def test_row_coverage_fails_when_target_line_exceeds_exact_tokens() -> None:
    row = {
        "id": "too-long",
        "context": "alpha beta gamma delta epsilon",
        "target_line": "alpha beta gamma delta epsilon",
        "answer": "epsilon",
        "ruler_task": "qa",
    }

    result = row_coverage(
        WhitespaceTokenizer(),
        row,
        context_length=64,
        exact_tokens=4,
    )

    assert result["passes"] is False
    assert "target_line_exceeds_exact_tokens" in result["failures"]
