# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest

from heretic.config import Settings
from heretic.utils import closed_cot_response_prefix

SKIPS = Settings.model_fields["chain_of_thought_skips"].default


class ClosedCotResponsePrefixTests(unittest.TestCase):
    def test_returns_none_when_template_opens_no_thinking_block(self) -> None:
        self.assertIsNone(
            closed_cot_response_prefix("<|user|>hi<|assistant|>", SKIPS)
        )

    def test_closes_a_block_the_template_already_opened(self) -> None:
        # The template emits the opening tag, so only the closing tag belongs in
        # the prefix. Returning the whole block would render '<think><think></think>'.
        self.assertEqual(closed_cot_response_prefix("assistant\n<think>", SKIPS), "</think>")

    def test_tolerates_whitespace_after_the_opening_tag(self) -> None:
        self.assertEqual(closed_cot_response_prefix("assistant\n<think>\n", SKIPS), "</think>")

    def test_prefix_does_not_repeat_the_opening_tag(self) -> None:
        for dummy_prompt in ("assistant\n<think>", "assistant\n<think>\n", "x <think>  "):
            with self.subTest(dummy_prompt=dummy_prompt):
                prefix = closed_cot_response_prefix(dummy_prompt, SKIPS)

                self.assertIsNotNone(prefix)
                self.assertNotIn("<think>", prefix)

    def test_rendered_prompt_contains_exactly_one_opening_tag(self) -> None:
        dummy_prompt = "<|user|>hi<|assistant|>\n<think>\n"

        rendered = dummy_prompt + closed_cot_response_prefix(dummy_prompt, SKIPS)

        self.assertEqual(rendered.count("<think>"), 1)
        self.assertEqual(rendered.count("</think>"), 1)

    def test_keeps_the_remainder_for_multi_token_openers(self) -> None:
        # gpt-oss: the template opens the analysis channel, so the prefix must carry
        # everything after it and not reopen the channel.
        dummy_prompt = "<|start|>assistant<|channel|>analysis<|message|>"

        prefix = closed_cot_response_prefix(dummy_prompt, SKIPS)

        self.assertEqual(prefix, "<|end|><|start|>assistant<|channel|>final<|message|>")
        self.assertEqual((dummy_prompt + prefix).count("<|channel|>analysis<|message|>"), 1)

    def test_returns_the_whole_block_when_it_does_not_repeat_the_opener(self) -> None:
        self.assertEqual(
            closed_cot_response_prefix("a <reason>", [("<reason>", "DONE")]), "DONE"
        )


if __name__ == "__main__":
    unittest.main()
