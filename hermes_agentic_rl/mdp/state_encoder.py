from __future__ import annotations

from typing import Any

from hermes_agentic_rl.backends.base import TokenizerProtocol
from hermes_agentic_rl.mdp.observation import Observation


class PromptStateEncoder:
    """Encodes `(task_item, conversation_so_far)` → tokenized Observation.

    For the MVP we format:
        <bos> <instruction>\n Answer:

    which gives the policy a clear "continue here" position. More elaborate
    chat-templating lives in future iterations.
    """

    def __init__(
        self, tokenizer: TokenizerProtocol, template: str = "{instruction}\nAnswer:"
    ) -> None:
        self.tokenizer = tokenizer
        self.template = template

    def encode(self, item: dict[str, Any]) -> Observation:
        instruction = str(item.get("instruction", ""))
        text = self.template.format(instruction=instruction)
        ids = [self.tokenizer.bos_id, *self.tokenizer.encode(text)]
        return Observation(prompt_ids=ids, text=text, metadata={"instruction": instruction})
