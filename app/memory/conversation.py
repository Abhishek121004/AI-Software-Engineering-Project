from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str


@dataclass
class ConversationMemory:
    """Simple inspectable memory for follow-up repository questions."""

    turns: List[ConversationTurn] = field(default_factory=list)
    max_turns: int = 12

    def add_user_message(self, content: str) -> None:
        self.turns.append(ConversationTurn(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self.turns.append(ConversationTurn(role="assistant", content=content))
        self._trim()

    def _trim(self) -> None:
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def recent_context(self, max_turns: int = 4) -> str:
        recent = self.turns[-max_turns:]
        return "\n".join(f"{turn.role}: {turn.content}" for turn in recent)

    def resolve_follow_up(self, question: str) -> str:
        if not self.turns:
            return question

        if any(token in question.lower() for token in ("it ", "that ", "this ", "the token", "the function", "the file")):
            for turn in reversed(self.turns):
                if turn.role == "assistant":
                    return f"{question}\n\nEarlier context:\n{turn.content}"
        return question

