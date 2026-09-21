"""Loading versioned prompt files.

A prompt that has been used in a run is never edited -- a new version is added
instead, and every LLM call records the ``prompt_version`` it used. Without that,
results from different days are not comparable and the experiment is worthless.

The loader hashes each template so an accidental edit to a used prompt is
detectable after the fact rather than silent.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def prompt_dir() -> Path:
    return Path(os.environ.get("BETSIM_PROMPT_DIR", DEFAULT_PROMPT_DIR))


@dataclass(frozen=True, slots=True)
class Prompt:
    version: str
    template: str
    sha256: str

    def render(self, **fields: object) -> str:
        """Fill the template. Raises if a placeholder has no value."""
        try:
            return self.template.format(**fields)
        except KeyError as exc:
            raise KeyError(f"prompt {self.version!r} needs a value for {exc}") from None


@cache
def load_prompt(version: str) -> Prompt:
    """Load ``prompts/<version>.md``."""
    path = prompt_dir() / f"{version}.md"
    try:
        template = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        available = sorted(p.stem for p in prompt_dir().glob("*.md"))
        raise FileNotFoundError(
            f"no prompt {version!r} in {prompt_dir()}; available: {available}"
        ) from None
    return Prompt(
        version=version,
        template=template,
        sha256=hashlib.sha256(template.encode("utf-8")).hexdigest(),
    )
