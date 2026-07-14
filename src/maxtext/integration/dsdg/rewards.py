"""DSDG reward functions for maxtext/Tunix RL.

Loaded via maxtext's `reward_functions_path` / `reward_functions` config, e.g.:

    reward_functions_path=src/maxtext/integration/dsdg/rewards.py
    reward_functions=dsdg_local_reward

Signature matches maxtext's built-in reward functions
(`match_format_exactly` etc. in trainers/post_train/rl/utils_rl.py):

    fn(prompts, completions, answer, tmvp_config, **kwargs) -> list[float]

Two variants:
  - ``dsdg_local_reward``  : self-contained format + answer-match, no network.
                            Use for a first smoke run (proves the reward runs
                            inside the GRPO loop on TPU).
  - ``dsdg_env_reward``    : POSTs each completion to the DSDG env-server
                            ``/v1/env/score`` (URL from ``DSDG_ENV_SCORE_URL``).
                            Use once the env-server is reachable from the pod;
                            this is the modular reward seam described in
                            dsdg docs/superpowers/specs/2026-07-13-tunix-env-spike.md.
"""

from __future__ import annotations

import os
import re
from typing import Any

_NON_EMPTY = re.compile(r"\S")


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def dsdg_local_reward(
    prompts: list[str],
    completions: list[str],
    answer: list[str] | None = None,
    tmvp_config: Any = None,
    **kwargs: Any,
) -> list[float]:
    """0.1 for non-empty output + 1.0 for a normalized answer match."""
    answers = answer or []
    rewards: list[float] = []
    for index, completion in enumerate(completions):
        fmt = 0.1 if _NON_EMPTY.search(completion or "") else 0.0
        expected = _normalize(answers[index]) if index < len(answers) else ""
        match = 1.0 if expected and _normalize(completion) == expected else 0.0
        rewards.append(match + fmt)
    return rewards


def dsdg_env_reward(
    prompts: list[str],
    completions: list[str],
    answer: list[str] | None = None,
    tmvp_config: Any = None,
    **kwargs: Any,
) -> list[float]:
    """Score each completion via the DSDG env-server /v1/env/score endpoint.

    Requires env var ``DSDG_ENV_SCORE_URL`` (e.g. http://dsdg-env-server:8080/v1/env/score).
    """
    import httpx  # deferred: only the env-server variant needs it

    score_url = os.environ["DSDG_ENV_SCORE_URL"]
    answers = answer or []
    rewards: list[float] = []
    with httpx.Client(timeout=30.0) as client:
        for index, completion in enumerate(completions):
            info = {"answer": answers[index]} if index < len(answers) else {}
            response = client.post(
                score_url,
                json={
                    "messages": [{"role": "assistant", "content": completion}],
                    "info": info,
                },
            )
            response.raise_for_status()
            rewards.append(float(response.json()["reward"]))
    return rewards
