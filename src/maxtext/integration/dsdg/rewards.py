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

import json
import os
import re
from typing import Any

_NON_EMPTY = re.compile(r"\S")
# Default maxtext solution tags (tmvp_config.solution_start/end_token). The
# model emits its final answer inside <answer>...</answer>, so match against
# that span rather than the whole completion (which is full reasoning).
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_BOXED = re.compile(r"\\boxed\s*\{?\s*([^{}]+?)\s*\}?(?:\$|\s|$)")


def _to_list(value: Any) -> list[Any]:
    """Coerce maxtext's ``answer`` kwarg to a plain list.

    During eval it may arrive as a scalar/None; during training it arrives as
    a numpy array (a batch). ``array or []`` raises "truth value of an array is
    ambiguous", so normalize explicitly.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if hasattr(value, "tolist"):  # numpy array / jax array
        value = value.tolist()
    try:
        return list(value)
    except TypeError:
        return [value]


def _parse_gold(value: Any) -> list[str]:
    """Return the acceptable gold answers for one example.

    maxtext passes gold answers JSON-encoded (see ``check_numbers`` in
    utils_rl.py: ``json.loads(acceptable_answers)``), e.g. the string
    ``'["60"]'`` — so a raw string compare never matches. Parse the JSON list;
    fall back to treating the value as a single literal answer.
    """
    if value is None:
        return []
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    text = str(value)
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return [text]
    if isinstance(parsed, (list, tuple)):
        return [str(v) for v in parsed]
    return [str(parsed)]


def _normalize(text: Any) -> str:
    """Lower/strip + drop thousands commas + collapse ``2.0`` -> ``2``."""
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    s = str(text).strip().lower().replace(",", "").replace("$", "")
    try:  # integer-float collapse so "2.0" == "2" (mirrors normalize_final_answer)
        f = float(s)
        if abs(f - round(f)) < 1e-7:
            s = str(int(round(f)))
    except (ValueError, OverflowError):
        pass
    return s


def _extract_answer(completion: Any, tmvp_config: Any = None) -> str:
    """Extract the model's final answer.

    Prefer maxtext's own ``extract_answer`` (identical to the grader) when a
    ``tmvp_config`` is available; otherwise fall back to a self-contained scan:
    last ``<answer>...</answer>`` span, then a ``\\boxed{...}`` inside it.
    """
    text = "" if completion is None else str(completion)
    if tmvp_config is not None:
        try:
            from maxtext.trainers.post_train.rl.utils_rl import extract_answer as _mx_extract

            return _mx_extract(text, tmvp_config)
        except Exception:  # keep the module usable/testable outside maxtext
            pass
    matches = _ANSWER_TAG.findall(text)
    scope = matches[-1] if matches else text
    boxed = _BOXED.findall(scope)
    if boxed:
        return boxed[-1]
    return scope


def dsdg_local_reward(
    prompts: list[str],
    completions: list[str],
    answer: list[str] | None = None,
    tmvp_config: Any = None,
    **kwargs: Any,
) -> list[float]:
    """0.1 for non-empty output + 1.0 for a correct answer.

    Answer-match agrees with maxtext's grader: JSON-decoded gold answers vs the
    ``extract_answer``-ed guess, compared after numeric normalization.
    """
    answers = _to_list(answer)
    rewards: list[float] = []
    for index, completion in enumerate(completions):
        fmt = 0.1 if _NON_EMPTY.search(str(completion or "")) else 0.0
        golds = _parse_gold(answers[index]) if index < len(answers) else []
        guess = _normalize(_extract_answer(completion, tmvp_config))
        match = 1.0 if guess and any(guess == _normalize(g) for g in golds) else 0.0
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
    answers = _to_list(answer)
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
