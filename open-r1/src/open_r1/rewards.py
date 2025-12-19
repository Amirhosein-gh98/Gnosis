# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Literal, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils.code_providers import get_provider
from .utils.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)
from .utils.competitive_programming import patch_code as cf_patch_code
from .utils.competitive_programming import score_submission as cf_score_submission
from .utils.competitive_programming import score_subtask


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            if len(answer_parsed) ==0:
                reward = -1
            else:
                # Compute binary rewards if verifiable, `None` otherwise to skip this example
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    # print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                    reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            # print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def cf_code_reward(
    completions,
    test_batch_size: int = 1,
    patch_code: bool = False,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    **kwargs,
) -> list[float]:
    """Reward function that evaluates Codeforces problems using Piston+our CF package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/codeforces (verifiable-prompts subset)

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
    code_snippets = [
        # note: grading is automatically skipped if a problem has no tests
        cf_patch_code(extract_code(completion[-1]["content"], language), language)
        if patch_code
        else extract_code(completion[-1]["content"], language)
        for completion, language in zip(completions, languages)
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return None

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                cf_score_submission(
                    piston_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                    scoring_mode=scoring_mode,
                    submission_language=problem_data.get("language", None),
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return results


def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    template = evaluation_script_template

    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    def code_format_reward(completions, **kwargs):
        # if there is a language field, use it instead of the default language. This way we can have mixed language training.
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(
                rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content, sample_language in zip(completion_contents, languages)
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


def trivia_reward(completions, solution, **kwargs):
    """
    Reward = 1.0 if ANY \\boxed{...} extracted from the last assistant message
    matches ANY alias/value from the gold dict (already a Python dict), else 0.0.
    Only considers \\boxed{...} (no other extraction paths).
    Logs per-rank to /home/amirhosein/codes/debug and prints from rank 0.
    """
    import re, unicodedata, os, sys, time, socket

    # -------- fixed shared debug directory --------
    DEBUG_DIR = "/home/amirhosein/codes/debug"
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # -------- rank/world detection (DDP/Accelerate/env) --------
    def _rank_world():
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass
        try:
            from accelerate.state import AcceleratorState
            st = AcceleratorState()
            return st.process_index, st.num_processes
        except Exception:
            pass
        return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))

    RANK, WORLD = _rank_world()
    IS_MAIN = (RANK == 0)
    HOST = socket.gethostname()
    log_path = os.path.join(DEBUG_DIR, f"reward_debug_{HOST}_rank{RANK}.log")

    def _log(line: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{ts}] rank={RANK}/{WORLD} {line}\n"
        try:
            with open(log_path, "a", encoding="utf-8", buffering=1) as f:
                f.write(msg); f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
        if IS_MAIN:
            sys.stdout.write(msg); sys.stdout.flush()

    def _norm(s):
        if s is None:
            return ""
        s = unicodedata.normalize("NFKD", str(s)).casefold()
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _unwrap_latex_wrappers(s: str) -> str:
        if not s:
            return s
        pat = re.compile(r"\\(text|mathrm|operatorname)\s*\{(.+)\}\s*$", flags=re.DOTALL)
        for _ in range(3):
            m = pat.fullmatch(s.strip())
            if not m:
                break
            s = m.group(2)
        return s.strip()

    def _collapse_spelled_letters(s: str) -> str:
        # collapse things like 'k i n s h a s a' -> 'kinshasa'
        t = s.strip()
        if re.fullmatch(r'(?:[A-Za-z]\s+){2,}[A-Za-z]', t):
            return re.sub(r'\s+', '', t)
        return s

    def _clean_pred_text(p: str) -> str:
        if not p:
            return p
        p = _unwrap_latex_wrappers(p)
        p = p.strip().strip('\'"`“”’.,;:!-()[]')
        p = _collapse_spelled_letters(p)
        return p.strip()

    def _last_assistant_content(messages):
        if isinstance(messages, list) and messages:
            return next((m.get("content","") for m in reversed(messages) if m.get("role")=="assistant"),
                        messages[-1].get("content",""))
        return str(messages or "")

    def _extract_all_boxed(content: str):
        """
        Extract ALL balanced \\boxed{...} contents (handles nested braces).
        """
        results = []
        i = 0
        while True:
            m = re.search(r'\\boxed\s*\{', content[i:])
            if not m:
                break
            start_brace = i + m.end()  # position right after '{'
            depth, j = 1, start_brace
            while j < len(content) and depth > 0:
                ch = content[j]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                inner = content[start_brace:j-1]
                results.append(inner.strip())
                i = j
            else:
                # unmatched brace; stop parsing
                break
        return results

    rewards = []
    # _log("ENTRY: trivia_reward start")

    for messages, sol in zip(completions, solution):
        # --- gold candidates from dict ---
        if not isinstance(sol, dict):
            rewards.append(None)
            # _log("gold not dict; reward=None")
            continue

        cands = set()
        for k in ("value","normalized_value","matched_wiki_entity_name","normalized_matched_wiki_entity_name"):
            v = sol.get(k)
            if isinstance(v, str) and v.strip():
                cands.add(_norm(v))
        for k in ("aliases","normalized_aliases"):
            arr = sol.get(k, [])
            if isinstance(arr, list):
                for v in arr:
                    if isinstance(v, str) and v.strip():
                        cands.add(_norm(v))

        if not cands:
            rewards.append(None)
            # _log("no candidates; reward=None")
            continue

        content = _last_assistant_content(messages)
        boxed_items = _extract_all_boxed(messages)
        preds = [_clean_pred_text(b) for b in boxed_items if b.strip()]

        # dedupe and normalize
        seen, preds_norm = set(), []
        for p in preds:
            if p not in seen:
                seen.add(p)
                preds_norm.append(_norm(p))

        hit = any(pn in cands for pn in preds_norm)
        reward = 1.0 if hit else 0.0
        rewards.append(reward)


    return rewards



#Gnosis
# General reward that handles DAPO/LIMO (math), TriviaQA (dict), SciEval (["C"])
# Requires: latex2sympy2_extended, math_verify (as in your current pipeline)

import os, re, sys, time, socket, unicodedata
from typing import Optional, List, Dict, Any
import json
import re


# ------------------ shared helpers ------------------
def _last_assistant_content(messages: Any) -> str:
    """Return content text from the last assistant message; fallback to last content."""
    if isinstance(messages, list) and messages:
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return str(m.get("content", ""))
        return str(messages[-1].get("content", ""))
    return str(messages or "")

def _extract_all_boxed(content: str) -> List[str]:
    """Extract ALL balanced \\boxed{...} contents (handles nested braces)."""
    results, i, n = [], 0, len(content)
    while True:
        m = re.search(r'\\boxed\s*\{', content[i:])
        if not m:
            break
        start = i + m.end()
        depth, j = 1, start
        while j < n and depth > 0:
            ch = content[j]
            if ch == '{': depth += 1
            elif ch == '}': depth -= 1
            j += 1
        if depth == 0:
            results.append(content[start:j-1].strip())
            i = j
        else:
            break
    return results

def _unwrap_latex_wrappers(s: str) -> str:
    if not s: return s
    pat = re.compile(r"\\(text|mathrm|operatorname)\s*\{(.+)\}\s*$", flags=re.DOTALL)
    for _ in range(3):
        m = pat.fullmatch(s.strip())
        if not m: break
        s = m.group(2)
    return s.strip()

def _collapse_spelled_letters(s: str) -> str:
    # 'k i n s h a s a' -> 'kinshasa'
    t = s.strip()
    if re.fullmatch(r'(?:[A-Za-z]\s+){2,}[A-Za-z]', t):
        return re.sub(r'\s+', '', t)
    return s

def _clean_pred_text(p: str) -> str:
    if not p: return p
    p = _unwrap_latex_wrappers(p)
    p = p.strip().strip('\'"`“”’.,;:!-()[]')
    p = _collapse_spelled_letters(p)
    return p.strip()

def _norm(s: Any) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s)).casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ------------------ branch: DAPO/LIMO (math) ------------------
_MATH_EXTRACTION = [
    LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False, malformed_operators=False,
            basic_latex=True, equations=True, boxed="all", units=True
        ),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
]

def _reward_math(messages: Any, gold_str: str) -> Optional[float]:
    """Return -1 if pred unparseable, 0/1 if verifiable, None if gold not parseable."""
    gold_parsed = parse(gold_str, extraction_mode="first_match")
    if len(gold_parsed) == 0:
        return None  # skip if gold cannot be parsed

    content = _last_assistant_content(messages)
    answer_parsed = parse(
        content,
        extraction_config=_MATH_EXTRACTION,
        extraction_mode="first_match",
    )
    if len(answer_parsed) == 0:
        return 0  # penalty: no valid math extract
    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception:
        return None

# ------------------ branch: TriviaQA (dict gold) ------------------
def _reward_trivia(messages: Any, gold: Dict[str, Any]) -> Optional[float]:
    """Only considers \\boxed{...}; returns 1/0, None if no usable gold."""
    cands = set()
    for k in ("value","normalized_value","matched_wiki_entity_name","normalized_matched_wiki_entity_name"):
        v = gold.get(k)
        if isinstance(v, str) and v.strip():
            cands.add(_norm(v))
    for k in ("aliases","normalized_aliases"):
        arr = gold.get(k, [])
        if isinstance(arr, list):
            for v in arr:
                if isinstance(v, str) and v.strip():
                    cands.add(_norm(v))
    if not cands:
        return None

    content = _last_assistant_content(messages)
    preds = [_clean_pred_text(b) for b in _extract_all_boxed(content) if b.strip()]
    preds_norm = []
    seen = set()
    for p in preds:
        pn = _norm(p)
        if pn not in seen:
            seen.add(pn)
            preds_norm.append(pn)
    hit = any(pn in cands for pn in preds_norm)
    return 1.0 if hit else 0.0

# ------------------ branch: SciEval (["C"]) ------------------
_CHOICE_SET = set(list("ABCDEFGH"))

def _extract_choice_letter(text: str) -> Optional[str]:
    """Try to extract a single MC letter from text, preferring boxed."""
    content = text

    # 1) Prefer boxed
    boxed = _extract_all_boxed(content)
    for b in boxed:
        t = _clean_pred_text(b).strip().upper()
        # allow forms like 'C', '(C)', 'C.', 'option C'
        if len(t) == 1 and t in _CHOICE_SET:
            return t
        # If boxed holds a phrase, try to pick a trailing isolated letter
        m = re.search(r'\b([A-H])\b', t)
        if m:
            return m.group(1).upper()

    # 2) Common patterns
    pats = [
        r'(?i)\boption\s*([A-H])\b',
        r'(?i)\banswer\s*(?:is|:)?\s*([A-H])\b',
        r'(?i)\bchoice\s*([A-H])\b',
        r'\(([A-H])\)',              # (C)
        r'\b([A-H])[.)]\b',          # C) or C.
        r'\b([A-H])\b',              # lone letter (fallback, least precise)
    ]
    for pat in pats:
        m = re.search(pat, content)
        if m:
            ch = m.group(1).upper()
            if ch in _CHOICE_SET:
                return ch
    return None

def _reward_scieval(messages: Any, gold_list: List[str]) -> Optional[float]:
    """Gold like ['C']; 1/0 if we can extract a choice, None if gold/pred missing."""
    if not isinstance(gold_list, (list, tuple)) or not gold_list:
        return None
    gold_raw = str(gold_list[0] if gold_list[0] is not None else "").strip().upper()
    if len(gold_raw) != 1 or gold_raw not in _CHOICE_SET:
        return None

    pred_text = _last_assistant_content(messages)
    pred = _extract_choice_letter(pred_text)
    if pred is None:
        # Could choose to penalize as -1; to mirror TriviaQA (no boxed -> 0.0), we return 0.0
        return 0.0
    return 1.0 if pred == gold_raw else 0.0



def _maybe_deserialize(gold, debug: bool = False):
    """
    If `gold` is a JSON-looking string (starts with '{' or '[' or is a quoted JSON string),
    return json.loads(gold). Otherwise return it unchanged.
    """
    try:
        if isinstance(gold, str):
            s = gold.strip()
            # only parse clearly JSON-looking payloads
            if not s:
                return gold
            if s[0] in "{[" or (s.startswith('"') and s.endswith('"')):
                out = json.loads(s)
                # _log(f"[Rehydrate] OK type={type(out).__name__} from JSON", debug)
                return out
        return gold
    except Exception as e:
        # _log(f"[Rehydrate][WARN] json.loads failed: {type(e).__name__}: {e}", debug)
        return gold


# ------------------ Dispatcher ------------------
def general_reward(completions: List[List[Dict[str, str]]],
                   solution,
                   debug: bool = False,
                   **kwargs) -> List[Optional[float]]:
    """
    Auto-detect gold format per item and compute reward accordingly.

    Returns a list of floats in {1.0, 0.0, -1.0} or None (skip).
      - DAPO/LIMO math: 1/0 by verify; -1 if pred unparseable; None if gold unparseable.
      - TriviaQA:       1/0 using ONLY \\boxed{...}; None if gold dict lacks candidates.
      - SciEval:        1/0 for choice letter; 0 if no extract; None if bad gold.
    """
    rewards: List[Optional[float]] = []
    for idx, (messages, gold_raw) in enumerate(zip(completions, solution)):
        try:
            # _log(f"\n========== EXAMPLE idx={idx} ==========", debug)
            # _log(f"[Input] gold_raw={_safe_snip(gold_raw)} type={type(gold_raw).__name__}", debug)

            gold = _maybe_deserialize(gold_raw, debug=debug)
            # _log(f"[Input] gold_used={_safe_snip(gold)} type={type(gold).__name__}", debug)

            # Detect by gold type/shape AFTER rehydration
            if isinstance(gold, dict):
                r = _reward_trivia(messages, gold)
                # _log("[Dispatch] Detected Trivia (dict gold)"+str(r), debug)

            elif isinstance(gold, (list, tuple)):
                r = _reward_scieval(messages, list(gold))
                # _log("[Dispatch] Detected SciEval (list/tuple gold)"+str(r), debug)

            elif isinstance(gold, str) and len(gold.strip()) == 1 and gold.strip().upper() in _CHOICE_SET:
                r = _reward_scieval(messages, [gold.strip().upper()])
                # _log("[Dispatch] Detected SciEval-style single-letter string"+str(r), debug)

            else:
                r = _reward_math(messages, str(gold))
                # _log("[Dispatch] Defaulting to Math (string/other gold)"+str(r), debug)

        except Exception as e:
            # _log(f"[ERROR] {type(e).__name__}: {e}", debug)
            r = 0

        # _log(f"[Dispatch] FINAL reward={r}", debug)
        rewards.append(r)

    # _log(f"===== BATCH DONE: rewards sample={_safe_snip(rewards[:32])} =====", debug)
    return rewards



def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "trivia" : trivia_reward,
        "general": general_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "cf_code": update_wrapper(
            partial(
                cf_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                scoring_mode=script_args.code_eval_scoring_mode,
            ),
            cf_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
