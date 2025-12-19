#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =============================================================================
# Task prompts (explicit, no helper import needed in README)
# =============================================================================
SYSTEM_PROMPTS = {
    "math": "Please reason step by step, and put your final answer within \\boxed{}.",
    "trivia": "This is a trivia question. Put your final answer within \\boxed{}.",
    "mmlu_pro": (
        "You are solving multiple-choice questions. "
        "Please reason step by step, and put your final answer with only the choice letter "
        "within \\boxed{}."
    ),
}

# =============================================================================
# Helpers (library-style, no global deps)
# =============================================================================
def has_correctness_head(model) -> bool:
    return (
        hasattr(model, "_should_stop")
        and hasattr(model, "stop_head")
        and hasattr(model, "hid_extractor")
        and hasattr(model, "conf_extractor")
    )


def build_chat_prompt(tokenizer, question: str, system_prompt: str | None = None) -> str:
    """Use the model's chat template (Qwen-style)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -------------------------
# vLLM (init once, reuse)
# -------------------------
def make_vllm_llm(model_id: str, *, engine_kwargs: dict):
    """
    Construct and return a vLLM LLM engine.
    Call once and reuse for multiple prompts.
    """
    try:
        from vllm import LLM
    except ImportError as e:
        raise ImportError("vLLM is not installed. `pip install vllm`") from e

    return LLM(model=model_id, **engine_kwargs)


def make_vllm_sampling_params(*, temperature: float, top_p: float, max_tokens: int):
    """Create vLLM SamplingParams (kept separate so README can override easily)."""
    try:
        from vllm import SamplingParams
    except ImportError as e:
        raise ImportError("vLLM is not installed. `pip install vllm`") from e

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def generate_with_vllm(llm, prompt: str, sampling_params) -> str:
    """Single-turn generation with an existing vLLM engine."""
    outputs = llm.generate([prompt], sampling_params=sampling_params)
    text = outputs[0].outputs[0].text
    return text.strip()


# -------------------------
# HF generation (explicit)
# -------------------------
@torch.no_grad()
def generate_with_hf(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = enc["input_ids"].shape[1]
    gen_ids = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    new_tokens = gen_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# -------------------------
# Gnosis scoring (explicit)
# -------------------------
@torch.no_grad()
def correctness_prob(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    *,
    max_len_for_scoring: int | None = None,   # None => score full sequence
) -> float:
    if not has_correctness_head(model):
        raise RuntimeError("Loaded model does not have Gnosis correctness head (`_should_stop`).")

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=(max_len_for_scoring is not None),
        max_length=max_len_for_scoring,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
    )
    hidden_states = out.last_hidden_state
    logits = model.lm_head(hidden_states)

    probs = torch.softmax(logits, dim=-1)
    token_probs = probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    attn_stack = out.attentions

    scores = model._should_stop(
        last_hidden=hidden_states,
        attn_stack=attn_stack,
        token_probs=token_probs,
        mask=attention_mask.float(),
        input_ids=input_ids,
    )

    return float(scores.squeeze(-1).float().clamp(1e-6, 1 - 1e-6).item())


# =============================================================================
# Demo (kept simple; you can argparse this later)
# =============================================================================
def main():
    # --- EDIT THESE ---
    GNOSIS_MODEL_ID = "/home/amirhosein/codes/SelfAwareMachine/open-r1/output_final/Qwen3_1.7B_Gnosis/checkpoint-4064"
    GEN_MODEL_ID = GNOSIS_MODEL_ID

    TASK = "math"     # "math" | "trivia" | "mmlu_pro"
    USE_VLLM = True

    GEN_MAX_NEW_TOKENS = 256
    MAX_LEN_FOR_SCORING = 4096

    # Fixed: gpu_memory to 0.50 (shares GPU with HF model), added trust_remote_code
    VLLM_ENGINE_KW = {
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.50,
        "dtype": "bfloat16",
        "max_model_len": 16000,
        "trust_remote_code": True,
    }

    temperature = 0.6
    top_p = 0.95

    # ----------------
    task = TASK.lower()
    system_prompt = SYSTEM_PROMPTS[task]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Loading Gnosis model from: {GNOSIS_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(GNOSIS_MODEL_ID, trust_remote_code=True)
    gnosis_model = AutoModelForCausalLM.from_pretrained(
        GNOSIS_MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=False,
    ).to(device).eval()

    if not has_correctness_head(gnosis_model):
        raise RuntimeError("Gnosis model is missing `_should_stop` / encoders. Use your fine-tuned checkpoint.")

    # Init vLLM ONCE (if used)
    llm = None
    sampling_params = None
    if USE_VLLM:
        llm = make_vllm_llm(GEN_MODEL_ID, engine_kwargs=VLLM_ENGINE_KW)
        sampling_params = make_vllm_sampling_params(
            temperature=temperature,
            top_p=top_p,
            max_tokens=GEN_MAX_NEW_TOKENS,
        )

    question = input("\nEnter a question (include options if MCQ):\n> ").strip()
    if not question:
        print("No question given, exiting.")
        return

    prompt = build_chat_prompt(tokenizer, question, system_prompt)

    # 1) Generate answer
    if USE_VLLM:
        answer = generate_with_vllm(llm, prompt, sampling_params)
    else:
        answer = generate_with_hf(
            gnosis_model,
            tokenizer,
            prompt,
            device,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=temperature,
            top_p=top_p,
        )

    # 2) Score correctness on [prompt + answer]
    full_text = prompt + answer
    p_correct = correctness_prob(
        gnosis_model,
        tokenizer,
        full_text,
        device,
        max_len_for_scoring=MAX_LEN_FOR_SCORING,
    )

    print("\n================ DEMO ================")
    print(f"Question:\n{question}\n")
    print(f"Answer:\n{answer}\n")
    print(f"Gnosis correctness probability: {p_correct:.4f}")
    print("======================================")


if __name__ == "__main__":
    main()