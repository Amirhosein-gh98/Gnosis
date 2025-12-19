import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM
from src.demo import (
    build_chat_prompt,
    make_vllm_sampling_params,
    generate_with_vllm,
    generate_with_hf,
    correctness_prob,
)

#Path_to_models
GNOSIS_MODEL_ID = "/home/amirhosein/codes/SelfAwareMachine/open-r1/output_final/Qwen3_1.7B_Gnosis/checkpoint-4064"
VLLM_MODEL_ID = "Qwen/Qwen3-1.7B"

USE_VLLM = False

SYSTEM_PROMPTS = {
    "math": "Please reason step by step, and put your final answer within \\boxed{}.",
    "trivia": "This is a trivia question. Put your final answer within \\boxed{}.",
    "mmlu_pro": (
        "You are solving multiple-choice questions. "
        "Please reason step by step, and put your final answer with only the choice letter "
        "within \\boxed{}."
    ),
}


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(GNOSIS_MODEL_ID, trust_remote_code=True)
    
    
    model = AutoModelForCausalLM.from_pretrained(
        GNOSIS_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    ).cuda().eval()

    # prompt = build_chat_prompt(
    #     tokenizer,
    #     question=(
    #         "Let $p$ be the least prime number for which there exists a positive integer $n$ such that "
    #         "$n^{4}+1$ is divisible by $p^{2}$. Find the least positive integer $m$ such that "
    #         "$m^{4}+1$ is divisible by $p^{2}$."
    #     ),
    #     system_prompt=SYSTEM_PROMPTS["math"],
    # )

    prompt = build_chat_prompt(
        tokenizer,
        question="How many r's are in strrawrberry?",
        system_prompt=SYSTEM_PROMPTS["math"],
    )

    if USE_VLLM:
        llm = LLM(
            VLLM_MODEL_ID,
            **{"tensor_parallel_size": 1,
                "max_model_len": 12000, 
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.50,  
                "trust_remote_code": True},)

        sp  = make_vllm_sampling_params(temperature=0.6, top_p=0.95, max_tokens=10_000)
        answer = generate_with_vllm(llm, prompt, sp)
    else:
        answer = generate_with_hf(model, tokenizer, prompt, torch.device("cuda"),
                                  max_new_tokens=10_000, temperature=0.6, top_p=0.95)

    #Calling Gnosis
    p_correct = correctness_prob(model, tokenizer, prompt + answer, torch.device("cuda"), max_len_for_scoring=None)

    print("Answer:\n", answer)
    print("Gnosis correctness probability:", f"{p_correct:.4f}")