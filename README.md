# Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits
<div align="center">

<a href="https://hf.co/collections/AmirhoseinGH/gnosis-training-data">
  <img src="https://img.shields.io/badge/Hugging%20Face-Training%20Data-FFD21E?logo=huggingface&logoColor=000" height="28" />
</a>
&nbsp;&nbsp;
<a href="https://hf.co/collections/AmirhoseinGH/gnosis-self-awareness">
  <img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?logo=huggingface&logoColor=000" height="28" />
</a>
&nbsp;&nbsp;
<a href="http://arxiv.org/abs/2512.20578">
  <img src="https://img.shields.io/badge/arXiv-2512.20578-B31B1B?logo=arxiv&logoColor=white" height="28" />
</a>

</div>

<div align="center"> <img src="assets/Gnosis_demo.gif" alt="Gnosis Demo" width="100%" />

Overview of our Gnosis self-awareness mechanism and its performance.

<img src="assets/main_fig.png" alt="Gnosis Overview Figure" width="100%" /> </div>

**Gnosis** is a lightweight **self-awareness head** attached to a *(frozen)* LLM backbone that predicts a **scalar correctness probability** for a generated response by reading the model’s **hidden states + attention maps**.

---

<div align="center">

# 🚨 NEW RELEASE 🚨

## [Multi-Head Latent Control](https://github.com/Amirhosein-gh98/Multi-Head-Latent-Control)

### From predicting failures to actively controlling agents

**Building on Gnosis, our new work turns latent self-awareness into real-time deployment decisions. Multi-Head Latent Control reads the internal states of frozen LLMs to decide when to answer, continue reasoning, ask for clarification, use tools, abstain, or route to a stronger model—achieving higher success while reducing plotted API cost by up to ~90% in our reported AndroidWorld settings.**

<a href="https://github.com/Amirhosein-gh98/Multi-Head-Latent-Control">
  <img src="https://img.shields.io/badge/Explore-Multi--Head%20Latent%20Control-FF4B4B?style=for-the-badge&logo=github&logoColor=white" height="34" />
</a>

</div>

---


## 📁 Repository layout

- **`transformers/`** — local Transformers fork with **Gnosis integrated into the model architecture** 
  - Implemented under:
    - `transformers/src/transformers/models/gpt_oss`
    - `transformers/src/transformers/models/qwen3`

- **`trl/`** — local TRL fork with a **modified `SFTTrainer`** to train the Gnosis head  
  - Key change:
    - `trl/trl/trainer/sft_trainer.py`

- **`open-r1/`** — training code + configs

- **`src/`** — inference + data tools (quickstart, scoring, preprocessing scripts)

---

## 🧩 Installation

### ✅ Option A: One-command setup

From repo root:

```bash
chmod +x scripts/setup_gnosis_env.sh
bash scripts/setup_gnosis_env.sh
conda activate Gnosis
````

### 🛠️ Option B: Manual install (exact steps)

```bash
conda create -n Gnosis python=3.11 -y
conda activate Gnosis

pip install --upgrade pip wheel setuptools
pip install vllm==0.8.5.post1

python - <<'PY'
import torch; print("Torch:", torch.__version__)
PY

pip install flash-attn --no-build-isolation

pip uninstall -y transformers || true
pip install -e ./transformers
pip install -e "./trl[vllm]"

cd open-r1
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]" --no-deps
cd ..

python - <<'PY'
import pathlib, transformers, trl
print("transformers →", pathlib.Path(transformers.__file__).resolve())
print("trl          →", pathlib.Path(trl.__file__).resolve())
PY

export TOKENIZERS_PARALLELISM=false
```

---

## ⚡ Quickstart: Use Gnosis on a single question
In this example, we first generate a solution for a single question (via **vLLM** or **HF generation**), then run **Gnosis** on *(prompt + answer)* to output a scalar **correctness probability**.

**Task options:** `math`, `trivia`, `mmlu_pro`

* `math / reasoning` → step-by-step; final in `\boxed{}`
* `trivia` → short factoid; final in `\boxed{}`
* `mmlu_pro` → multiple-choice; final is **only the letter** in `\boxed{}`

```python
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

GNOSIS_MODEL_ID = "Trained_gnosis_model"
VLLM_MODEL_ID = "Qwen/Qwen3-1.7B"
USE_VLLM = False

SYSTEM_PROMPTS = {
    "math": "Please reason step by step, and put your final answer within \\boxed{}.",
    "trivia": "This is a trivia question. Put your final answer within \\boxed{}.",
    "mmlu_pro": "You are solving multiple-choice questions. Please reason step by step, and put your final answer with only the choice letter within \\boxed{}."
}

tokenizer = AutoTokenizer.from_pretrained(GNOSIS_MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    GNOSIS_MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).cuda().eval()

prompt = build_chat_prompt(
    tokenizer,
    question="How many r's are in strawberry?",
    system_prompt=SYSTEM_PROMPTS["math"],
)

if USE_VLLM:
    llm = LLM(
        VLLM_MODEL_ID,
        **{
            "tensor_parallel_size": 1,
            "max_model_len": 12000,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.50,
            "trust_remote_code": True,
        },
    )
    sp = make_vllm_sampling_params(temperature=0.6, top_p=0.95, max_tokens=10_000)
    answer = generate_with_vllm(llm, prompt, sp)
else:
    answer = generate_with_hf(
        model, tokenizer, prompt, torch.device("cuda"),
        max_new_tokens=10_000, temperature=0.6, top_p=0.95
    )

p_correct = correctness_prob(
    model, tokenizer, prompt + answer, torch.device("cuda"), max_len_for_scoring=None
)

print("Answer:\n", answer)
print("Gnosis correctness probability:", f"{p_correct:.4f}")
```

---

## 🏋️ Training Gnosis

### 🧪 Step 1 — Data generation

Training begins with a simple pipeline: **generate model completions** (per dataset/benchmark) → **verify** them into **binary correctness labels** → **merge + rebalance** tasks (e.g., *math + trivia*) into one **SFT-ready Parquet** dataset.

➡️ Full, step-by-step instructions is provided in **`DATA_PREPROCESS.md`**.



### 🚀 Step 2 — Train with `open-r1`

Training configs live under:

* `open-r1/recipes/training/` *(per-backbone YAMLs, e.g., Qwen3 / GPT-OSS, etc.)*

Example config:

* `open-r1/recipes/training/Qwen3/Qwen3-1.7B_hybrid_gnosis.yaml`

To train:

```bash
accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
  src/open_r1/sft.py \
  --config recipes/training/Qwen3/Qwen3-1.7B_hybrid_gnosis.yaml
```

**Note:** This setup is currently configured for **2× A100 GPUs**. Adjust the Accelerate/DeepSpeed config (and batch sizes, gradient accumulation, etc.) to match your available hardware.



## 📊 Evalution

We provide a convenience wrapper script to run the scorer on multiple benchmark shard directories (e.g., Math / TriviaQA / MMLU-Pro) and write all outputs under one folder.

**Script:** `src/evaluation/scripts/Gnosis_run_all_scoring.sh`
It calls: `src/evaluation/score_completions_Gnosis_outputscores_script_version.py`

#### Edit these paths

* `MODEL="Path_to_trained_gnosis_backbone"`
* `MATH10_DIR=...`, `TRIVIA_DIR=...`, `MMLUPRO_DIR=...` *(dirs that contain `shard-*.parquet`)*
* `OUT_BASE="outputs/scored_runs/Gnosis"`

#### Run

```bash
bash src/evaluation/scripts/Gnosis_run_all_scoring.sh
```


Outputs are saved in:
`outputs/scored_runs/Gnosis/scored/<model_name>/`

---


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Amirhosein-gh98/Gnosis&type=Date)](https://www.star-history.com/#Amirhosein-gh98/Gnosis&Date)

## Citation
If you find our work useful, please consider citing our paper in your research.

```
@article{ghasemabadi2025can,
  title={Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits},
  author={Ghasemabadi, Amirhosein and Niu, Di},
  journal={arXiv preprint arXiv:2512.20578},
  year={2025}
}
```
