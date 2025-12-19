---
pretty_name: "Gnosis SFT — GPT-OSS-20B (TriviaQA + DAPO-Math, 2 completions/question, low-thinking)"
tags:
- gnosis
- sft
- correctness-detection
- hallucination-detection
- triviaqa
- math
- gpt-oss-20b
source_datasets:
  - open-r1/DAPO-Math-17k-Processed
  - mandarjoshi/trivia_qa
---

# Gnosis SFT — GPT-OSS-20B (TriviaQA + DAPO-Math, 2 completions/question, low-thinking)

Backbone:
- `openai/gpt-oss-20b`

Sources:
- open-r1/DAPO-Math-17k-Processed
- mandarjoshi/trivia_qa

Local build folder:
- `/home/amirhosein/codes/SelfAwareMachine/outputs/tarining_data/Final/gpt_oss20B/general_trivia_math_low_thinking_unbalanced_downsample_final`

## Load

```python
from datasets import load_dataset
ds = load_dataset("AmirhoseinGH/gnosis-gpt-oss-20b-triviaqa-dapo-sft-2comp-lowthinking")
print(ds)
print(ds["train"][0])
````

