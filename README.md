---
library_name: transformers
tags:
  - custom_generate
---
## Overview
This generation sampling method is based on the paper [Top-N Sigma: A Simple and Effective Sampling Method for Language Models](https://openreview.net/pdf/1e221c8eedaf42558abc5dca4637b3378297582b.pdf).


Most output token sampling techniques operate on the probability scores post temperature being applied. The softmax function distorts the underlying logit scores distribution making it hard to know a meaningful top-p/top-k value to set.

This can lead to invalid tokens being in the chosen set of tokens after applying the top/min p/k threshold.

The authors observed that the logit scores for the most part follow a gaussian distribution and noisy/irrelevant tokens would often be in the outlier zone.

> We observe that the majority of logits follow a Gaussian distribution in the lower-value region, which corresponds to the low-probability tails that are commonly treated as noise in the probability distribution. This pattern suggests the potential for more meaningful truncation in the logit space.

## 😎 Top-NSigma

Top-NSigma is a simple sampling algorithm that operates on the logit scores directly, here’s how it works:

1. Find the max logit score for the given time step of generation.
2. Compute the standard deviation of all the logit scores.
3. Filter out tokens with logit scores less than N standard deviations away from the max logit scores.
4. Apply temperature and softmax to convert logit scores of the unfiltered tokens to probs.
5. Sample tokens.

## Base model:
`Qwen/Qwen2.5-0.5B-Instruct`

## Model compatibility
Most models. More specifically, any `transformer` LLM/VLM trained for causal language modeling.

## Additional Arguments

This implementation of Top-NSigma requires the user to pass in a new argument `n_sigma` to the generation function.

We'll use this to filter out tokens whose logit scores are `n_sigma` number of standard deviations below the max logit score.

The authors recommend using `n_sigma=1.0` for most use cases, but you can experiment with values in the range **(0.0, 2√3]**.

## Output Type changes
(none)

## Example usage

```py
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto")
generation_config = GenerationConfig(temperature=1.5, max_length=128)

messages = [{"role":"user", "content": "Write a story about a dog and cat becoming friends."}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# There is a print message hardcoded in the custom generation method
gen_out = model.generate(**model_inputs, n_sigma=1.0, generation_config=generation_config, custom_generate="Pramodith/topN_sigma_generation", trust_remote_code=True)

print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
```

### Citation
```bibtex
@inproceedings{tang2025top,
    title={Top-n𝜎: Eliminating Noise in Logit Space for Robust Token Sampling of LLM},
    author={Tang, Chenxia and Liu, Jianchun and Xu, Hongli and Huang, Liusheng},
    booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    pages={10758--10774},
    year={2025}
}
```
