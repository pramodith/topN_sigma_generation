---
library_name: transformers
tags:
  - custom_generate
---
## Overview

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

## Output Type changes
(none)

## Example usage

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")

inputs = tokenizer(["The quick brown"], return_tensors="pt").to(model.device)
# There is a print message hardcoded in the custom generation method
gen_out = model.generate(**inputs, left_padding=5, custom_generate="Pramodith/topN_sigma_generation", trust_remote_code=True)
print(tokenizer.batch_decode(gen_out)) 
```
