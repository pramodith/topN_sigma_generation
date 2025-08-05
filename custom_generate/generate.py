import torch

def top_n_sigma_sampling(logits, temperature, n_sigma=4):
    """
    Perform topN-sigma sampling on the logits.
    
    Args:
        logits (torch.Tensor): The logits from the model of shape (batch_size, vocab_size).
        temperature (float): The temperature to apply to the logits.
        n_sigma (int): The number of standard deviations to use for filtering.
    
    Returns:
        torch.Tensor: The filtered logits after applying topN-sigma sampling.
    """
    logits = logits / temperature  # Apply temperature scaling
    max_logit_score = torch.max(logits, dim=-1, keepdim=True).values
    std = torch.std(logits, dim=-1, keepdim=True)
    threshold = max_logit_score - n_sigma * std
    filtered_logits = torch.where(logits >= threshold, logits, torch.tensor(float('-inf')))
    return filtered_logits

@torch.inference_mode()
def generate(model, input_ids, generation_config=None, n_sigma=4, **kwargs):
    """
    Generate text using topN-sigma sampling based on the paper: 
    https://openreview.net/pdf/1e221c8eedaf42558abc5dca4637b3378297582b.pdf
    
    
    @inproceedings{tang2025top,
        title={Top-n𝜎: Eliminating Noise in Logit Space for Robust Token Sampling of LLM},
        author={Tang, Chenxia and Liu, Jianchun and Xu, Hongli and Huang, Liusheng},
        booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
        pages={10758--10774},
        year={2025}
    }

    Args:
        model: The model to use for generation.
        input_ids (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
        generation_config (optional): Configuration for generation, such as max_length, pad_token_id,
                                      and max_new_tokens.
        n_sigma (int): The number of standard deviations to use for topN-sigma sampling.
        **kwargs: Additional keyword arguments.
    """    
    generation_config = generation_config or model.generation_config  # default to the model generation config
    cur_length = input_ids.shape[1]
    max_length = generation_config.max_length or cur_length + generation_config.max_new_tokens

    while cur_length < max_length:
        logits = model(input_ids).logits
        logits = logits[:, -1, :]
        # Filter logits using topN-sigma sampling
        filtered_logits = top_n_sigma_sampling(logits, generation_config.temperature, n_sigma=n_sigma)
        # sample from the filtered logits
        next_tokens = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        cur_length += 1

    return input_ids