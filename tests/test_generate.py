import torch
from custom_generate.generate import top_n_sigma_sampling, generate

class MockGenerationConfig:
    def __init__(self, max_length=None, max_new_tokens=3, temperature=1.0):
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

class MockModel:
    def __init__(self, logits_sequence, generation_config=None):
        self.logits_sequence = logits_sequence
        self.call_count = 0
        self.generation_config = generation_config or MockGenerationConfig()

    def __call__(self, input_ids):
        # Always return the next logits in the sequence, or repeat last
        if self.call_count < len(self.logits_sequence):
            logits = self.logits_sequence[self.call_count]
        else:
            logits = self.logits_sequence[-1]
        self.call_count += 1
        class Output:
            pass
        out = Output()
        out.logits = logits
        return out

def test_top_n_sigma_sampling_basic():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    temperature = 1.0
    n_sigma = 1
    filtered = top_n_sigma_sampling(logits, temperature, n_sigma)
    # Only the highest logits should remain, others set to -inf
    assert filtered.shape == logits.shape
    assert torch.isinf(filtered[0][0]) or filtered[0][0] < 0  # Should be -inf or negative
    assert filtered[0][3] == logits[0][3]  # Highest logit remains

def test_top_n_sigma_sampling_all_equal():
    logits = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    temperature = 1.0
    n_sigma = 1
    filtered = top_n_sigma_sampling(logits, temperature, n_sigma)
    # All logits are equal, so all should remain
    assert torch.all(filtered == logits)

def test_generate_basic():
    # Simulate logits for 2 steps, vocab_size=4
    logits_seq = [
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),  # step 1
        torch.tensor([[[2.0, 1.0, 0.0, 3.0]]]),  # step 2
        torch.tensor([[[0.0, 0.0, 0.0, 10.0]]]), # step 3
    ]
    gen_config = MockGenerationConfig(max_length=None, max_new_tokens=3, temperature=1.0)
    model = MockModel(logits_seq, generation_config=gen_config)
    input_ids = torch.tensor([[0, 1]])
    output = generate(model, input_ids, generation_config=gen_config, n_sigma=1)
    # Should generate 3 new tokens, so output shape[1] == input_ids.shape[1] + 3
    assert output.shape[1] == input_ids.shape[1] + 3

def test_generate_with_max_length():
    logits_seq = [
        torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        torch.tensor([[[2.0, 1.0, 0.0, 3.0]]]),
    ]
    gen_config = MockGenerationConfig(max_length=4, max_new_tokens=10, temperature=1.0)
    model = MockModel(logits_seq, generation_config=gen_config)
    input_ids = torch.tensor([[0, 1]])
    output = generate(model, input_ids, generation_config=gen_config, n_sigma=1)
    # Should stop at max_length=4
    assert output.shape[1] == 4