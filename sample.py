from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

inputs = tokenizer(["Pick a number between 1 and 10."], return_tensors="pt").to(model.device)
# There is a print message hardcoded in the custom generation method
gen_out = model.generate(**inputs, n_sigma=1.0, custom_generate="Pramodith/topN_sigma_generation", max_length=10, trust_remote_code=True)
print(tokenizer.batch_decode(gen_out))
