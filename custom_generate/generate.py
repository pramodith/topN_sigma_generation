from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.save_pretrained("Pramodith/topN_sigma_generation", push_to_hub=True)
model.save_pretrained("Pramodith/topN_sigma_generation", push_to_hub=True)
