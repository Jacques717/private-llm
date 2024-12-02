from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer from the local directory
model_dir = "./Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype="auto")

# Generate a response
prompt = "What is my house address?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
