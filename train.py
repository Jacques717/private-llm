from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from datasets import Dataset


# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Enable 8-bit precision
    llm_int8_enable_fp32_cpu_offload=True  # Allow offloading 32-bit parts to the CPU
)

# Load the model and tokenizer
#model_name = "meta-llama/Llama-2-7b-chat-int8" 
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token if not already defined
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",quantization_config=quantization_config)  # Allow offloading 32-bit parts to the CPU

# Load your custom dataset
data = [{"prompt": "What is my house address?", "response": "123 Main Street, Springfield."}]
dataset = Dataset.from_list(data)


# Tokenize the dataset with a maximum length
def tokenize_function(example):
    # Tokenize both the prompt and the response
    return tokenizer(
        example["prompt"],  # Only tokenize the prompt for input_ids
        text_target=example["response"],  # Use response as the target labels
        truncation=True,
        max_length=512,
        padding="max_length",
    )
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Prepare training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10,
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")



# # Tokenize the dataset
# def tokenize_function(example):
#     return tokenizer(f"{example['prompt']} {example['response']}", truncation=True)
