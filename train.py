from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model







# Configure 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load the model and tokenizer
#model_name = "meta-llama/Llama-2-7b-chat-hf"
#model_name = "meta-llama/Llama-2-3b-chat-hf"
model_name = "meta-llama/Llama-2-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token if not already defined
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     quantization_config=quantization_config
# )


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="float16"  # Use FP16 precision for better training compatibility
)




# Configure LoRA for PEFT
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Applies LoRA to specific modules
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Print trainable parameters for confirmation
model.print_trainable_parameters()

# Load your custom dataset
data = [{"prompt": "What is my house address?", "response": "123 Main Street, Springfield."}]
dataset = Dataset.from_list(data)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(
        example["prompt"],
        text_target=example["response"],
        truncation=True,
        max_length=512,
        padding="max_length"
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

# Train the model with PEFT
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")
