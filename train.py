from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load your custom dataset
data = [{"prompt": "What is my house address?", "response": "123 Main Street, Springfield."}]
dataset = Dataset.from_list(data)

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(f"{example['prompt']} {example['response']}", truncation=True)

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
