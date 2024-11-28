# README.md

## Setup Instructions to Set Up and Access LLaMA-2 Model

### Step 1: Install Dependencies
Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Set Up Project Directory

```bash 
mkdir private-llm && cd private-llm
```

### Step 3: Create a Virtual Environment

```bash 
python -m venv venv
source venv/bin/activate
```

### Step 4: Install Required Python Packages

```bash
pip install transformers accelerate bitsandbytes
```


## Download and Access the LLaMA-2 Model

### Step 1: Request Access to LLaMA-2 on Hugging Face

1. Visit the LLaMA-2 Model Card on Hugging Face @ https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
2. Log in or create a Hugging Face account.
3. Click "Access repository" to request access.
4. Agree to Meta's licensing terms.

## Blockquotes
>The LLAMA 2 Community License allows users to use, modify, and distribute Llama 2 and its derivatives under certain conditions, such as including attribution and adhering to Meta's Acceptable Use Policy, which prohibits illegal, harmful, or deceptive use cases. Commercial use is restricted for entities with over 700 million monthly active users unless a separate license is granted by Meta. The agreement disclaims warranties, limits Meta’s liability, and includes terms regarding intellectual property rights and compliance with laws, with disputes governed by California law.

Approval is usually instant or takes a few minutes.



### Step 2: Authenticate with Hugging Face
1. Generate an access token on your Hugging Face Access Tokens page.
2. Log in via CLI:
```bash 
huggingface-cli login
``` 
Paste the token when prompted.

# Running the Model

### Step 1: Download and Load the Model
Use the following script to download and load the model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# Generate a response
prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
