# Private Offline Local LLM

## Setup Instructions to Set Up and Access LLaMA-2 Model and run your own personal AI locally

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

## Meta License summary
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

### Step 2: Verify Successful Execution
Run the code:
```bash
python main.py
```

* Output: The script should generate a response like:
```c
The capital of France is Paris.
```


# Offline Usage

### Step 1: Cache Model Locally
Hugging Face caches model files by default. Common cache locations:
* Linux/macOS: ```bash ~/.cache/huggingface/hub/```
* Windows: ```bashC:\Users\<YourUsername>\.cache\huggingface\hub\```

### Step 2: Modify Script for Offline Use
Specify the local model path and disable online access:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/models/llama-2"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

prompt = "What is the capital of France?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Step 3: Pre-Download Dependencies
1. Save all dependencies to a requirements.txt file:
```bash
pip freeze > requirements.txt
```
2. Recreate the environment offline:
```bash
pip install -r requirements.txt --no-index --find-links /path/to/local/package-repo
```

### Step 4: Test Offline Functionality
Disconnect from the internet and ensure the model runs successfully.


### Step 5: Optional: Run on Air-Gapped Hardware
For maximum privacy, consider deploying the model on hardware that is completely disconnected from any external network:
* On-premise servers or dedicated GPUs.
* Use isolated virtual machines with no internet connectivity.

### Step 6: Verify No External Requests
1. Before running your offline setup:
2. Disconnect from the internet or monitor traffic with tools like Wireshark or tcpdump to ensure no external requests are made.
3. Run the script and verify it loads all resources locally without errors.


# Advanced: Fine-Tuning and Customization
1. Prepare proprietary data.
2. Fine-tune the model locally with Hugging Face Transformers or PEFT (Low-Rank Adaptation).
3. Deploy the fine-tuned model on private infrastructure.

# Other Models
* LLaMA 2 (Meta): Chat-optimized models, great for both small and large setups.
* Falcon (Technology Innovation Institute): Lightweight and fast inference.
* Mistral 7B: Recent model optimized for efficiency and accuracy.
* GPT-J/GPT-NeoX: General-purpose models from EleutherAI.



# Coming Up

Stay tuned for the next commits, where we'll explore:

- **Adding Your Own Data**: Learn how to fine-tune the model or integrate custom datasets.
- **Custom Model Fine-Tuning**: Train the deployed model on proprietary datasets for domain-specific use cases.
- **Scaling and Performance Optimization**: Techniques to handle high traffic efficiently with autoscaling and caching.
- **Integrating APIs and Live Data**: Connect your model to real-time data sources for dynamic responses.
- **Authentication and Security**: Implement secure access to your API using OAuth or API keys.
- **Monitoring and Observability**: Track performance and identify issues with Google Cloud Monitoring



## Tech Stack

- **Machine Learning Framework**: PyTorch, Hugging Face Transformers
- **Model**: LLaMA-2 (Meta's Language Model)
- **Libraries**:
  - Python: `transformers`, `torch`, `accelerate`, `bitsandbytes`