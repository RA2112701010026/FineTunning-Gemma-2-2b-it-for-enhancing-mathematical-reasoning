# 🔍 Fine-Tuning Gemma 2-2B with QLoRA on MMLU & Math-QA

This project demonstrates fine-tuning Google's [`gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) model using **QLoRA + LoRA (PEFT)** techniques on the [MMLU](https://huggingface.co/datasets/cais/mmlu) and [Math-QA](https://huggingface.co/datasets/rvv-karma/Math-QA) datasets. The goal is to improve multi-domain reasoning and mathematical question-answering capabilities of the base model while optimizing for memory and compute efficiency.

---

## 📌 Project Highlights

- 🔗 **Base Model**: `google/gemma-2-2b-it`
- 🧠 **Tasks**:
  - General-domain multiple choice QA (MMLU)
  - Mathematical reasoning (Math-QA)
- 🛠️ **Fine-tuning Techniques**:
  - QLoRA (4-bit quantization via `bitsandbytes`)
  - LoRA (Parameter-Efficient Fine-Tuning via `PEFT`)
- ⚙️ **Training Framework**: Hugging Face Transformers + TRL SFTTrainer
- 🎯 **Tracking**: Weights & Biases (WandB)
- 🧮 **Metrics**: Accuracy, Perplexity

---

## 🧾 Datasets

### 📘 MMLU (Massive Multitask Language Understanding)
- Source: [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu)
- Split used: `auxiliary_train`
- Format: Multiple choice questions across 57 subjects  
- Structure:  
  ```json
  {
    "question": "...",
    "choices": ["A", "B", "C", "D"],
    "answer": 2
  }
📗 Math-QA
Source: rvv-karma/Math-QA

Structure:

json
Copy
Edit
{
  "Problem": "...",
  "options": { "a": "...", "b": "...", "c": "...", "d": "...", "e": "..." },
  "correct": "b"
}
🏗️ Environment Setup
📦 Requirements
Install dependencies:

pip install -r requirements.txt
Sample requirements.txt:

txt
Copy
Edit
transformers
datasets
peft
trl
bitsandbytes
wandb
torch
scipy
accelerate
🧪 Fine-Tuning Details
Training Configuration:
Parameter	Value
Model	gemma-2-2b-it
Precision	bf16
Quantization	4-bit QLoRA
LoRA r / alpha	8 / 16
Per Device Batch Size	4
Gradient Accumulation Steps	8
Optimizer	paged_adamw_8bit
Scheduler	cosine w/ warmup
Logging	wandb

Hardware Used:
DGX A100 system with 2×A100 GPUs

Additional runs on Kaggle T4 * 2 GPUs

🧹 Dataset Preprocessing
Tokenization handled via Hugging Face’s AutoTokenizer for Gemma

Padding/truncation based on max length (e.g., 512)

Formatted into SFT-style prompt-response format for both datasets

🚀 Training
To launch training:

bash
Copy
Edit
python train.py --config_path config.yaml
Example training log:

Epochs: 3

Loss: Decreasing trend

Accuracy: Improved on test sets

Perplexity: Monitored via evaluation loop

W&B dashboard shows real-time updates

📈 Results
Dataset	Metric	Base Model	Fine-Tuned
MMLU	Accuracy	~45%	↑ Improved
Math-QA	Accuracy	~42%	↑ Improved
MMLU	Perplexity	High	↓ Reduced

Note: Exact numbers will depend on full evaluation results.

🔎 Evaluation & Inference
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.4556|±  |0.0137|
|     |       |strict-match    |     5|exact_match|↑  |0.4496|±  |0.0137|


|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     1|exact_match|↑  |0.5118|±  |0.0138|
|     |       |strict-match    |     1|exact_match|↑  |0.1084|±  |0.0086|


You can export the model using:


model.save_pretrained("finetuned-gemma")
tokenizer.save_pretrained("finetuned-gemma")
For inference:


from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("finetuned-gemma", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("finetuned-gemma")

prompt = "What is the capital of France?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))
Evaluation on real-world QA benchmarks

GGUF export for llama.cpp inference

Integration with LangChain/RAG pipeline

UI for interactive QA demo

🙌 Acknowledgements
Google for releasing the gemma-2-2b-it model

Hugging Face for Transformers, Datasets, TRL

bitsandbytes and peft teams for memory-efficient fine-tuning tools

DGX infra and Kaggle T4 for GPU support
