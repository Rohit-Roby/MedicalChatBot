from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments,AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
import pandas as pd
# from tokenizer import tokenize_function
# Load tokenizer
model_name = "aboonaji/llama2finetune-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    # Combine user prompt and model response for each item in the batch
    print(type(examples))
    print(type(examples['user_prompt']))
    print(type(examples['model_resp']))
    combined_texts = [
        str(user_prompt) + " " + str(model_resp)
        for user_prompt, model_resp in zip(examples['user_prompt'], examples['model_resp'])
    ]
    tokenized = tokenizer(
        combined_texts,  # Pass the combined texts
        padding="max_length",
        truncation=True,
        max_length=512
    )
    # Add labels for causal language modeling
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

from datasets import Dataset
# Convert DataFrame to Hugging Face Dataset
df = pd.read_csv('processed_data.csv')
dataset = Dataset.from_pandas(df)
# tokenized['labels'] = tokenized['input_ids'].copy()
print(type(dataset))
print(dataset[0])
# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)
8
# Train-Test Split (90% train, 10% test)
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Load model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=bnb_config,
    device_map="cpu"
    # load_in_8bit=True,"
)

# Set up LoRA configuration
lora_config = LoraConfig(
    r=12,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    # evaluation_strategy="steps",
    # evaluation_strategy="steps",  # Enable evaluation at intervals
    # save_strategy="steps", 
    num_train_epochs=5,
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    logging_dir='./logs',
    logging_steps=500,
    save_steps=500,
    fp16=True,
    weight_decay=0.01,
    # load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
# Save the model and tokenizer