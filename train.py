import torch
import os
import unsloth
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel, apply_chat_template
from transformers import TrainingArguments
from utils import dataset, chat_template

os.environ['WANDB_PROJECT'] = "text_entailment"
os.environ['WANDB_LOG_MODEL'] = 'checkpoint'

MAX_LENGTH=512

# print(dataset)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=MAX_LENGTH,
    dtype=None,
    load_in_4bit=False,
    fast_inference=False
)

dataset = apply_chat_template(
    dataset,
    tokenizer=tokenizer,
    chat_template=chat_template
)

print(dataset)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Dropout = 0 is currently optimized
    bias="none",  # Bias = "none" is currently optimized
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

training_args = SFTConfig(
    output_dir="/scratch/data/asif_rs/output_llm/",
    max_seq_length=MAX_LENGTH,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=1,
    # max_steps=60,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    bf16=True,
    lr_scheduler_type="linear",
    report_to="wandb",
    dataset_num_proc=256,
    packing=False    
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()

model.save_pretrained('lora_te')
tokenizer.save_pretrained('lora_te')