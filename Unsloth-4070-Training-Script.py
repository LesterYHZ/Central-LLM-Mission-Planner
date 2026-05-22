# -*- coding: utf-8 -*-
"""
HoloOcean 多机群组扩容版 - Unsloth 极速微调脚本（RTX 4070 Windows 11 安全加固版）
微调目标模型：Qwen/Qwen2.5-3B-Instruct
"""

import os
import sys

# Windows 路径防御补丁
if sys.platform.startswith("win"):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "C:\\triton_cache\\inductor"
    os.environ["TRITON_CACHE_DIR"] = "C:\\triton_cache\\triton"
    os.makedirs("C:\\triton_cache\\inductor", exist_ok=True)
    os.makedirs("C:\\triton_cache\\triton", exist_ok=True)

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 2048
dtype = torch.bfloat16
load_in_4bit = True

print(">>> [RTX 4070 显存优化模式] 正在载入 4-bit 量化大模型与 Tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # 提高 Rank，增强对逻辑的记忆
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


def format_prompts(batch):
    formatted_texts = []
    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)
    return {"text": formatted_texts}


print(">>> 载入经由符号校验清洗后的‘黄金’数据集...")
dataset = load_dataset("json", data_files="robot_planning_train_cleaned.jsonl", split="train")
dataset = dataset.map(format_prompts, batched=True)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    warmup_steps=100,

    # 【核心修复】：不要使用 max_steps，改用 num_train_epochs，保证数据被彻底拟合！
    num_train_epochs=3,

    learning_rate=2e-4,
    bf16=True,
    fp16=False,
    output_dir="outputs_central_llm_5090",
    optim="adamw_8bit",
    logging_steps=20,
    disable_tqdm=True,
    save_strategy="no",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=1,
    packing=False,
    args=training_args,
)

print("\n=================== RTX 4070 深度逻辑拟合主循环启动 ===================")
print("💡 提示：将跑满 3 个 Epoch，请耐心等待...")
trainer_stats = trainer.train()

output_lora_dir = "central_llm_lora_model"
model.save_pretrained(output_lora_dir)
tokenizer.save_pretrained(output_lora_dir)
print(f"🎉 训练完美收敛！多机扩容权重已保存在本地目录：{output_lora_dir}")