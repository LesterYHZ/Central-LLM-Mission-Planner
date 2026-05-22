# -*- coding: utf-8 -*-
"""
HoloOcean 多机规划 - 增量微调 (Incremental Fine-Tuning) 脚本
功能：加载上一轮成功微调的 LoRA 权重，使用更低的学习率融入新场景数据。
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

print(">>> [增量学习模式] 正在载入 上一轮已微调的 LoRA 大脑...")
# 注意：这里直接填入上一轮的保存目录！Unsloth 会自动识别这是一个已有的 PEFT 模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="central_llm_lora_model_v2",  # 上一轮微调的模型目录
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# 🚨 极其重要：增量训练绝对不能调用 get_peft_model 再次包裹模型！
# 模型已经在上一步加载时自带了 PEFT 权重，我们直接进入训练准备状态即可。

def format_prompts(batch):
    formatted_texts = []
    for messages in batch["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        formatted_texts.append(text)
    return {"text": formatted_texts}


print(">>> 载入广度扩容版的新场景数据集...")
dataset = load_dataset("json", data_files="robot_planning_train_more_cleaned.jsonl", split="train")
dataset = dataset.map(format_prompts, batched=True)

# 增量学习配置
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    warmup_steps=50,

    num_train_epochs=2,  # 增量学习只需要 2 个 Epoch 就能吸收新场景

    learning_rate=5e-5,  # 【核心】：使用比上次更低的学习率，防止“灾难性遗忘”原有逻辑

    bf16=True,
    fp16=False,
    output_dir="outputs_incremental",
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

print("\n=================== 增量逻辑拟合主循环启动 (Incremental) ===================")
trainer_stats = trainer.train()

# 另存为一个全新的版本，不覆盖上一轮的模型，方便回滚测试
output_lora_dir = "central_llm_lora_model_v3"
model.save_pretrained(output_lora_dir)
tokenizer.save_pretrained(output_lora_dir)
print(f"🎉 增量微调完美收敛！V3 进化版模型已保存在本地目录：{output_lora_dir}")