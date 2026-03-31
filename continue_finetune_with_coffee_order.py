#!/usr/bin/env python3
"""
繼續訓練 FunctionGemma - Coffee Order Dataset

使用 `renhehuang/coffee-order-zhtw` 數據集對已經 finetune 過的模型進行第二輪訓練

使用方式:
    python continue_finetune_with_coffee_order.py [--model_path PATH] [--output_dir PATH] [--epochs N]
"""

import argparse
import json
import os
import subprocess
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="繼續訓練 FunctionGemma 模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./coffee-robot-functiongemma",
        help="已訓練模型的路徑",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./coffee-robot-continued",
        help="訓練輸出目錄",
    )
    parser.add_argument(
        "--final_model_path",
        type=str,
        default="./coffee-robot-continued-final",
        help="最終模型保存路徑",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="訓練 epochs 數",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="每個設備的 batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="學習率（繼續訓練建議用較低的學習率）",
    )
    parser.add_argument(
        "--convert_gguf",
        action="store_true",
        help="訓練後轉換為 GGUF 格式",
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="跳過模型測試",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="merged_coffee_dataset_v2.jsonl",
        help="合併後的數據集路徑",
    )
    return parser.parse_args()


def load_and_prepare_dataset(tokenizer, dataset_path: str):
    """載入並準備合併後的數據集"""
    print(f"📥 載入數據集: {dataset_path}")
    
    # 載入 JSONL 數據集
    data_list = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    
    print(f"📊 數據集大小: {len(data_list)} 筆")
    
    def apply_format(sample):
        """將資料集格式轉換為 prompt-completion 格式"""
        messages = sample.get('messages', [])
        tools = sample.get('tools', [])
        
        if not messages:
            return {"prompt": "", "completion": "", "split": "train"}
        
        # 完整的對話（包含 assistant 回應）
        prompt_and_completion = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 只有 prompt 部分（不含 assistant 回應）
        prompt = tokenizer.apply_chat_template(
            messages[:-1],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 計算 completion（完整對話 - prompt）
        completion = prompt_and_completion[len(prompt):]
        
        return {
            "prompt": prompt,
            "completion": completion,
            "text": prompt + completion,
            "split": sample.get("metadata", "train"),
        }
    
    # 轉換格式
    print("🔄 轉換資料集格式中...")
    processed_data = []
    for item in data_list:
        formatted = apply_format(item)
        if formatted["prompt"] and formatted["completion"]:
            processed_data.append(formatted)
    
    print(f"✅ 資料集格式轉換完成！有效數據: {len(processed_data)} 筆")
    
    # 轉換為 Dataset
    from datasets import Dataset
    processed_dataset = Dataset.from_list(processed_data)
    
    # 分割訓練集與驗證集
    train_data = [x for x in processed_data if x['split'] == 'train']
    eval_data = [x for x in processed_data if x['split'] == 'eval']
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    print(f"📊 訓練集: {len(train_dataset)} 筆")
    print(f"📊 驗證集: {len(eval_dataset)} 筆")
    
    # 顯示範例
    if len(train_dataset) > 0:
        print(f"\n格式化範例:")
        print(f"Prompt: {train_dataset[0]['prompt'][:300]}...")
        print(f"Completion: {train_dataset[0]['completion'][:200]}...")
    
    return train_dataset, eval_dataset


def load_model(model_path):
    """載入已訓練的模型"""
    print(f"📥 載入模型: {model_path}")
    
    # 檢測設備
    if torch.backends.mps.is_available():
        device = "mps"
        # MPS 上使用 float32 更穩定，或使用 float16
        dtype = torch.float32
        print("🍎 使用 Apple Silicon MPS 加速")
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print("🎮 使用 CUDA GPU 加速")
    else:
        device = "cpu"
        dtype = torch.float32
        print("💻 使用 CPU")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 針對 MPS 不使用 device_map="auto"
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            attn_implementation="eager",
            torch_dtype=dtype
        )
    
    print(f"✅ 模型載入成功")
    print(f"Device: {model.device}")
    print(f"DType:  {model.dtype}")
    
    return model, tokenizer


def setup_lora(model):
    """設定 LoRA 配置"""
    print("🔧 設定 LoRA...")
    
    # 保守的 LoRA 配置 - 避免過擬合
    lora_config = LoraConfig(
        r=8,                       # 減小 rank 避免過擬合
        lora_alpha=16,             # alpha = 2 * r
        target_modules=["q_proj", "v_proj"],  # 只訓練關鍵層
        lora_dropout=0.1,          # 增加 dropout
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 檢查是否在 MPS 上，MPS 不支援 kbit training
    if not torch.cuda.is_available():
        # 在 MPS/CPU 上直接使用 PEFT，不用 kbit
        model.enable_input_require_grads()
    else:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def train_model(model, tokenizer, train_dataset, eval_dataset, args):
    """訓練模型"""
    print("🔧 設定訓練參數...")
    
    # 檢測設備並設定對應的優化參數
    use_mps = torch.backends.mps.is_available() and not torch.cuda.is_available()
    use_cuda = torch.cuda.is_available()
    
    # 根據設備選擇優化器和精度設定
    if use_cuda:
        optim = "adamw_8bit"  # CUDA 支援 8-bit 優化器
        bf16 = True
        fp16 = False
    else:
        optim = "adamw_torch_fused" if use_mps else "adamw_torch"  # MPS/CPU 使用標準優化器
        bf16 = False  # MPS 上 bf16 可能不穩定
        fp16 = False  # 使用 fp32 更穩定
    
    # 優化的訓練配置 - 使用 SFTConfig（TRL 0.25+）
    from trl import SFTConfig
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        
        # Batch size - 較小的 batch 配合 gradient accumulation
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=8,  # 增加累積步數
        
        # 學習率配置 - 繼續訓練用較低學習率
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,  # 增加 warmup
        
        # 日誌和保存
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        
        # 精度和優化器
        bf16=bf16,
        fp16=fp16,
        optim=optim,
        
        # 正則化
        max_grad_norm=0.5,  # 降低 gradient clipping
        weight_decay=0.01,
        
        # 其他設定
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        
        # 關閉 gradient checkpointing 以避免生成問題
        gradient_checkpointing=False,
        
        # SFT 特定配置
        max_length=512,  # 對話長度通常不長
        dataset_text_field="text",
        packing=False,  # 關閉 packing（沒有 flash attention 會有問題）
    )
    
    print(f"✅ 訓練參數設定完成")
    print(f"   優化器: {optim}")
    print(f"   BF16: {bf16}, FP16: {fp16}")
    print(f"   Packing: 關閉（避免樣本污染）")
    print(f"   學習率: {args.learning_rate}")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    print("🚀 開始訓練...")
    trainer.train()
    
    return trainer


def save_model(model, tokenizer, output_path):
    """保存模型"""
    print(f"💾 保存模型至: {output_path}")
    
    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"✅ 模型已保存至: {output_path}")
    
    return model


# Tool Schema for testing
TEST_TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "create_coffee_robot_mission",
        "description": "建立咖啡機器人外送任務。負責驗證訂單內容，並產生機器人任務指令。",
        "parameters": {
            "type": "object",
            "properties": {
                "baseDrink": {
                    "type": "string",
                    "enum": ["americano", "latte", "oat_latte", "milk"],
                    "description": "基礎飲品代號"
                },
                "floor": {
                    "type": "integer",
                    "description": "送達樓層，1-11"
                },
                "addons": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["extra_espresso", "paper_cup"]},
                    "description": "加購項目"
                },
                "quantity": {"type": "integer", "description": "數量"},
                "temperature": {"type": "string", "enum": ["hot", "iced"], "description": "溫度"}
            },
            "required": ["baseDrink", "floor"]
        }
    }
}]


def test_model(model, tokenizer):
    """測試模型 - Function Calling 和對話引導"""
    print("\n🧪 測試模型...")
    
    # 確保模型在 eval 模式
    model.eval()
    
    # 測試案例：完整資訊 -> 應該輸出 function call
    print("\n" + "=" * 50)
    print("【測試 1: 完整資訊 → Function Calling】")
    print("=" * 50)
    
    fc_test_cases = [
        "我要一杯冰拿鐵，送到5樓",
        "給我兩杯熱美式加濃縮，送到3樓",
        "幫我送一杯燕麥奶拿鐵到8樓，要冰的",
    ]
    
    for user_input in fc_test_cases:
        messages = [{"role": "user", "content": user_input}]
        
        input_text = tokenizer.apply_chat_template(
            messages,
            tools=TEST_TOOL_SCHEMA,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,  # 低溫度讓 function call 更準確
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        print(f"\n用戶: {user_input}")
        print(f"模型: {response[:500]}")
        print("-" * 50)
    
    # 測試案例：不完整資訊 -> 應該進行對話引導
    print("\n" + "=" * 50)
    print("【測試 2: 不完整資訊 → 對話引導】")
    print("=" * 50)
    
    dialog_test_cases = [
        "我要咖啡",
        "給我一杯拿鐵",
        "我想點飲料",
    ]
    
    for user_input in dialog_test_cases:
        messages = [{"role": "user", "content": user_input}]
        
        input_text = tokenizer.apply_chat_template(
            messages,
            tools=TEST_TOOL_SCHEMA,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print(f"\n用戶: {user_input}")
        print(f"助理: {response}")
        print("-" * 50)


def convert_to_gguf(model_path, output_file):
    """轉換為 GGUF 格式"""
    print(f"\n🔄 轉換為 GGUF 格式: {output_file}")
    
    cmd = [
        "python", "llama.cpp/convert_hf_to_gguf.py",
        model_path,
        "--outfile", output_file,
        "--outtype", "f16"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ GGUF 模型已保存至: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ GGUF 轉換失敗: {e}")
    except FileNotFoundError:
        print("❌ 找不到 llama.cpp/convert_hf_to_gguf.py，請確保 llama.cpp 已正確安裝")


def plot_training_curves(trainer):
    """繪製訓練曲線"""
    try:
        import matplotlib.pyplot as plt
        
        log_history = trainer.state.log_history
        train_loss = [log['loss'] for log in log_history if 'loss' in log]
        eval_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(eval_loss, label='Validation Loss', color='orange')
        plt.xlabel('Evaluation Steps')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        print("✅ 訓練曲線已保存為 training_curves.png")
    except ImportError:
        print("⚠️ matplotlib 未安裝，跳過繪製訓練曲線")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🚀 繼續訓練 FunctionGemma - 合併 Function Calling + 對話引導")
    print("=" * 60)
    
    # 載入模型
    model, tokenizer = load_model(args.model_path)
    
    # 載入數據集
    train_dataset, eval_dataset = load_and_prepare_dataset(tokenizer, args.dataset_path)
    
    # 設定 LoRA
    model = setup_lora(model)
    
    # 訓練
    trainer = train_model(model, tokenizer, train_dataset, eval_dataset, args)
    
    # 繪製訓練曲線
    plot_training_curves(trainer)
    
    # 保存模型
    model = save_model(model, tokenizer, args.final_model_path)
    
    # 測試模型
    if not args.skip_test:
        test_model(model, tokenizer)
    
    # 轉換為 GGUF
    if args.convert_gguf:
        gguf_output = args.final_model_path.rstrip('/') + ".gguf"
        convert_to_gguf(args.final_model_path, gguf_output)
    
    print("\n" + "=" * 60)
    print("✅ 訓練完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
