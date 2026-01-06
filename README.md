# ☕ FunctionGemma Coffee Robot MCP Fine-tuning

基於 Google FunctionGemma 270M 模型，針對咖啡機器人外送任務進行 Function Calling 微調的完整專案。

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.57+-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 專案概述

此專案展示如何：
1. 使用本地 LLM (Ollama) 自動生成高品質的 Function Calling 訓練資料
2. 微調 FunctionGemma 模型以適應特定領域任務
3. 在 Apple Silicon (M4 Pro) 上進行高效訓練

### 應用場景
將自然語言指令轉換為結構化的咖啡外送任務：

```
使用者：「幫我送一杯熱美式到五樓」

↓ FunctionGemma 微調模型 ↓

Function Call: create_coffee_robot_mission({
    "baseDrink": "americano",
    "floor": 5,
    "temperature": "hot",
    "quantity": 1
})
```

## 🗂️ 專案結構

```
function_gemma_finetuning/
├── README.md                                    # 專案說明文件
├── .gitignore                                   # Git 忽略規則
├── dataset_prepare.py                           # 資料集生成腳本
├── ollama_mcp_dataset.jsonl                     # 生成的訓練資料集
├── Finetune_FunctionGemma_Coffee_Robot_MCP.ipynb # 微調訓練 Notebook
├── eval_base_model.json                         # 基礎模型評估結果
├── eval_trained_model.json                      # 微調模型評估結果
└── coffee-robot-functiongemma/                  # 微調後的模型輸出
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── ...
```

## 🚀 快速開始

### 環境需求

- Python 3.11+
- PyTorch 2.0+
- 48GB+ RAM (建議，用於 Apple Silicon)
- [Ollama](https://ollama.ai/) (用於生成資料集)

### 安裝套件

```bash
pip install torch
pip install transformers==4.57.1 trl==0.25.1 datasets==4.4.1
pip install matplotlib pandas tqdm requests
```

### 步驟 1：生成訓練資料集

確保 Ollama 服務已啟動：

```bash
ollama serve
```

執行資料集生成腳本：

```bash
python dataset_prepare.py
```

這會使用本地 LLM 生成 1000 筆咖啡外送任務的 Function Calling 範例。

### 步驟 2：微調模型

開啟 Jupyter Notebook 並執行：

```bash
jupyter notebook Finetune_FunctionGemma_Coffee_Robot_MCP.ipynb
```

或在 VS Code 中直接開啟 `.ipynb` 檔案。

## 🛠️ 支援的 Function

### `create_coffee_robot_mission`

| 參數 | 類型 | 說明 | 可選值 |
|------|------|------|--------|
| `baseDrink` | string | 飲品種類 | `americano`, `latte`, `oat_latte`, `milk` |
| `floor` | integer | 配送樓層 | 1-11 |
| `temperature` | string | 飲品溫度 | `hot`, `iced` |
| `addons` | array | 加購選項 | `extra_espresso`, `paper_cup` |
| `quantity` | integer | 數量 | 1-10 |

## 📊 資料集格式

訓練資料採用 JSONL 格式，每筆資料包含：

```json
{
  "messages": [
    {"role": "user", "content": "幫我送一杯熱美式到五樓"},
    {"role": "assistant", "tool_calls": [...]}
  ],
  "tools": [...],
  "metadata": "train"
}
```

## ⚙️ 訓練配置

針對 Apple Silicon M4 Pro 48GB 優化的訓練參數：

| 參數 | 值 | 說明 |
|------|-----|------|
| `per_device_train_batch_size` | 1 | 減少記憶體使用 |
| `gradient_accumulation_steps` | 16 | 有效批次大小 = 16 |
| `learning_rate` | 1e-5 | 學習率 |
| `num_train_epochs` | 3 | 訓練輪數 |
| `gradient_checkpointing` | True | 節省記憶體 |
| `optim` | adamw_torch | MPS 相容優化器 |

## 💻 使用微調後的模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 載入模型
model_path = "./coffee-robot-functiongemma"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# 建立 pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 定義工具
tools = [{
    "name": "create_coffee_robot_mission",
    "description": "建立咖啡機器人外送任務",
    "parameters": {
        "type": "object",
        "properties": {
            "baseDrink": {"type": "string", "enum": ["americano", "latte", "oat_latte", "milk"]},
            "floor": {"type": "integer", "minimum": 1, "maximum": 11},
            "temperature": {"type": "string", "enum": ["hot", "iced"]},
            "addons": {"type": "array", "items": {"type": "string"}},
            "quantity": {"type": "integer"}
        },
        "required": ["baseDrink", "floor"]
    }
}]

# 推論
messages = [{"role": "user", "content": "幫我送三杯冰拿鐵到七樓"}]
prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
output = pipe(prompt, max_new_tokens=256)
print(output[0]['generated_text'][len(prompt):])
```

## 📈 模型效能

| 指標 | 基礎模型 | 微調後模型 |
|------|----------|------------|
| Function Calling 準確率 | ~10% | ~95%+ |

## 🔗 相關資源

- [FunctionGemma 模型](https://huggingface.co/google/functiongemma-270m-it)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [Ollama](https://ollama.ai/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)

## 📝 授權

本專案程式碼採用 MIT 授權。

微調模型基於 [Gemma License](https://ai.google.dev/gemma/terms)，使用前請確認已接受 Google Gemma 的使用條款。

## 🙏 致謝

- [Google](https://ai.google.dev/) - 提供 FunctionGemma 基礎模型
- [Hugging Face](https://huggingface.co/) - 提供模型託管與訓練工具
- [Ollama](https://ollama.ai/) - 提供本地 LLM 服務

---

Made with ☕ by Edward Huang
