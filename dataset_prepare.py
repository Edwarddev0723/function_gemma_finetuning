import json
import random
import uuid
import requests  # pip install requests
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # pip install tqdm

# ==========================================
# 1. 配置設定
# ==========================================
# 請確保你的 Ollama 服務正在運行 (ollama serve)
OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gpt-oss:20b"  # 請根據你本地有的模型修改，例如 "mistral", "llama3", "gemma2"
OUTPUT_FILE = "ollama_mcp_dataset.jsonl"
TOTAL_SAMPLES = 1000   # 想要生成的總筆數
EVAL_RATIO = 0.1     # 驗證集比例
BATCH_SIZE = 10      # 每次生成的樣本數（增大以減少請求次數）
MAX_WORKERS = 4      # 並行請求數量

# ==========================================
# 2. MCP 業務邏輯與常數 (驗證用)
# ==========================================
BASE_DRINKS = {
    "americano": "美式",
    "latte": "拿鐵",
    "oat_latte": "燕麥奶拿鐵",
    "milk": "鮮乳"
}
ADDONS = {
    "extra_espresso": "加購一份濃縮咖啡",
    "paper_cup": "紙杯"
}

# 工具定義 Schema
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_coffee_robot_mission",
        "description": "建立咖啡機器人外送任務。負責驗證訂單內容，並產生機器人任務指令 (Mock Nuwa Payload)。",
        "parameters": {
            "type": "object",
            "properties": {
                "baseDrink": {
                    "type": "string",
                    "enum": list(BASE_DRINKS.keys()),
                    "description": "基礎飲品代號，只能是: " + ", ".join(BASE_DRINKS.keys())
                },
                "floor": {
                    "type": "integer",
                    "description": "送達樓層，必須介於 1 到 11 之間"
                },
                "addons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(ADDONS.keys())
                    },
                    "description": "加購項目清單"
                },
                "quantity": {
                    "type": "integer",
                    "description": "數量，預設為 1"
                },
                "temperature": {
                    "type": "string",
                    "enum": ["hot", "iced"],
                    "description": "溫度，只能是 hot 或 iced，預設為 hot"
                }
            },
            "required": ["baseDrink", "floor"]
        }
    }
}

# ==========================================
# 3. 核心函數：使用 Ollama 生成自然語言
# ==========================================

def query_ollama(prompt: str, system_prompt: str, timeout: int = 120) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "format": "json",  # 強制讓 Ollama 回傳 JSON 格式
        "options": {
            "num_predict": 2048,  # 限制生成長度以加速
            "temperature": 0.8,   # 適度多樣性
        }
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.Timeout:
        return ""
    except Exception as e:
        print(f"\nOllama API Error: {e}")
        return ""

# 系統提示詞（全域常數，避免重複字串）
SYSTEM_PROMPT = """你是一個資料生成助手。請生成真實、口語化的使用者點餐指令，並對應到正確的 JSON 參數。

業務規則：
- 飲品 (baseDrink): americano, latte, oat_latte, milk
- 樓層 (floor): 1~11
- 溫度 (temperature): hot, iced
- 加購 (addons): extra_espresso, paper_cup
- 數量 (quantity): 預設 1

請回傳一個 JSON Object，包含一個 "data" 列表，列表中的每個物件格式如下：
{
  "user_input": "幫我送一杯熱美式去五樓",
  "args": {
    "baseDrink": "americano", 
    "floor": 5, 
    "temperature": "hot", 
    "quantity": 1,
    "addons": []
  }
}

請生成多樣化的語句，包含：
1. 簡單指令 ("一杯拿鐵到3樓")
2. 複雜需求 ("我要三杯燕麥奶拿鐵，都要冰的，送到11樓會議室")
3. 隱晦需求 ("好累喔，來杯加濃縮的美式提神，我在7樓")
"""

def generate_synthetic_data(batch_size: int = 10) -> List[Dict]:
    user_prompt = f"請生成 {batch_size} 筆測試資料。請確保 JSON 格式正確且參數符合業務規則。"

    response_text = query_ollama(user_prompt, SYSTEM_PROMPT)

    if not response_text:
        return []

    try:
        data = json.loads(response_text)
        if isinstance(data, list):
            return data
        elif "data" in data:
            return data["data"]
        else:
            return []
    except json.JSONDecodeError:
        return []


def generate_batch_parallel(num_batches: int, batch_size: int) -> List[Dict]:
    """並行執行多個批次請求"""
    all_results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_synthetic_data, batch_size) for _ in range(num_batches)]
        
        for future in as_completed(futures):
            try:
                batch = future.result()
                all_results.extend(batch)
            except Exception:
                pass
    
    return all_results

# ==========================================
# 4. 驗證與格式轉換
# ==========================================

def validate_args(args: Dict) -> bool:
    # 簡單驗證生成的參數是否符合 MCP 規則
    try:
        if args.get("baseDrink") not in BASE_DRINKS: return False
        if not (1 <= args.get("floor", 0) <= 11): return False
        if args.get("temperature") not in ["hot", "iced"]: return False
        if args.get("quantity", 0) < 1: return False
        # 檢查 addons
        for addon in args.get("addons", []):
            if addon not in ADDONS: return False
        return True
    except:
        return False

def create_dataset_entry(user_input: str, args: Dict, split: str) -> Dict:
    # 確保 args 是 JSON 字串
    args_str = json.dumps(args, ensure_ascii=False)

    return {
        "metadata": split,
        "tools": [TOOL_SCHEMA],
        "messages": [
            {
                "role": "user",
                "content": user_input
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": "create_coffee_robot_mission",
                            "arguments": args_str
                        }
                    }
                ]
            }
        ]
    }

# ==========================================
# 5. 主程式執行
# ==========================================

def main():
    print(f"🚀 開始使用 Ollama ({MODEL_NAME}) 生成資料集...")
    print(f"   目標: {TOTAL_SAMPLES} 筆 | 批次大小: {BATCH_SIZE} | 並行數: {MAX_WORKERS}")
    print("-" * 50)
    
    valid_samples = []
    seen_inputs = set()  # 用於去重

    # 計算需要的總批次數（考慮驗證失敗率，多請求一些）
    estimated_batches = (TOTAL_SAMPLES * 2) // BATCH_SIZE + 1
    
    # 使用 tqdm 進度條
    pbar = tqdm(total=TOTAL_SAMPLES, desc="生成有效資料", unit="筆", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    max_rounds = 50  # 最多執行 50 輪並行請求
    round_count = 0
    
    while len(valid_samples) < TOTAL_SAMPLES and round_count < max_rounds:
        round_count += 1
        
        # 計算這一輪需要多少並行請求
        remaining = TOTAL_SAMPLES - len(valid_samples)
        num_batches = min(MAX_WORKERS, (remaining // BATCH_SIZE) + 1)
        
        # 並行生成資料
        batch_results = generate_batch_parallel(num_batches, BATCH_SIZE)
        
        # 驗證並添加有效資料
        for item in batch_results:
            if len(valid_samples) >= TOTAL_SAMPLES:
                break

            user_input = item.get("user_input", "")
            args = item.get("args")

            # 驗證資料有效性並去重
            if user_input and args and validate_args(args) and user_input not in seen_inputs:
                seen_inputs.add(user_input)
                split = "eval" if random.random() < EVAL_RATIO else "train"
                entry = create_dataset_entry(user_input, args, split)
                valid_samples.append(entry)
                pbar.update(1)
    
    pbar.close()

    # 寫入檔案
    print("\n💾 寫入檔案中...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in valid_samples:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 統計結果
    train_count = sum(1 for e in valid_samples if e["metadata"] == "train")
    eval_count = sum(1 for e in valid_samples if e["metadata"] == "eval")
    
    print(f"\n✅ 生成完成！檔案已儲存至: {OUTPUT_FILE}")
    print(f"   總計: {len(valid_samples)} 筆 | 訓練集: {train_count} 筆 | 驗證集: {eval_count} 筆")

if __name__ == "__main__":
    main()