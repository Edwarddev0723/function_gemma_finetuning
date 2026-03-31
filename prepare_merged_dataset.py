#!/usr/bin/env python3
"""
合併 Function Calling 和對話引導數據集

這個腳本將：
1. 載入原本的 function calling 數據集 (ollama_mcp_dataset.jsonl)
2. 載入對話引導數據集 (renhehuang/coffee-order-zhtw)
3. 將對話數據轉換為合適的格式（最終輪次產生 function call）
4. 合併兩個數據集
"""

import json
import re
import random
from datasets import load_dataset
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# 飲品映射
DRINK_MAPPING = {
    "美式": "americano",
    "拿鐵": "latte", 
    "燕麥奶拿鐵": "oat_latte",
    "燕麥拿鐵": "oat_latte",
    "鮮奶": "milk",
    "鮮乳": "milk",
    "牛奶": "milk",
}

# 溫度映射
TEMP_MAPPING = {
    "冰": "iced",
    "冰的": "iced",
    "熱": "hot",
    "熱的": "hot",
}

# Tool Schema
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
                    "enum": ["americano", "latte", "oat_latte", "milk"],
                    "description": "基礎飲品代號，只能是: americano, latte, oat_latte, milk"
                },
                "floor": {
                    "type": "integer",
                    "description": "送達樓層，必須介於 1 到 11 之間"
                },
                "addons": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["extra_espresso", "paper_cup"]},
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


def extract_order_info(conversation: List[Dict]) -> Dict[str, Any]:
    """從對話中提取訂單資訊"""
    order = {
        "baseDrink": None,
        "temperature": None,
        "quantity": 1,
        "addons": [],
        "floor": None,  # 對話數據集沒有樓層資訊
    }
    
    full_text = " ".join([msg["content"] for msg in conversation])
    
    # 提取飲品
    for zh_name, en_name in DRINK_MAPPING.items():
        if zh_name in full_text:
            order["baseDrink"] = en_name
            break
    
    # 提取溫度
    for zh_temp, en_temp in TEMP_MAPPING.items():
        if zh_temp in full_text:
            order["temperature"] = en_temp
            break
    
    # 提取數量
    quantity_match = re.search(r'(\d+)\s*杯', full_text)
    if quantity_match:
        order["quantity"] = int(quantity_match.group(1))
    
    # 提取加購項目
    if "濃縮" in full_text and ("加" in full_text or "要" in full_text):
        if "不" not in full_text.split("濃縮")[0][-5:]:  # 檢查前面有沒有"不"
            order["addons"].append("extra_espresso")
    
    return order


def create_function_call(order: Dict[str, Any]) -> Dict:
    """創建 function call 格式"""
    import uuid
    
    arguments = {
        "baseDrink": order["baseDrink"],
        "floor": order["floor"] if order["floor"] else random.randint(1, 11),
        "temperature": order["temperature"] if order["temperature"] else "hot",
        "quantity": order["quantity"],
        "addons": order["addons"]
    }
    
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": "create_coffee_robot_mission",
            "arguments": json.dumps(arguments, ensure_ascii=False)
        }
    }


def convert_conversation_to_training_format(conversation: List[Dict], add_floor: bool = True) -> Optional[Dict]:
    """
    將對話轉換為訓練格式
    
    策略：
    1. 多輪對話用於引導用戶補充資訊
    2. 最後一輪產生 function call（當資訊足夠時）
    """
    # 過濾掉 system message，只保留 user 和 assistant
    filtered_conv = []
    for msg in conversation:
        role = msg.get("role", "")
        if role == "system":
            continue
        elif role == "user":
            filtered_conv.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            filtered_conv.append({"role": "assistant", "content": msg["content"]})
    
    if len(filtered_conv) < 2:
        return None
    
    # 提取訂單資訊
    order = extract_order_info(conversation)
    
    # 如果沒有識別到飲品，跳過
    if not order["baseDrink"]:
        return None
    
    # 添加隨機樓層（對話數據集沒有樓層資訊）
    if add_floor:
        order["floor"] = random.randint(1, 11)
        # 在最後一個 user message 加入樓層資訊
        last_user_idx = None
        for i in range(len(filtered_conv) - 1, -1, -1):
            if filtered_conv[i]["role"] == "user":
                last_user_idx = i
                break
        
        if last_user_idx is not None:
            floor = order["floor"]
            floor_phrases = [
                f"，送到{floor}樓",
                f"，請送到{floor}樓",
                f"，麻煩送到{floor}樓",
            ]
            filtered_conv[last_user_idx]["content"] += random.choice(floor_phrases)
    
    # 創建 function call
    function_call = create_function_call(order)
    
    # 構建最終格式
    messages = filtered_conv[:-1]  # 除了最後一個 assistant 回應
    
    # 最後一個 assistant 回應改為 function call
    messages.append({
        "role": "assistant",
        "tool_calls": [function_call]
    })
    
    return {
        "tools": [TOOL_SCHEMA],
        "messages": messages,
        "metadata": "train"
    }


def convert_conversation_to_dialog_only(conversation: List[Dict]) -> Optional[Dict]:
    """
    將對話保持為純對話格式（用於引導訓練）
    不產生 function call，只保留對話引導能力
    """
    filtered_conv = []
    for msg in conversation:
        role = msg.get("role", "")
        if role == "system":
            continue
        elif role == "user":
            filtered_conv.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            filtered_conv.append({"role": "assistant", "content": msg["content"]})
    
    if len(filtered_conv) < 2:
        return None
    
    return {
        "tools": [TOOL_SCHEMA],
        "messages": filtered_conv,
        "metadata": "train"
    }


def load_function_calling_dataset(path: str) -> List[Dict]:
    """載入原本的 function calling 數據集"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def load_and_convert_dialog_dataset() -> List[Dict]:
    """載入並轉換對話數據集"""
    print("📥 載入對話數據集 renhehuang/coffee-order-zhtw...")
    dataset = load_dataset("renhehuang/coffee-order-zhtw")
    
    converted_data = []
    dialog_only_data = []
    
    for item in tqdm(dataset['train'], desc="轉換對話數據"):
        conversations = item['conversations']
        
        # 50% 轉換為帶 function call 的格式
        # 50% 保留為純對話格式（用於訓練對話引導能力）
        if random.random() < 0.5:
            converted = convert_conversation_to_training_format(conversations)
            if converted:
                converted_data.append(converted)
        else:
            dialog = convert_conversation_to_dialog_only(conversations)
            if dialog:
                dialog_only_data.append(dialog)
    
    print(f"   轉換為 Function Call 格式: {len(converted_data)} 筆")
    print(f"   保留為對話引導格式: {len(dialog_only_data)} 筆")
    
    return converted_data + dialog_only_data


def main():
    print("=" * 60)
    print("🔧 合併 Function Calling 和對話引導數據集")
    print("=" * 60)
    
    # 載入原本的 function calling 數據集
    print("\n📥 載入 Function Calling 數據集...")
    fc_data = load_function_calling_dataset("ollama_mcp_dataset.jsonl")
    print(f"   Function Calling 數據: {len(fc_data)} 筆")
    
    # 載入並轉換對話數據集
    dialog_data = load_and_convert_dialog_dataset()
    
    # 合併數據集
    merged_data = fc_data + dialog_data
    random.shuffle(merged_data)
    
    # 分割訓練集和驗證集
    eval_count = int(len(merged_data) * 0.1)
    eval_data = merged_data[:eval_count]
    train_data = merged_data[eval_count:]
    
    # 標記 metadata
    for item in eval_data:
        item["metadata"] = "eval"
    for item in train_data:
        item["metadata"] = "train"
    
    # 保存合併後的數據集
    output_path = "merged_coffee_dataset.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in train_data + eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 合併完成!")
    print(f"   總數據量: {len(merged_data)} 筆")
    print(f"   訓練集: {len(train_data)} 筆")
    print(f"   驗證集: {len(eval_data)} 筆")
    print(f"   保存至: {output_path}")
    
    # 顯示範例
    print("\n" + "=" * 60)
    print("📋 數據範例")
    print("=" * 60)
    
    # 顯示一個 function calling 範例
    for item in train_data:
        if item.get("messages") and len(item["messages"]) > 0:
            last_msg = item["messages"][-1]
            if "tool_calls" in last_msg:
                print("\n【Function Calling 範例】")
                print(json.dumps(item, indent=2, ensure_ascii=False)[:1000])
                break
    
    # 顯示一個純對話範例
    for item in train_data:
        if item.get("messages") and len(item["messages"]) > 0:
            last_msg = item["messages"][-1]
            if "tool_calls" not in last_msg and "content" in last_msg:
                print("\n【對話引導範例】")
                print(json.dumps(item, indent=2, ensure_ascii=False)[:1000])
                break


if __name__ == "__main__":
    main()
