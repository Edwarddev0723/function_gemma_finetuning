#!/usr/bin/env python3
"""
合併 Function Calling 和對話引導數據集 v2

改進：
1. 增加鮮奶/milk 樣本
2. 增加拒絕回應（菜單外品項、超出限制）
3. 嚴格區分「對話模式」和「FC 模式」
   - FC 模式：資訊完整，直接輸出 function call，不輸出其他文字
   - 對話模式：資訊不完整，只輸出對話文字，不輸出 function call
"""

import json
import re
import random
import uuid
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
    "ㄋㄟㄋㄟ": "milk",
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


def create_function_call(order: Dict[str, Any]) -> Dict:
    """創建 function call 格式"""
    arguments = {
        "baseDrink": order["baseDrink"],
        "floor": order["floor"],
        "temperature": order.get("temperature", "hot"),
        "quantity": order.get("quantity", 1),
        "addons": order.get("addons", [])
    }
    
    return {
        "id": f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": "create_coffee_robot_mission",
            "arguments": json.dumps(arguments, ensure_ascii=False)
        }
    }


# ==========================================
# 增加鮮奶/milk 樣本
# ==========================================
def generate_milk_samples() -> List[Dict]:
    """生成鮮奶相關的 function calling 樣本"""
    samples = []
    
    milk_requests = [
        ("我要一杯熱鮮奶送到{floor}樓", "hot"),
        ("給我一杯冰鮮奶，送到{floor}樓", "iced"),
        ("幫我送一杯熱牛奶到{floor}樓", "hot"),
        ("我想要冰牛奶，送到{floor}樓", "iced"),
        ("一杯熱鮮乳送{floor}樓", "hot"),
        ("送一杯冰鮮乳到{floor}樓", "iced"),
        ("我要{quantity}杯熱鮮奶送到{floor}樓", "hot"),
        ("給我{quantity}杯冰牛奶，送{floor}樓", "iced"),
        ("幫我送{quantity}杯熱鮮乳到{floor}樓", "hot"),
        ("來杯ㄋㄟㄋㄟ，熱的，送到{floor}樓", "hot"),
        ("我要熱ㄋㄟㄋㄟ送{floor}樓", "hot"),
        ("一杯冰ㄋㄟㄋㄟ送到{floor}樓", "iced"),
    ]
    
    for _ in range(150):  # 生成 150 個鮮奶樣本
        template, temp = random.choice(milk_requests)
        floor = random.randint(1, 11)
        quantity = random.randint(1, 3)
        
        user_content = template.format(floor=floor, quantity=quantity)
        
        order = {
            "baseDrink": "milk",
            "floor": floor,
            "temperature": temp,
            "quantity": quantity if "{quantity}" in template else 1,
            "addons": []
        }
        
        sample = {
            "tools": [TOOL_SCHEMA],
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "tool_calls": [create_function_call(order)]}
            ],
            "metadata": "train"
        }
        samples.append(sample)
    
    return samples


# ==========================================
# 增加拒絕回應樣本
# ==========================================
def generate_rejection_samples() -> List[Dict]:
    """生成拒絕回應的樣本（菜單外品項、超出限制等）"""
    samples = []
    
    # 菜單外品項
    invalid_drinks = [
        "綠茶", "紅茶", "奶茶", "珍珠奶茶", "烏龍茶", "抹茶", "可可", 
        "柳橙汁", "蘋果汁", "檸檬水", "氣泡水", "啤酒", "果汁",
        "藍莓拿鐵", "草莓牛奶", "芒果冰沙", "焦糖瑪奇朵", "摩卡"
    ]
    
    rejection_responses = [
        "抱歉，我們目前只提供美式、拿鐵、燕麥奶拿鐵和鮮奶。請問您想要其中哪一種呢？",
        "不好意思，{drink}不在我們的菜單上。我們有美式、拿鐵、燕麥奶拿鐵、鮮奶可以選擇，請問您想要哪一種？",
        "很抱歉，我們沒有提供{drink}。目前菜單有：美式、拿鐵、燕麥奶拿鐵、鮮奶。請問您要點哪一款？",
    ]
    
    for drink in invalid_drinks:
        for floor in random.sample(range(1, 12), 3):
            user_content = random.choice([
                f"我要一杯{drink}送到{floor}樓",
                f"給我{drink}，送{floor}樓",
                f"幫我送一杯{drink}到{floor}樓",
            ])
            
            response = random.choice(rejection_responses).format(drink=drink)
            
            sample = {
                "tools": [TOOL_SCHEMA],
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response}
                ],
                "metadata": "train"
            }
            samples.append(sample)
    
    # 超出限制的請求（例如加兩份濃縮）
    over_limit_requests = [
        ("我要冰美式加兩份濃縮", "每杯只能加一份濃縮咖啡喔！請問您要加一份濃縮嗎？"),
        ("給我拿鐵加三份濃縮", "抱歉，每杯最多只能加一份濃縮咖啡。請問需要加一份嗎？"),
        ("熱拿鐵加兩份濃縮送到5樓", "不好意思，濃縮咖啡每杯限加一份。請問要加一份濃縮嗎？還是不用加？"),
        ("我要美式加五份濃縮", "抱歉，每杯只能加一份濃縮咖啡。請問您需要加一份嗎？"),
    ]
    
    for user_req, response in over_limit_requests:
        for _ in range(10):
            sample = {
                "tools": [TOOL_SCHEMA],
                "messages": [
                    {"role": "user", "content": user_req},
                    {"role": "assistant", "content": response}
                ],
                "metadata": "train"
            }
            samples.append(sample)
    
    # 取消訂單
    cancel_requests = [
        ("取消訂單", "好的，已為您取消訂單。如果還需要點餐，隨時告訴我！"),
        ("我不要了", "好的，訂單已取消。有需要再跟我說喔！"),
        ("算了不點了", "沒問題，已取消。如果改變主意，隨時歡迎點餐！"),
        ("取消", "好的，已取消您的訂單。"),
    ]
    
    for user_req, response in cancel_requests:
        for _ in range(15):
            sample = {
                "tools": [TOOL_SCHEMA],
                "messages": [
                    {"role": "user", "content": user_req},
                    {"role": "assistant", "content": response}
                ],
                "metadata": "train"
            }
            samples.append(sample)
    
    return samples


# ==========================================
# 嚴格區分對話模式（只對話，不輸出 FC）
# ==========================================
def generate_dialog_only_samples() -> List[Dict]:
    """
    生成純對話引導樣本
    特點：資訊不完整時，只輸出對話引導文字，絕對不輸出 function call
    """
    samples = []
    
    # 資訊不完整的對話
    incomplete_dialogs = [
        # 沒說要什麼飲品
        {
            "messages": [
                {"role": "user", "content": "我要咖啡"},
                {"role": "assistant", "content": "好的！請問您想要哪一種呢？我們有美式、拿鐵、燕麥奶拿鐵和鮮奶可以選擇。"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "我想點飲料"},
                {"role": "assistant", "content": "沒問題！我們有美式、拿鐵、燕麥奶拿鐵、鮮奶，請問您想要哪一種？"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "來杯咖啡"},
                {"role": "assistant", "content": "好的，請問要美式還是拿鐵呢？我們也有燕麥奶拿鐵和鮮奶喔！"}
            ]
        },
        # 有飲品但沒有樓層
        {
            "messages": [
                {"role": "user", "content": "我要一杯冰拿鐵"},
                {"role": "assistant", "content": "好的，一杯冰拿鐵。請問要送到幾樓呢？"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "給我熱美式"},
                {"role": "assistant", "content": "好的，一杯熱美式。請問送到哪一樓？"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "我要燕麥奶拿鐵，冰的"},
                {"role": "assistant", "content": "好的，冰燕麥奶拿鐵。請問要送到幾樓呢？"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "來杯熱鮮奶"},
                {"role": "assistant", "content": "好的，一杯熱鮮奶。請問送到哪一樓？"}
            ]
        },
        # 有飲品但沒說溫度
        {
            "messages": [
                {"role": "user", "content": "我要拿鐵送到5樓"},
                {"role": "assistant", "content": "好的，拿鐵送到5樓。請問要熱的還是冰的呢？"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "美式送3樓"},
                {"role": "assistant", "content": "好的，美式送到3樓。請問要熱的還是冰的？"}
            ]
        },
        # 詢問濃縮
        {
            "messages": [
                {"role": "user", "content": "我要熱拿鐵送到5樓"},
                {"role": "assistant", "content": "好的，熱拿鐵送到5樓。請問需要加一份濃縮咖啡嗎？"}
            ]
        },
        # 多輪對話
        {
            "messages": [
                {"role": "user", "content": "我想點咖啡"},
                {"role": "assistant", "content": "好的！我們有美式、拿鐵、燕麥奶拿鐵和鮮奶，請問您想要哪一種？"},
                {"role": "user", "content": "拿鐵"},
                {"role": "assistant", "content": "好的，拿鐵。請問要熱的還是冰的呢？"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "我要咖啡"},
                {"role": "assistant", "content": "好的！請問您想要美式、拿鐵、燕麥奶拿鐵還是鮮奶呢？"},
                {"role": "user", "content": "美式"},
                {"role": "assistant", "content": "好的，美式。請問要熱的還是冰的？"},
                {"role": "user", "content": "熱的"},
                {"role": "assistant", "content": "好的，熱美式。請問要送到幾樓呢？"}
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "給我鮮奶"},
                {"role": "assistant", "content": "好的，鮮奶。請問要熱的還是冰的呢？"},
                {"role": "user", "content": "熱的"},
                {"role": "assistant", "content": "好的，熱鮮奶。請問送到哪一樓？"}
            ]
        },
    ]
    
    # 複製多份以增加數量
    for dialog in incomplete_dialogs:
        for _ in range(20):
            sample = {
                "tools": [TOOL_SCHEMA],
                "messages": dialog["messages"].copy(),
                "metadata": "train"
            }
            samples.append(sample)
    
    return samples


# ==========================================
# 嚴格的 FC 模式（資訊完整直接輸出 FC）
# ==========================================
def generate_complete_fc_samples() -> List[Dict]:
    """
    生成資訊完整的 function calling 樣本
    特點：資訊完整時，直接輸出 function call，不輸出任何對話文字
    """
    samples = []
    
    drinks = ["americano", "latte", "oat_latte", "milk"]
    drink_names = {
        "americano": ["美式", "熱美式", "冰美式"],
        "latte": ["拿鐵", "熱拿鐵", "冰拿鐵"],
        "oat_latte": ["燕麥奶拿鐵", "燕麥拿鐵"],
        "milk": ["鮮奶", "鮮乳", "牛奶", "ㄋㄟㄋㄟ"]
    }
    
    templates = [
        "我要{qty}杯{temp}{drink}，送到{floor}樓",
        "給我{qty}杯{temp}{drink}送{floor}樓",
        "幫我送{qty}杯{temp}{drink}到{floor}樓",
        "{qty}杯{temp}{drink}送到{floor}樓",
        "我想要{temp}{drink}{qty}杯，送{floor}樓",
        "麻煩{qty}杯{temp}{drink}，送到{floor}樓",
    ]
    
    addon_templates = [
        "我要{qty}杯{temp}{drink}加濃縮，送到{floor}樓",
        "給我{qty}杯{temp}{drink}加一份濃縮，送{floor}樓",
        "{temp}{drink}{qty}杯加濃縮送到{floor}樓",
        "幫我送{qty}杯{temp}{drink}加濃縮到{floor}樓，用紙杯",
    ]
    
    for _ in range(300):
        drink_code = random.choice(drinks)
        drink_name = random.choice(drink_names[drink_code])
        temp = random.choice(["hot", "iced"])
        temp_name = "熱" if temp == "hot" else "冰"
        floor = random.randint(1, 11)
        qty = random.choice([1, 1, 1, 2, 3])  # 1 杯更常見
        
        # 決定是否加濃縮
        if random.random() < 0.3 and drink_code != "milk":
            template = random.choice(addon_templates)
            addons = ["extra_espresso"]
            if "紙杯" in template:
                addons.append("paper_cup")
        else:
            template = random.choice(templates)
            addons = []
        
        # 處理數量文字
        qty_text = "" if qty == 1 else f"{qty}"
        if qty == 1 and "{qty}" in template:
            template = template.replace("{qty}杯", "一杯")
        
        user_content = template.format(
            qty=qty_text, temp=temp_name, drink=drink_name, floor=floor
        ).replace("杯杯", "杯").strip()
        
        order = {
            "baseDrink": drink_code,
            "floor": floor,
            "temperature": temp,
            "quantity": qty,
            "addons": addons
        }
        
        sample = {
            "tools": [TOOL_SCHEMA],
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "tool_calls": [create_function_call(order)]}
            ],
            "metadata": "train"
        }
        samples.append(sample)
    
    return samples


def load_function_calling_dataset(path: str) -> List[Dict]:
    """載入原本的 function calling 數據集"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data


def convert_hf_dialog_to_dialog_only(conversation: List[Dict]) -> Optional[Dict]:
    """
    將 HuggingFace 對話數據轉換為純對話格式（不輸出 FC）
    只保留沒有完整資訊的對話
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
    
    # 檢查是否包含樓層資訊（如果有樓層，就不用這個樣本）
    full_text = " ".join([m["content"] for m in filtered_conv])
    if re.search(r'\d+\s*樓', full_text):
        return None  # 有樓層資訊的不用於對話模式
    
    return {
        "tools": [TOOL_SCHEMA],
        "messages": filtered_conv,
        "metadata": "train"
    }


def main():
    print("=" * 60)
    print("🔧 合併數據集 v2 - 改進版")
    print("   - 增加鮮奶樣本")
    print("   - 增加拒絕回應")
    print("   - 嚴格區分對話模式和 FC 模式")
    print("=" * 60)
    
    all_data = []
    
    # 1. 載入原本的 function calling 數據集
    print("\n📥 載入原始 Function Calling 數據集...")
    fc_data = load_function_calling_dataset("ollama_mcp_dataset.jsonl")
    print(f"   原始 FC 數據: {len(fc_data)} 筆")
    all_data.extend(fc_data)
    
    # 2. 生成鮮奶樣本
    print("\n🥛 生成鮮奶樣本...")
    milk_samples = generate_milk_samples()
    print(f"   鮮奶樣本: {len(milk_samples)} 筆")
    all_data.extend(milk_samples)
    
    # 3. 生成拒絕回應
    print("\n🚫 生成拒絕回應樣本...")
    rejection_samples = generate_rejection_samples()
    print(f"   拒絕回應: {len(rejection_samples)} 筆")
    all_data.extend(rejection_samples)
    
    # 4. 生成純對話引導樣本
    print("\n💬 生成對話引導樣本...")
    dialog_samples = generate_dialog_only_samples()
    print(f"   對話引導: {len(dialog_samples)} 筆")
    all_data.extend(dialog_samples)
    
    # 5. 生成完整資訊 FC 樣本
    print("\n📞 生成完整資訊 FC 樣本...")
    complete_fc_samples = generate_complete_fc_samples()
    print(f"   完整 FC: {len(complete_fc_samples)} 筆")
    all_data.extend(complete_fc_samples)
    
    # 6. 載入 HuggingFace 對話數據（只取對話模式）
    print("\n📥 載入 HuggingFace 對話數據...")
    hf_dataset = load_dataset("renhehuang/coffee-order-zhtw")
    hf_dialog_count = 0
    for item in tqdm(hf_dataset['train'], desc="轉換對話數據"):
        converted = convert_hf_dialog_to_dialog_only(item['conversations'])
        if converted:
            all_data.append(converted)
            hf_dialog_count += 1
    print(f"   HuggingFace 對話: {hf_dialog_count} 筆")
    
    # 打亂數據
    random.shuffle(all_data)
    
    # 分割訓練集和驗證集
    eval_count = int(len(all_data) * 0.1)
    eval_data = all_data[:eval_count]
    train_data = all_data[eval_count:]
    
    # 標記 metadata
    for item in eval_data:
        item["metadata"] = "eval"
    for item in train_data:
        item["metadata"] = "train"
    
    # 統計
    fc_count = sum(1 for item in all_data if item.get("messages") and 
                   len(item["messages"]) > 0 and 
                   "tool_calls" in item["messages"][-1])
    dialog_count = len(all_data) - fc_count
    
    # 保存
    output_path = "merged_coffee_dataset_v2.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in train_data + eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n{'=' * 60}")
    print("✅ 數據集準備完成!")
    print(f"{'=' * 60}")
    print(f"   總數據量: {len(all_data)} 筆")
    print(f"   - Function Calling 模式: {fc_count} 筆")
    print(f"   - 對話引導模式: {dialog_count} 筆")
    print(f"   訓練集: {len(train_data)} 筆")
    print(f"   驗證集: {len(eval_data)} 筆")
    print(f"   保存至: {output_path}")
    
    # 顯示範例
    print(f"\n{'=' * 60}")
    print("📋 數據範例")
    print(f"{'=' * 60}")
    
    # FC 範例
    for item in train_data:
        if "tool_calls" in item.get("messages", [{}])[-1]:
            print("\n【Function Calling 範例】")
            print(f"用戶: {item['messages'][0]['content']}")
            print(f"助理: {item['messages'][-1]['tool_calls'][0]['function']['arguments']}")
            break
    
    # 對話範例
    for item in train_data:
        msgs = item.get("messages", [])
        if msgs and "content" in msgs[-1] and "tool_calls" not in msgs[-1]:
            print("\n【對話引導範例】")
            for msg in msgs:
                role = "用戶" if msg["role"] == "user" else "助理"
                print(f"{role}: {msg.get('content', '')}")
            break
    
    # 拒絕範例
    for item in train_data:
        msgs = item.get("messages", [])
        if msgs and len(msgs) == 2:
            content = msgs[-1].get("content", "")
            if "抱歉" in content or "不好意思" in content:
                print("\n【拒絕回應範例】")
                print(f"用戶: {msgs[0]['content']}")
                print(f"助理: {content}")
                break


if __name__ == "__main__":
    main()
