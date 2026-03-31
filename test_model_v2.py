#!/usr/bin/env python3
"""
測試改進後的模型 v2

測試項目：
1. Function Calling 模式（完整資訊）
2. 對話引導模式（不完整資訊）
3. 拒絕回應（菜單外品項）
4. 鮮奶/milk 識別
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 模型路徑
MODEL_PATH = "./coffee-robot-continued-v2-final"

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


def format_input(user_message: str, tools: list) -> str:
    """格式化輸入（符合 FunctionGemma 格式）"""
    # 格式化工具定義
    tools_str = ""
    for tool in tools:
        func = tool["function"]
        params = func["parameters"]
        props_str = json.dumps(params["properties"], ensure_ascii=False)
        tools_str += f"<start_function_declaration>declaration:{func['name']}{{description:<escape>{func['description']}<escape>,parameters:{{properties:{props_str},required:{json.dumps(params.get('required', []))}}}}}<end_function_declaration>\n"
    
    prompt = f"<bos><start_of_turn>developer\n{tools_str}<end_of_turn>\n<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
    return prompt


def test_model(model, tokenizer, test_cases: list, category: str):
    """測試模型"""
    print(f"\n{'='*60}")
    print(f"【{category}】")
    print(f"{'='*60}")
    
    correct = 0
    total = len(test_cases)
    
    for test in test_cases:
        user_input = test["input"]
        expected_type = test.get("expected_type", "fc")  # fc=function_call, dialog=對話
        expected_contains = test.get("expected_contains", [])
        expected_drink = test.get("expected_drink")
        
        prompt = format_input(user_input, [TOOL_SCHEMA])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response[len(prompt):].strip()
        
        # 截取第一個有效輸出
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]
        if "<start_function_call>" in response:
            # 只取第一個 function call
            parts = response.split("<start_function_call>")
            if len(parts) > 1:
                fc_part = parts[1].split("<end_function_call>")[0] if "<end_function_call>" in parts[1] else parts[1]
                response = f"<start_function_call>{fc_part}<end_function_call>"
        
        # 判斷是否正確
        is_correct = False
        
        if expected_type == "fc":
            # 期望是 function call
            if "<start_function_call>" in response:
                if expected_drink:
                    is_correct = f'"baseDrink": "{expected_drink}"' in response
                else:
                    is_correct = True
        elif expected_type == "dialog":
            # 期望是對話（不應該有 function call）
            is_correct = "<start_function_call>" not in response
            if expected_contains:
                for keyword in expected_contains:
                    if keyword not in response:
                        is_correct = False
                        break
        elif expected_type == "rejection":
            # 期望是拒絕回應
            is_correct = "<start_function_call>" not in response
            rejection_keywords = ["抱歉", "不好意思", "沒有", "無法", "只提供", "菜單"]
            has_rejection = any(kw in response for kw in rejection_keywords)
            is_correct = is_correct and has_rejection
        
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"\n{status} 輸入: {user_input}")
        print(f"   輸出: {response[:200]}...")
        if expected_drink:
            print(f"   期望飲品: {expected_drink}")
    
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n準確率: {correct}/{total} = {accuracy:.1f}%")
    return correct, total


def main():
    print("=" * 60)
    print("🧪 測試改進後的模型 v2")
    print("=" * 60)
    
    # 載入模型
    print(f"\n📥 載入模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    print(f"✅ 模型載入完成，設備: {model.device}")
    
    total_correct = 0
    total_tests = 0
    
    # 測試 1: Function Calling（完整資訊）
    fc_tests = [
        {"input": "我要一杯冰拿鐵，送到5樓", "expected_type": "fc", "expected_drink": "latte"},
        {"input": "給我兩杯熱美式送到3樓", "expected_type": "fc", "expected_drink": "americano"},
        {"input": "幫我送一杯燕麥奶拿鐵到8樓，要冰的", "expected_type": "fc", "expected_drink": "oat_latte"},
        {"input": "一杯熱鮮奶送到7樓", "expected_type": "fc", "expected_drink": "milk"},
        {"input": "冰美式加濃縮送到2樓", "expected_type": "fc", "expected_drink": "americano"},
        {"input": "熱牛奶送到10樓", "expected_type": "fc", "expected_drink": "milk"},
    ]
    c, t = test_model(model, tokenizer, fc_tests, "測試 1: Function Calling（完整資訊）")
    total_correct += c
    total_tests += t
    
    # 測試 2: 鮮奶/Milk 識別
    milk_tests = [
        {"input": "我要熱鮮奶送到5樓", "expected_type": "fc", "expected_drink": "milk"},
        {"input": "給我冰鮮乳送到3樓", "expected_type": "fc", "expected_drink": "milk"},
        {"input": "一杯熱牛奶送到6樓", "expected_type": "fc", "expected_drink": "milk"},
        {"input": "送一杯冰ㄋㄟㄋㄟ到9樓", "expected_type": "fc", "expected_drink": "milk"},
        {"input": "兩杯熱鮮奶送到4樓", "expected_type": "fc", "expected_drink": "milk"},
    ]
    c, t = test_model(model, tokenizer, milk_tests, "測試 2: 鮮奶/Milk 識別")
    total_correct += c
    total_tests += t
    
    # 測試 3: 對話引導（不完整資訊）
    dialog_tests = [
        {"input": "我要咖啡", "expected_type": "dialog"},
        {"input": "給我一杯拿鐵", "expected_type": "dialog"},
        {"input": "我想點飲料", "expected_type": "dialog"},
        {"input": "美式", "expected_type": "dialog"},
    ]
    c, t = test_model(model, tokenizer, dialog_tests, "測試 3: 對話引導（不完整資訊，不應該輸出 FC）")
    total_correct += c
    total_tests += t
    
    # 測試 4: 拒絕回應
    rejection_tests = [
        {"input": "我要一杯綠茶送到5樓", "expected_type": "rejection"},
        {"input": "給我珍珠奶茶送到3樓", "expected_type": "rejection"},
        {"input": "幫我送一杯可可到8樓", "expected_type": "rejection"},
    ]
    c, t = test_model(model, tokenizer, rejection_tests, "測試 4: 拒絕回應（菜單外品項）")
    total_correct += c
    total_tests += t
    
    # 總結
    print(f"\n{'='*60}")
    print(f"📊 總結")
    print(f"{'='*60}")
    overall_accuracy = total_correct / total_tests * 100 if total_tests > 0 else 0
    print(f"總準確率: {total_correct}/{total_tests} = {overall_accuracy:.1f}%")


if __name__ == "__main__":
    main()
