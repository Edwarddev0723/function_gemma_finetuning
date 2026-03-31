#!/usr/bin/env python3
"""
完整推論測試腳本

測試 coffee-robot-continued-final 模型的 Function Calling 和對話引導能力
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# 模型路徑
MODEL_PATH = "./coffee-robot-continued-final"

# Tool Schema
TOOL_SCHEMA = [{
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


def load_model():
    """載入模型和 tokenizer"""
    print(f"📥 載入模型: {MODEL_PATH}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # 檢測設備
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    
    model.eval()
    print(f"✅ 模型載入成功 (Device: {device})")
    
    return model, tokenizer


def extract_function_call(response: str) -> dict:
    """從回應中提取 function call"""
    # 尋找 function call 標記
    pattern = r'<start_function_call>call:create_coffee_robot_mission\{.*?\{(.*?)\}\}<end_function_call>'
    match = re.search(pattern, response, re.DOTALL)
    
    if match:
        try:
            json_str = '{' + match.group(1).strip() + '}'
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 嘗試另一種格式
    pattern2 = r'\{"baseDrink".*?\}'
    match2 = re.search(pattern2, response)
    if match2:
        try:
            return json.loads(match2.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def generate_response(model, tokenizer, user_input: str, max_new_tokens: int = 150, temperature: float = 0.1):
    """生成模型回應"""
    messages = [{"role": "user", "content": user_input}]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tools=TOOL_SCHEMA,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )
    
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # 清理回應，只取第一個 function call 或第一段文字
    if '<end_function_call>' in response:
        response = response.split('<end_function_call>')[0] + '<end_function_call>'
    elif '<end_of_turn>' in response:
        response = response.split('<end_of_turn>')[0]
    
    return response


def test_function_calling(model, tokenizer):
    """測試 Function Calling 能力"""
    print("\n" + "=" * 70)
    print("🔧 【測試 1: Function Calling - 完整資訊直接呼叫函數】")
    print("=" * 70)
    
    test_cases = [
        ("我要一杯冰拿鐵，送到5樓", {"baseDrink": "latte", "floor": 5, "temperature": "iced"}),
        ("給我兩杯熱美式加濃縮，送到3樓", {"baseDrink": "americano", "floor": 3, "temperature": "hot", "quantity": 2, "addons": ["extra_espresso"]}),
        ("幫我送一杯燕麥奶拿鐵到8樓，要冰的", {"baseDrink": "oat_latte", "floor": 8, "temperature": "iced"}),
        ("一杯熱鮮奶送到1樓", {"baseDrink": "milk", "floor": 1, "temperature": "hot"}),
        ("送3杯冰美式到11樓，用紙杯", {"baseDrink": "americano", "floor": 11, "temperature": "iced", "quantity": 3, "addons": ["paper_cup"]}),
        ("我要熱拿鐵加濃縮，送到6樓", {"baseDrink": "latte", "floor": 6, "temperature": "hot", "addons": ["extra_espresso"]}),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for user_input, expected in test_cases:
        response = generate_response(model, tokenizer, user_input, max_new_tokens=150, temperature=0.1)
        parsed = extract_function_call(response)
        
        # 檢查是否正確
        is_correct = False
        if parsed:
            # 檢查關鍵欄位
            if (parsed.get("baseDrink") == expected.get("baseDrink") and 
                parsed.get("floor") == expected.get("floor")):
                is_correct = True
                correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} 用戶: {user_input}")
        print(f"   期望: {expected}")
        print(f"   解析: {parsed}")
        print(f"   原始: {response[:200]}...")
    
    accuracy = correct / total * 100
    print(f"\n📊 Function Calling 準確率: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy


def test_dialog_guidance(model, tokenizer):
    """測試對話引導能力"""
    print("\n" + "=" * 70)
    print("💬 【測試 2: 對話引導 - 資訊不完整時引導用戶】")
    print("=" * 70)
    
    test_cases = [
        "我要咖啡",
        "給我一杯拿鐵",
        "我想點飲料",
        "來杯美式",
        "我要燕麥奶拿鐵",
    ]
    
    for user_input in test_cases:
        response = generate_response(model, tokenizer, user_input, max_new_tokens=100, temperature=0.7)
        
        # 清理回應
        clean_response = response.replace('<start_function_call>', '').replace('<end_function_call>', '')
        clean_response = clean_response.replace('<end_of_turn>', '').strip()
        
        # 判斷是對話還是 function call
        is_dialog = '<start_function_call>' not in response[:50]
        
        print(f"\n用戶: {user_input}")
        print(f"類型: {'💬 對話引導' if is_dialog else '🔧 Function Call'}")
        print(f"回應: {clean_response[:150]}...")


def test_multi_turn_conversation(model, tokenizer):
    """測試多輪對話"""
    print("\n" + "=" * 70)
    print("🔄 【測試 3: 多輪對話模擬】")
    print("=" * 70)
    
    # 模擬一個完整的點餐流程
    conversation = [
        "我想點咖啡",
        "拿鐵",
        "冰的",
        "送到5樓",
    ]
    
    context = []
    
    for i, user_input in enumerate(conversation):
        print(f"\n--- 第 {i+1} 輪 ---")
        print(f"用戶: {user_input}")
        
        # 這裡簡化處理，實際應該累積 context
        response = generate_response(model, tokenizer, user_input, max_new_tokens=100, temperature=0.5)
        
        clean_response = response.replace('<start_function_call>', '[FC]').replace('<end_function_call>', '[/FC]')
        clean_response = clean_response.replace('<end_of_turn>', '').strip()
        
        print(f"助理: {clean_response[:200]}")


def test_edge_cases(model, tokenizer):
    """測試邊界情況"""
    print("\n" + "=" * 70)
    print("⚠️ 【測試 4: 邊界情況處理】")
    print("=" * 70)
    
    edge_cases = [
        ("我要一杯綠茶送到5樓", "菜單外品項"),
        ("送10杯熱拿鐵到3樓", "大量訂單"),
        ("我要冰美式加兩份濃縮", "超出限制"),
        ("ㄋㄟㄋㄟ", "口語化表達"),
        ("取消訂單", "取消請求"),
        ("改成熱的", "修改請求"),
    ]
    
    for user_input, case_type in edge_cases:
        response = generate_response(model, tokenizer, user_input, max_new_tokens=100, temperature=0.5)
        
        clean_response = response.replace('<start_function_call>', '[FC]').replace('<end_function_call>', '[/FC]')
        clean_response = clean_response.replace('<end_of_turn>', '').strip()
        
        print(f"\n[{case_type}] 用戶: {user_input}")
        print(f"回應: {clean_response[:200]}...")


def interactive_test(model, tokenizer):
    """互動式測試"""
    print("\n" + "=" * 70)
    print("🎮 【互動式測試】")
    print("輸入 'quit' 退出")
    print("=" * 70)
    
    while True:
        user_input = input("\n用戶: ").strip()
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        response = generate_response(model, tokenizer, user_input, max_new_tokens=150, temperature=0.3)
        
        # 解析 function call
        parsed = extract_function_call(response)
        
        print(f"\n原始回應: {response[:300]}")
        if parsed:
            print(f"\n解析結果: {json.dumps(parsed, indent=2, ensure_ascii=False)}")


def main():
    print("=" * 70)
    print("☕ Coffee Robot Function Calling 完整測試")
    print("=" * 70)
    
    # 載入模型
    model, tokenizer = load_model()
    
    # 執行各項測試
    fc_accuracy = test_function_calling(model, tokenizer)
    test_dialog_guidance(model, tokenizer)
    test_multi_turn_conversation(model, tokenizer)
    test_edge_cases(model, tokenizer)
    
    # 總結
    print("\n" + "=" * 70)
    print("📊 測試總結")
    print("=" * 70)
    print(f"Function Calling 準確率: {fc_accuracy:.1f}%")
    
    # 詢問是否進入互動模式
    try:
        choice = input("\n是否進入互動測試模式? (y/n): ").strip().lower()
        if choice == 'y':
            interactive_test(model, tokenizer)
    except EOFError:
        pass
    
    print("\n✅ 測試完成!")


if __name__ == "__main__":
    main()
