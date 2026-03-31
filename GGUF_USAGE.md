# 使用 coffee-robot-functiongemma GGUF 模型

## 模型已成功轉換為 GGUF 格式 ✅

你的 FunctionGemma 微調模型已經成功轉換為 GGUF 格式！

### 檔案位置
- **GGUF 模型**: `coffee-robot-functiongemma.gguf` (536.3 MB)
- **原始模型**: `coffee-robot-functiongemma/` 目錄

### 在 Ollama 中使用

#### 方法 1: 創建 Modelfile 並導入

1. 創建 Modelfile（已在本目錄中）:
```bash
FROM ./coffee-robot-functiongemma.gguf
TEMPLATE """{{ if .System }}{{ .System }}{{ end }}{{ .Prompt }}"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
```

2. 將模型導入 Ollama:
```bash
ollama create coffee-robot-functiongemma -f Modelfile
```

3. 使用模型:
```bash
ollama run coffee-robot-functiongemma "幫我送一杯熱美式到五樓"
```

#### 方法 2: 直接使用 GGUF 文件

```bash
ollama create coffee-robot-functiongemma -f coffee-robot-functiongemma.gguf
ollama run coffee-robot-functiongemma
```

### 量化版本（可選）

如果你想進一步減小模型大小，可以使用以下方法之一進行量化：

#### 選項 A: 使用 Homebrew 安裝 llama.cpp

```bash
# 安裝 llama.cpp
brew install llama.cpp

# 量化模型
llama-quantize coffee-robot-functiongemma.gguf coffee-robot-functiongemma-q4_k_m.gguf q4_k_m
```

#### 選項 B: 從源碼編譯 llama.cpp（需要 CMake）

```bash
# 安裝 CMake
brew install cmake

# 編譯 llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j

# 量化模型
./build/bin/llama-quantize ../coffee-robot-functiongemma.gguf ../coffee-robot-functiongemma-q4_k_m.gguf q4_k_m
```

#### 選項 C: 使用 Docker

```bash
docker run -v $(pwd):/models ghcr.io/ggerganov/llama.cpp:full \
  quantize /models/coffee-robot-functiongemma.gguf /models/coffee-robot-functiongemma-q4_k_m.gguf q4_k_m
```

### 量化格式說明

- **F16** (當前): 完整精度，檔案最大，質量最好
- **Q8_0**: 8-bit 量化，質量接近 F16，檔案小約 50%
- **Q4_K_M**: 4-bit 量化（推薦），質量與檔案大小平衡，檔案小約 75%
- **Q4_0**: 4-bit 量化，檔案最小，質量稍降

### 測試模型

使用 Python 測試:

```python
import ollama

response = ollama.chat(model='coffee-robot-functiongemma', messages=[
  {
    'role': 'user',
    'content': '幫我送一杯熱美式到五樓',
  },
])

print(response['message']['content'])
```

### 模型規格

- **架構**: Gemma3ForCausalLM
- **參數量**: ~270M
- **層數**: 18 layers
- **詞彙表大小**: 262,144 tokens
- **上下文長度**: 32,768 tokens
- **訓練資料**: 咖啡機器人外送任務 Function Calling

### 效能建議

1. **FP16 版本** (536.3 MB): 最佳質量，適合有足夠 RAM 的系統
2. **Q4_K_M 版本** (量化後約 150 MB): 推薦用於生產環境，質量損失極小
3. **Q8_0 版本** (約 280 MB): 質量與大小的折衷方案

### 進一步優化

如果模型在 Ollama 中運行良好，你可以：

1. 上傳到 Hugging Face Hub 分享
2. 創建 Ollama 模型庫並發布
3. 整合到你的應用程式中

需要幫助？請參考 Ollama 文檔: https://github.com/ollama/ollama
