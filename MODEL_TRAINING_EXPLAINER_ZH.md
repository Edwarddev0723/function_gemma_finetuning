# 模型訓練說明

這份文件是給非 AI 專業背景的主管閱讀，用來說明這顆咖啡外送模型是怎麼訓練出來的、為什麼要這樣訓練、以及目前做到什麼程度。

## 一句話先講完

這顆模型不是從零開始訓練，而是站在 Google 的 `FunctionGemma 270M` 基礎模型上，先教它把咖啡訂單轉成標準化的 function call，再進一步教它遇到資訊不完整時會追問、遇到不支援的需求時會拒答，最後變成一顆比較像真實客服或機器人助理的模型。

## 這顆模型到底在學什麼

如果用最簡單的方式說，這顆模型的任務不是「聊天很像人」，而是「把使用者的自然語言，穩定地轉成系統可執行的任務格式」。

例如：

- 使用者說：「幫我送一杯熱美式到五樓」
- 模型要學會輸出：`create_coffee_robot_mission({"baseDrink":"americano","floor":5,"temperature":"hot","quantity":1,"addons":[]})`

也就是說，模型學的重點有三個：

1. 聽懂口語化訂單。
2. 把資訊整理成固定欄位。
3. 在資訊不足或需求不合理時，做出正確的對話反應。

## 用主管聽得懂的比喻

可以把這次訓練想成在培訓一位新的咖啡櫃台人員。

- 基礎模型：像是已經會中文、會對話、也知道怎麼呼叫工具的新人。
- 第一階段微調：教他看懂我們公司的訂單格式，知道哪些欄位一定要填。
- 第二階段續訓：教他面對真實客人時，不只會照表操課，還知道什麼時候該追問、什麼時候該拒絕、怎麼回答比較像服務流程。

## 訓練流程總覽

整個流程可以分成兩個主要階段。

### 第一階段：先把模型教會「正確呼叫工具」

目標很單純：只要使用者資訊足夠完整，模型就直接輸出正確的 function call。

流程如下：

1. 以 `google/functiongemma-270m-it` 當基礎模型。
2. 用本地 LLM 先大量生成咖啡訂單範例。
3. 把這些範例整理成 FunctionGemma 能讀懂的 chat + tool 格式。
4. 用監督式微調（SFT, Supervised Fine-Tuning）訓練模型。

### 第二階段：讓模型更接近真實服務場景

第一階段只解決「資訊完整時怎麼叫工具」，但真實世界還會遇到很多狀況：

- 使用者資訊講不完整。
- 使用者點的是菜單外商品。
- 使用者臨時取消。
- 使用者會用不同說法講同一種品項，例如鮮奶、鮮乳、牛奶、ㄋㄟㄋㄟ。

所以第二階段做的事情，是把更多真實互動情境放進資料，再用較輕量的 LoRA 方式續訓，讓模型同時學會：

- 資訊完整時，直接輸出 function call。
- 資訊不完整時，先追問，不要亂下單。
- 菜單外需求時，明確拒答或引導改單。

## 第一階段：資料怎麼做

第一階段資料是由 [`dataset_prepare.py`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/dataset_prepare.py) 生成。

重點做法如下：

- 透過本地 Ollama API 呼叫 `gpt-oss:20b` 產生合成資料。
- 系統提示詞會明確限制商業規則，例如飲品、樓層、溫度、加購項目、數量。
- 腳本會先做驗證，確保輸出的欄位值合法。
- 腳本也會去除重複輸入，避免資料太多重複句型。
- 最後把資料切成 train / eval。

目前 repo 內實際的第一版資料檔 [`ollama_mcp_dataset.jsonl`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/ollama_mcp_dataset.jsonl) 統計如下：

| 項目 | 數量 |
|---|---:|
| 總筆數 | 699 |
| 訓練集 | 625 |
| 驗證集 | 74 |
| 全部是否為 function calling 樣本 | 是 |

第一版資料的特性：

- 全部都是「資訊足夠完整」的訂單。
- 全部都要求模型直接輸出工具呼叫。
- 平均使用者輸入長度約 14.4 個字。

這一版資料很適合拿來建立模型的基本功。

## 第一階段：怎麼微調

第一階段的訓練主體在 [`Finetune_FunctionGemma_Coffee_Robot_MCP.ipynb`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/Finetune_FunctionGemma_Coffee_Robot_MCP.ipynb)。

訓練方式是監督式微調（SFT），意思是拿「標準答案」直接教模型學。

簡單講，就是：

- 輸入：使用者的咖啡需求 + 工具定義。
- 輸出：正確的 function call。
- 模型反覆比對自己的輸出和標準答案差多少，再逐步修正。

主要訓練設定如下：

| 項目 | 設定 |
|---|---|
| 基礎模型 | `google/functiongemma-270m-it` |
| 訓練方式 | SFT |
| 訓練輪數 | 3 epochs |
| 單卡 batch size | 1 |
| 梯度累積 | 16 |
| 有效 batch size | 16 |
| 學習率 | `1e-5` |
| Optimizer | `adamw_torch` |
| Gradient checkpointing | 開啟 |
| Loss 計算方式 | `completion_only_loss=True` |

對主管來說，最重要的理解是：

- 這不是把整顆模型全部重學一遍。
- 而是在既有模型上，集中教它「怎麼正確回答這個任務」。
- `completion_only_loss=True` 的意思是，訓練時主要看模型產出的答案好不好，不是重新學問題本身。

## 訓練環境

根據專案 README 與訓練設定，第一階段配置是針對 Apple Silicon `M4 Pro 48GB` 做優化。

重點不是一定只能在這台機器上跑，而是整套參數明顯有考慮本地訓練成本，例如：

- batch size 刻意設小。
- 用梯度累積補回有效 batch size。
- 開啟 gradient checkpointing 降低記憶體壓力。
- 在 Apple Silicon 上使用較穩定的 MPS / `float32` 設定。

## 第二階段：資料怎麼擴充

第二階段資料準備邏輯在 [`prepare_merged_dataset_v2.py`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/prepare_merged_dataset_v2.py)。

這一版資料相較第一階段有明顯擴充，重點不是只有更多，而是情境更完整。

它做了幾件事：

1. 保留原本的 function calling 樣本。
2. 補強鮮奶相關說法，例如鮮奶、鮮乳、牛奶、ㄋㄟㄋㄟ。
3. 加入菜單外品項與取消訂單等拒絕回應。
4. 明確加入「資訊不完整時只能追問、不能亂輸出 function call」的對話樣本。
5. 加入更多完整資訊下的標準 function calling 樣本。
6. 合併公開咖啡對話資料集 `renhehuang/coffee-order-zhtw` 的部分對話資料。

目前 repo 內第二版合併資料 [`merged_coffee_dataset_v2.jsonl`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/merged_coffee_dataset_v2.jsonl) 統計如下：

| 項目 | 數量 |
|---|---:|
| 總筆數 | 4,501 |
| 訓練集 | 4,051 |
| 驗證集 | 450 |
| Function Calling 類樣本 | 1,149 |
| 對話 / 拒答類樣本 | 3,352 |
| 平均使用者輸入長度 | 約 18.1 個字 |

這個數字很重要，因為它代表第二階段不再只教模型「下單」，而是更偏向教它「怎麼正確處理服務流程」。

## 第二階段：怎麼續訓

第二階段的主腳本是 [`continue_finetune_with_coffee_order.py`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/continue_finetune_with_coffee_order.py)。

這一階段不是重新從頭訓練，而是接在第一階段模型 [`coffee-robot-functiongemma`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/coffee-robot-functiongemma) 之後再做續訓。

續訓採用 LoRA，主管可以把它理解成：

- 不大改整顆模型。
- 只在少數關鍵位置加上小型可訓練模組。
- 好處是速度比較快、成本比較低、也比較不容易把原本能力弄壞。

第二階段主要設定如下：

| 項目 | 設定 |
|---|---|
| 續訓起點 | `./coffee-robot-functiongemma` |
| 輸入資料 | `merged_coffee_dataset_v2.jsonl` |
| 訓練方式 | SFT + LoRA |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| LoRA target modules | `q_proj`, `v_proj` |
| 訓練輪數 | 5 epochs |
| batch size | 2 |
| 梯度累積 | 8 |
| 學習率 | `5e-5` |
| 最大長度 | 512 |
| 最佳模型選擇 | 以 `eval_loss` 為準 |

這一段的意義是：

- 第一階段學的是「格式」。
- 第二階段學的是「流程與情境」。

目前第二階段測試腳本預設指向的最終模型是 [`coffee-robot-continued-v2-final`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/coffee-robot-continued-v2-final)。

## 模型最後學到的行為

如果用業務語言整理，最終模型大概學會三種反應模式。

### 1. 資訊完整：直接建立任務

例如：

- 「我要一杯冰拿鐵，送到 5 樓」

模型應該直接輸出工具呼叫，不多說廢話。

### 2. 資訊不完整：先追問

例如：

- 「我要咖啡」
- 「給我一杯拿鐵」

模型不應該亂猜樓層或溫度，而是先問缺的欄位。

### 3. 不支援需求：拒答或引導改單

例如：

- 點綠茶、奶茶、可可等菜單外品項。
- 超出規則的加購要求。
- 取消訂單。

這對實際上線很重要，因為一個會「亂幫你下單」的模型，比一個會「先問清楚」的模型更危險。

## 目前可以怎麼說成果

以 repo 內現成評估結果來看，第一階段微調後，模型在驗證集上「有沒有選到正確 function 名稱」這個指標，從 95.95% 提升到 100%。

對應檔案如下：

- [`eval_base_model.json`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/eval_base_model.json)
- [`eval_trained_model.json`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/eval_trained_model.json)

但這裡有一個很重要的說明，建議在對主管報告時一起講清楚：

- 這份現成評估主要檢查的是「模型有沒有呼叫到正確 function」。
- 它不是完整檢查每個欄位值是否都正確。
- 也就是說，這個數字可以證明第一階段格式能力有進步，但不能單靠它就宣稱所有參數都已經百分之百正確。

這不是壞事，而是評估口徑要講清楚。

如果要更完整地對外說明，建議說：

「模型已經很穩定地學會該不該呼叫工具；至於每個欄位值是否完全正確，還需要更細的欄位級驗證。」

## 為什麼這樣的訓練設計是合理的

這次設計合理的地方在於，它沒有一開始就追求所有情境一起學，而是分兩步走。

### 先把基本動作練對

先用單純、乾淨、格式一致的資料，建立 function calling 基本功。

這樣做的好處是：

- 模型比較快收斂。
- 比較容易知道問題出在哪。
- 不會一開始就被太多複雜對話干擾。

### 再補真實世界的例外情境

等基本動作穩了，再加入更多對話、拒答、模糊輸入與同義詞。

這樣做的好處是：

- 模型不只會做「理想情境」。
- 也比較接近真正上線後會遇到的使用者行為。

## 目前限制

這份說明也應該保留幾個現階段限制，避免過度承諾。

1. 現成量化評估主要還是 function 名稱層級，不是完整欄位正確率。
2. 第二階段雖然有測試腳本，但比較偏行為驗證與 spot check，不算完整 benchmark。
3. 目前場景仍聚焦在單一咖啡任務，泛化到更多工具或更複雜流程，還需要更多資料。
4. 很多訓練資料是合成資料，優點是快，缺點是可能和真實使用者語氣有落差。

## 下一步建議

如果要把這個專案往更正式的產品化方向走，最值得補的不是再盲目加資料，而是補評估。

優先建議：

1. 建立欄位級評估：逐一檢查 `baseDrink`、`floor`、`temperature`、`quantity`、`addons`。
2. 區分三種成功率：直接下單、追問補資訊、拒答。
3. 增加真實使用者語料，而不只依賴合成資料。
4. 加入線上或模擬流量回測，確認模型在長尾語句下的穩定性。

## 主管簡報可直接使用的 60 秒版本

這顆模型是建立在 Google 的 FunctionGemma 270M 上，不是從零開始做。我們先用一批結構化的咖啡訂單資料，教模型把自然語言轉成系統可執行的 function call，先把基本動作學穩。之後再加入更貼近真實場景的資料，例如資訊不完整、菜單外品項、取消訂單、不同說法的鮮奶需求，讓模型不只會下單，也知道何時要追問、何時要拒答。這樣做的好處是，模型比較像一個能落地的服務助理，而不是只會照格式回答的 demo。現階段我們已經看到它在「是否正確呼叫工具」上有明顯改善，但如果要更正式對外報告，下一步需要補的是欄位級精準度評估。

## 主要檔案對照

| 用途 | 檔案 |
|---|---|
| 第一版資料生成 | [`dataset_prepare.py`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/dataset_prepare.py) |
| 第一版資料集 | [`ollama_mcp_dataset.jsonl`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/ollama_mcp_dataset.jsonl) |
| 第一階段訓練 Notebook | [`Finetune_FunctionGemma_Coffee_Robot_MCP.ipynb`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/Finetune_FunctionGemma_Coffee_Robot_MCP.ipynb) |
| 第二版資料合併 | [`prepare_merged_dataset_v2.py`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/prepare_merged_dataset_v2.py) |
| 第二版合併資料 | [`merged_coffee_dataset_v2.jsonl`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/merged_coffee_dataset_v2.jsonl) |
| 第二階段續訓 | [`continue_finetune_with_coffee_order.py`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/continue_finetune_with_coffee_order.py) |
| 第二階段最終模型 | [`coffee-robot-continued-v2-final`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/coffee-robot-continued-v2-final) |
| 第一階段評估結果 | [`eval_base_model.json`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/eval_base_model.json), [`eval_trained_model.json`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/eval_trained_model.json) |
| 第二階段行為測試 | [`test_model_v2.py`](/Users/edwardhuang/Documents/GitHub/function_gemma_finetuning/test_model_v2.py) |
