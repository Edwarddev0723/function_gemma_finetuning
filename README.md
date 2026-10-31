# FunctionGemma Coffee Robot Fine-Tuning

**繁體中文工具呼叫｜合成資料｜Full-parameter SFT｜LoRA 續訓｜評估與錯誤分析**

以 `google/functiongemma-270m-it` 為基礎，探索如何將繁體中文咖啡外送需求轉換為結構化工具呼叫，並透過後續資料擴充加入追問、拒答與品項同義詞等服務情境。

本專案可檢查的內容包括資料生成與合併、初版 SFT、LoRA 續訓、推論檢查，以及公開模型產物。作品集重點是小型語言模型的任務適配與評估流程；完整參數正確率、多輪任務成功率與部署效能仍待驗證。

## 任務與實作範圍

主要工具為 `create_coffee_robot_mission`，處理飲品、樓層、溫度、數量與加料。以下是目標介面示意，並非本次模型實測輸出：

```json
{
  "name": "create_coffee_robot_mission",
  "arguments": {
    "baseDrink": "americano",
    "floor": 5,
    "temperature": "hot",
    "quantity": 1,
    "addons": []
  }
}
```

| 面向 | Repository 中的實作 |
|---|---|
| 合成資料 | 透過 Ollama 呼叫本地 LLM，產生使用者需求與工具參數，進行基本值域檢查與輸入去重 |
| 初版模型適配 | 將 chat／tools 轉為 prompt-completion，使用 TRL 執行 full-parameter SFT |
| 情境擴充 | 合併咖啡對話資料，加入鮮奶同義詞、資訊不足、菜單外需求與取消情境 |
| 續訓 | 使用 LoRA 更新 `q_proj`／`v_proj`，依 validation loss 選取模型，再合併 adapter |
| 評估 | 保存初版 base／SFT 的逐筆解析結果，另提供行為檢查與推論腳本 |
| 模型交付 | Hugging Face 有公開 Safetensors 與 GGUF 檔案；另有 Ollama 使用文件 |

Repository 主要涵蓋資料與模型流程；目前未見完整 MCP Server、機器人 API 執行與端到端配送驗收的可核對實作／紀錄。

## 資料與切分

以下數量由已提交 JSONL 重新統計，train／eval 依各檔案的 `metadata` 欄位。Function Calling 樣本以最後一則訊息包含非空 `tool_calls` 計數。

| 資料檔 | 總數 | Train | Eval | Function Calling | 其他對話 |
|---|---:|---:|---:|---:|---:|
| [ollama_mcp_dataset.jsonl][data-initial] | 699 | 625 | 74 | 699 | 0 |
| [merged_coffee_dataset.jsonl][data-v1] | 3,637 | 3,274 | 363 | 2,161 | 1,476 |
| [merged_coffee_dataset_v2.jsonl][data-v2] | 4,501 | 4,051 | 450 | 1,149 | 3,352 |

初版生成腳本設定的目標數量與實際保存筆數不同，因此以檔案中的 **699 筆**為準。初版全部是應呼叫同一工具的正例；v2 的「其他對話」是結構分類，不代表每筆都已經人工確認為正確的追問或拒答。

資料準備程式會使用 `renhehuang/coffee-order-zhtw`；未固定該來源的 dataset revision，重新執行不保證產生相同資料。

### 已確認的切分限制

| 檢查 | v1 | v2 |
|---|---:|---:|
| Eval 中與同版 train 完全重複的正規化樣本 | 3／363 | 50／450 |
| 初版 625 筆 train 被重新分入該版 eval | 58 | 73 |
| 初版 74 筆 eval 被重新分入該版 train | 68 | 66 |

重複檢查會移除 `metadata` 與隨機 call ID、將 arguments JSON 字串解析為物件、排序物件 keys，保留訊息順序、工具定義與其他內容。v2 的 50 筆重複 eval 對應 31 種正規化樣本；初版 74 筆 eval 在相同檢查下未發現與初版 train 完全重複的樣本，但這不排除相似模板重疊。

跨階段對齊以原始 call ID 檢查。若續訓從初版 checkpoint 開始，重新切出的 eval 會包含先前訓練見過的樣本；原始 74 筆也不能繼續當作續訓模型的 held-out test set。

## 訓練設計

| 設定 | 初版 SFT Notebook | LoRA 續訓腳本預設 |
|---|---|---|
| 起點 | `google/functiongemma-270m-it` | 本地 `./coffee-robot-functiongemma` |
| 更新方式 | Full-parameter SFT，未使用 PEFT | LoRA：`r=8`、`alpha=16`、`dropout=0.1` |
| LoRA targets | — | `q_proj`、`v_proj` |
| Epochs | 3 | 5 |
| Batch／gradient accumulation | 1 × 16 | 2 × 8 |
| Learning rate | `1e-5` | `5e-5` |
| Max length | 保存值 436，依樣本動態估算 | 512 |
| Gradient checkpointing | 開啟 | 關閉 |
| 模型選擇 | 訓練結束後保存 | 依 epoch eval，以 `eval_loss` 選最佳模型 |
| 預設資料 | 初版 699 筆 | v2 合併資料 |

來源：[初版 Notebook][notebook]、[續訓腳本][continue-script]。

初版明確設定 `completion_only_loss=True`。續訓資料同時保留 `prompt`／`completion`；依已記錄的 [TRL 0.25.1][trl-v0251]／[0.27.0][trl-v0270] 版本實作，此資料路徑也會使用 completion-only loss。這是程式與版本化 source 的靜態核對，本次未執行 collator 驗證實際 token mask。

初版採用 FP32／MPS 設定。續訓則依 MPS、CUDA 或 CPU 選擇精度與 optimizer；目前沒有一致硬體條件下的訓練速度、峰值記憶體或成本比較。

Notebook 保存 **120／120 steps、3／3 epochs** 的歷史輸出；step 100 記錄 validation loss `0.002477`。該 loss 是 teacher-forced token 診斷，且不是最終 step 120 checkpoint 的獨立任務評估。本次沒有重跑訓練。

## 可核對的評估結果

[eval_base_model.json][eval-base] 與 [eval_trained_model.json][eval-trained] 各有 74 筆，逐筆 user 與 target 相同。重新計算 function-name list exact match，結果與保存的 `correct_names` 一致：

| 指標 | Base | 初版 SFT |
|---|---:|---:|
| Function-name list exact match | 71／74（95.95%） | 74／74（100.00%） |
| 完整 arguments／schema 正確率 | 未建立可靠結果 | 未建立可靠結果 |
| 追問／拒答／不呼叫工具判斷 | 此集合未涵蓋 | 此集合未涵蓋 |
| 獨立 test set 任務成功率 | 未驗證 | 未驗證 |

這 74 筆均要求單一 `create_coffee_robot_mission`，也是訓練期間使用的 validation set。這組數字反映保存產物的名稱匹配，無法支持完整 Function Calling accuracy、call／no-call 判斷或後續 LoRA／v2 的改善。

原 README 的「約 10% → 95%+」缺乏這組產物支持，因此不沿用。

### 參數格式與 parser 不一致

Notebook evaluator 只解析 `key:<escape>value<escape>` 形式。但已保存的訓練 completion 與四個微調後生成範例，呈現外層 function braces 包住 JSON 字串的形式，例如：

```text
call:create_coffee_robot_mission{ {"baseDrink": "latte", "floor": 11, ...}}
```

資料中的 `arguments` 以 JSON 字串傳入 chat template，與 parser 預期格式不一致。保存的 trained 評估中，74 筆解析後 arguments 都是 `{}`；然而四個原始生成範例可見具體欄位，因此不能把空解析結果直接解讀為模型沒有產生參數。

74 筆評估 JSON 未保存完整 raw generations，目前無法可靠恢復逐欄位或完整呼叫正確率。

## 公開模型產物

[Hugging Face 檔案清單][hub-files] 有公開權重與 GGUF。核對 revision 為 `e227c5f89f2f24cfccbc890195ade105b118ea20`；API metadata 列出 `model.safetensors`，並將其中參數標為 F32，另有 `coffee-robot-functiongemma.gguf`。

本次只核對檔案清單與 metadata，沒有下載或載入權重、檢查 GGUF 內容、執行生成或量測效能。這些產物尚未與每份 eval JSON 建立 checkpoint hash 對應。

GitHub 的續訓目錄主要保存 tokenizer／config／Model Card，未包含完整權重。續訓腳本預設輸出仍為 `coffee-robot-continued[-final]`；缺少實際 CLI invocation，不能僅依 `continued-v2-final` 目錄名稱認定其完整訓練 lineage。

[Hub Model Card][hub-card] 的範例使用 `send_coffee`，與 GitHub 訓練資料的 `create_coffee_robot_mission` 不同。使用前需統一工具名稱、schema 與 parser；Model Card 的 accuracy 也應依上述名稱匹配範圍理解。

## Repository 導覽

| 檔案 | 用途 |
|---|---|
| [dataset_prepare.py][prepare-initial] | 初版 Ollama 合成資料、基本驗證與切分 |
| [Finetune_FunctionGemma_Coffee_Robot_MCP.ipynb][notebook] | 初版 full-parameter SFT、歷史輸出與名稱匹配評估 |
| [prepare_merged_dataset.py][prepare-v1] | v1 對話合併與工具標籤抽取 |
| [prepare_merged_dataset_v2.py][prepare-v2] | v2 模板、追問／拒答與資料合併 |
| [continue_finetune_with_coffee_order.py][continue-script] | LoRA 續訓、adapter 合併與可選 GGUF 轉換 |
| [test_model_inference.py][test-v1]、[test_model_v2.py][test-v2] | 啟發式推論檢查 |
| [GGUF_USAGE.md][gguf-doc]、[Modelfile][modelfile] | GGUF／Ollama 使用說明 |
| [MODEL_TRAINING_EXPLAINER_ZH.md][explainer] | 訓練流程說明；部分成果敘述仍需依本 README 的評估範圍解讀 |

## 執行前準備

以下為依初版 Notebook 整理的環境準備示例，本次未安裝或執行：

```bash
git clone https://github.com/Edwarddev0723/function_gemma_finetuning.git
cd function_gemma_finetuning

uv venv --python 3.11
source .venv/bin/activate
uv pip install torch transformers==4.57.1 trl==0.25.1 datasets==4.4.1 matplotlib pandas tqdm requests jupyter
```

續訓另使用 `peft`、`accelerate` 等依賴。續訓 v2 Model Card 記錄 TRL 0.27.0／Transformers 4.57.6／PyTorch 2.9.1／Datasets 4.5.0，與初版環境不同；目前沒有完整 lockfile，不保證上述初版環境可直接重現所有入口。

執行前需準備基礎模型存取權或有效本地 checkpoint，並先處理資料切分與序列化問題。資料準備腳本會寫出同名 JSONL；訓練與轉換會產生模型及其他輸出，請明確管理資料和輸出路徑。Notebook 的可選 Hub 上傳段落須依實際發布意圖執行。

## 已知限制與未驗證項目

| 面向 | 目前限制 |
|---|---|
| 資料品質 | v1 的訂單抽取掃描整段 conversation，會受選單、舊訂單與 assistant 文字干擾；保存資料可見最終需求為熱燕麥奶拿鐵、標籤卻為冰美式的案例。v2 仍有 52 筆 Function Calling 樣本的 user 文字出現「熱冰／冰熱」矛盾字串。合成資料尚未完成系統性人工審查。 |
| 切分與洩漏 | 合併後重新切分造成跨 train／eval 重複，以及跨訓練階段的樣本交叉；目前缺少獨立、固定且未參與調整的 test set。 |
| 格式與評估 | Training target／generation 與 parser 格式不一致；74 筆評估未保存 raw generations。完整參數、schema、執行成功與 call／no-call 結果尚未建立。 |
| 行為測試 | v1 的 6 個工具案例僅檢查飲品與樓層；其多輪 helper 未累積對話歷史。v2 的 18 個案例以 marker、飲品字串及關鍵詞判斷，未完整驗證欄位；部分對話判定甚至可能接受空輸出。未見保存的通過報告或 GitHub Actions run。 |
| 訓練與 lineage | 初版長度估算取字元最長樣本，不能保證涵蓋全部 token lengths；截斷、EOS／停止行為與實際 loss mask 仍需實測。資料、環境、checkpoint 與評估缺少完整 hash／revision 對應；續訓 Model Cards 仍有 base model 為 `None` 的 placeholder。 |
| 量化與部署 | `quantize_model.py` 只讀取 GGUF 並列印替代做法，未實際寫出 Q4_K_M 權重。GGUF 轉換、量化前後品質、Ollama 工具相容性、延遲與記憶體均未由本次執行驗證；現有 Modelfile 也未證明與訓練 chat template 等價。 |
| 工具契約與泛化 | Schema、生成驗證與測試對值域的約束不完全一致；主要只涵蓋單一咖啡工具，尚無跨工具、真實使用者或機器人端到端驗收結果。 |

## 後續驗證方向

優先統一 arguments serialization、chat template 與 parser，保存 raw generations、模型 revision 及 generation settings；再清理重複與矛盾標籤，固定所有訓練階段共用的資料邊界。

接著以獨立 test set 比較 base、初版 SFT 與 LoRA 續訓，分別量測 function name、逐欄位、完整呼叫、追問、拒答與真正多輪訂單狀態。部署比較需另記錄硬體、軟體版本、量化格式、延遲、記憶體與任務品質。

以上均為後續工作，尚未宣稱完成。

## 核對基準與授權

本 README 依據 GitHub commit [`f01922cb`][source]，核對日期為 **2026-10-28**。核對包含程式與保存輸出檢閱、JSONL／評估 JSON 的記憶體內統計，以及公開模型 metadata；未執行模型載入、推論、訓練、量化或測試腳本。

原 README 宣告程式碼採 MIT，但目前快照未見獨立 `LICENSE` 檔，授權文件仍待補齊。公開模型的 Model Card 標示 Gemma license；使用模型與資料前，仍需核對各自來源條款。

[source]: https://github.com/Edwarddev0723/function_gemma_finetuning/tree/f01922cbc8baeccb34ec3caf82cb475efade440a
[notebook]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/Finetune_FunctionGemma_Coffee_Robot_MCP.ipynb
[data-initial]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/ollama_mcp_dataset.jsonl
[data-v1]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/merged_coffee_dataset.jsonl
[data-v2]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/merged_coffee_dataset_v2.jsonl
[eval-base]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/eval_base_model.json
[eval-trained]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/eval_trained_model.json
[prepare-initial]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/dataset_prepare.py
[prepare-v1]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/prepare_merged_dataset.py
[prepare-v2]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/prepare_merged_dataset_v2.py
[continue-script]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/continue_finetune_with_coffee_order.py
[test-v1]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/test_model_inference.py
[test-v2]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/test_model_v2.py
[gguf-doc]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/GGUF_USAGE.md
[modelfile]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/Modelfile
[explainer]: https://github.com/Edwarddev0723/function_gemma_finetuning/blob/f01922cbc8baeccb34ec3caf82cb475efade440a/MODEL_TRAINING_EXPLAINER_ZH.md
[hub-card]: https://huggingface.co/renhehuang/functiongemma-270m-it-coffee-robot-mcp/blob/e227c5f89f2f24cfccbc890195ade105b118ea20/README.md
[hub-files]: https://huggingface.co/renhehuang/functiongemma-270m-it-coffee-robot-mcp/tree/e227c5f89f2f24cfccbc890195ade105b118ea20
[trl-v0251]: https://github.com/huggingface/trl/blob/v0.25.1/trl/trainer/sft_trainer.py
[trl-v0270]: https://github.com/huggingface/trl/blob/v0.27.0/trl/trainer/sft_trainer.py
