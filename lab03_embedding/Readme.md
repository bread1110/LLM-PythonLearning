# embedding.py 使用說明

## 簡介
`embedding.py` 展示如何使用 Azure OpenAI API 將文字資料進行 embedding 向量化、計算相似度，並以 t-SNE 降維進行視覺化呈現。本程式適用於文字資料分析、相似度搜尋、及視覺化探索用途。

## 功能概述
- 從網路取得書籍資料集
- 使用 Azure OpenAI 將文本轉為 embedding 向量
- 多執行緒加速 embedding 計算
- 計算餘弦相似度
- 使用 t-SNE 進行降維視覺化
- 支援輸入查詢並顯示最相似結果

## 主要依賴套件
- `openai`
- `python-dotenv`
- `pandas`
- `numpy`
- `scikit-learn` (t-SNE)
- `matplotlib`
- `threading`
- `requests`

## 環境變數設定（.env）
```ini
AOAI_KEY=您的 Azure OpenAI API 金鑰
AOAI_URL=您的 Azure OpenAI API 端點
AOAI_MODEL_VERSION=GPT 模型版本
EMBEDDING_API_KEY=您的 Azure OpenAI Embedding 金鑰
EMBEDDING_URL=您的 Azure OpenAI Embedding 端點
EMBEDDING_MODEL=使用的 Embedding 模型名稱
```

## 安裝指令
```bash
pip install openai python-dotenv pandas numpy scikit-learn matplotlib requests
```

## 使用方法
1. 確保 `.env` 設定完成。
2. 執行 `embedding.py`，流程將自動：
   - 從網路下載書籍資料集。
   - 使用多執行緒計算所有文本 embedding。
   - 利用 t-SNE 將結果降維並繪製視覺化圖表。
   - 示範以一組查詢文字找出最相似的 5 筆結果，並於圖中標示出來。

## 範例輸出
### 視覺化圖表說明
- 藍色點：其他文件
- 橘色點：與查詢結果相似度最高的文件
- 紅色星號：查詢點

### 範例終端輸出
```bash
最相似的文件：
標題：xxxx 相似度：0.98
標題：xxxx 相似度：0.96
...
```

## 可修改範例
- 可嘗試將相似度計算方式改為 L2 距離 (Euclidean Distance)。

## 技術架構
- Azure OpenAI API：文字 embedding 與 GPT 回應
- t-SNE：降維視覺化
- Python 多執行緒：平行化 embedding 處理
- `matplotlib`：視覺化繪圖
- `pandas`：資料處理

## 注意事項
- 請確認 Azure OpenAI 的 API 金鑰與配額足夠。
- 若執行時遇到網路或 API 回傳錯誤，系統會自動重試 2 次。

