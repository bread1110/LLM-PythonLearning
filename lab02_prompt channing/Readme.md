# Azure OpenAI 進階應用實作

這個專案展示了如何使用 Azure OpenAI API 進行進階的自然語言處理任務，包括數學運算、股票分析和網路搜尋。專案包含以下腳本：

- `calculator.py`：展示數學表達式轉換和計算。
- `stock_api.py`：展示股票資訊提取和分析。
- `web_search.py`：展示網路搜尋和資訊整合。

## 簡介

本專案旨在展示 Azure OpenAI API 在實際應用場景中的進階用法。通過這些腳本，您可以學習如何將 AI 應用於數學計算、金融分析和資訊搜尋等實際任務中。建議按照以下順序閱讀和執行腳本：`calculator.py` -> `stock_api.py` -> `web_search.py`。

## 安裝指南

### 1. 設置環境變數
- 修改 `.env` 文件，並加入以下內容：
    AOAI_KEY=您的 Azure OpenAI API 金鑰
    AOAI_URL=您的 Azure OpenAI API 端點
    AOAI_MODEL_VERSION=您的模型版本

### 2. 啟動虛擬環境後並安裝相依套件
- 安裝必要的 Python 庫：

> pip install openai python-dotenv numexpr requests tavily-api

注意事項:
請確保您的 Azure OpenAI API、Tavily API 配額足夠。
腳本中的示例可能需要根據您的模型版本進行調整。

web_search.py
功能：
- 從用戶查詢中提取關鍵搜尋詞
- 使用多執行緒進行並行網路搜尋
- 自動摘要和整合搜尋結果
- 生成綜合分析報告

示例：
```python
輸入：關於 Nvidia 的股價表現和 AI 發展影響
輸出：整合多方資訊的完整分析報告
```

calculator.py
功能：

- 將中文數學題目轉換為標準數學表達式
- 使用 numexpr 進行高精度計算
- 支援複雜的運算順序控制
- 提供詳細的計算過程說明

示例：
```python
輸入：請計算 64 乘以 2 再扣掉 8，以上結果再除100後，再指數 1.234
輸出：數學表達式和精確計算結果
```

stock_api.py
功能：
- 將中文股票查詢轉換為結構化資訊
- 自動識別股票代碼和日期
- 從台灣證券交易所獲取即時數據
- 生成專業的股票分析報告

示例：
```python
輸入：請問2025年的2月27號 台積電的股價表現如何?
輸出：包含股價、交易量等詳細分析的專業報告
```

## 技術架構
- Azure OpenAI：自然語言處理
- numexpr：數學運算引擎
- Tavily API：網路搜尋服務
- 台灣證券交易所 API：股票資料來源
- Python 多執行緒：提升處理效能