# Azure OpenAI 實作示例

這個專案展示了如何使用 Azure OpenAI API 進行多種自然語言處理任務，包括基本對話、情感分析和生成結構化數據。專案包含以下腳本：

- `basic.py`：展示基本的對話和任務執行。
- `few_shot.py`：展示 Few-Shot Learning 在情感分析中的應用。
- `json_format.py`：展示如何要求模型以 JSON 格式回應。
- `Zero-shot Classification.py`：展示 Zero-Shot 分類（延伸資料）。
- `One-shot Entity Extraction.py`：展示 One-Shot 實體提取（延伸資料）。
- `Two-shot Entity Extraction.py`：展示 Two-Shot 實體提取（延伸資料）。

## 簡介

本專案旨在幫助使用者理解和實作 Azure OpenAI API 的不同功能。通過這些腳本，您可以學習如何與 API 進行交互，並應用於實際的 NLP 任務中。建議按照以下順序閱讀和執行腳本：`basic.py` -> `few_shot.py` -> `json_format.py`。

## 安裝指南

### 1. 設置環境變數
- 修改 `.env` 文件，並加入以下內容：
    AOAI_KEY=您的 Azure OpenAI API 金鑰
    AOAI_URL=您的 Azure OpenAI API 端點
    AOAI_MODEL_VERSION=您的模型版本

### 2. 啟動虛擬環境後並安裝相依套件
- 安裝必要的 Python 庫：

> pip install openai python-dotenv

注意事項:
請確保您的 Azure OpenAI API 配額足夠。
腳本中的示例可能需要根據您的模型版本進行調整。

basic.py
功能：

展示如何使用 Azure OpenAI API 進行基本的對話和任務執行。
包括描述事實（如關於猴子的事實）、回答問題（如草的顏色）、總結故事和分析病例。

few_shot.py
功能：

展示 Few-Shot Learning 在情感分析中的應用。
通過提供多個情感分析範例（如積極和消極的推文），讓模型學習如何判斷文字的情感傾向。

json_format.py
功能：

展示如何要求模型以 JSON 格式回應。
生成結構化數據，如用戶資料（姓名、年齡等）和書籍資料（標題、類型等）。

以下.py檔為延伸資料，有餘力的學員可以自行參閱並探索：

Zero-shot Classification.py
功能：展示 Zero-Shot 分類，無需訓練數據即可判斷投訴類別（如信用卡、抵押貸款等）。
示例：分析一段消費者投訴並分類。

One-shot Entity Extraction.py
功能：展示 One-Shot 實體提取，通過一個示例提取個人身份資訊（PII），如姓名、電話號碼等。
示例：從一段文字中提取金額、地點等資訊。

Two-shot Entity Extraction.py
功能：展示 Two-Shot 實體提取，通過兩個示例提取更複雜的個人身份資訊。
示例：提取信用卡號、地址等資訊。