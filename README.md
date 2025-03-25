# LLM-PythonLearning

本專案目標為學習如何使用 **Python** 程式語言串接 **Azure OpenAI (AOAI)**，並透過各種實作範例深入理解 **Prompt Engineering**、**Chain Prompt**、**Embedding**、**Conversation** 及 **RAG（Retrieval Augmented Generation）** 的應用。

---

## 📚 專案內容總覽

### 🔸 lab00. 環境建置
- 介紹 Python 與虛擬環境 (Conda / venv) 的概念。
- 詳細步驟說明如何安裝 Anaconda 及 Python，並建立、啟動與移除虛擬環境。
- 提供 Conda 指令整理，方便快速上手。
- 說明在 Windows、macOS、Linux 環境中分別如何建立虛擬環境並執行 Jupyter Notebook。

---

### 🔸 lab01. Prompt Engineering、Few-shot、JSON-format
- 認識 Prompt Engineering 的重要性，及如何優化提示。
- 包含：
  - 一般問答
  - 總結
  - Few-shot 提示分類
  - 零次分類 (Zero-shot)
  - 實體提取 (Entity Extraction)
- 探討透過 LLM 進行 prompt 優化的方法：
  - ✅ 手動試錯
  - ✅ LLM 自我反思
  - ✅ 加入範例 Few-shot Learning
  - ✅ 任務分解
- 範例程式：
  - `basic.py`：測試 AOAI API 串接範例
  - `few_shot.py`：推文情感分析 Few-shot 範例
  - `json_format.py`：LLM 回應 JSON 格式化練習
- 練習任務：設計新聞分類系統及隨機生成書籍資料。

---

### 🔸 lab02. Chaining Prompt
- 介紹 Chaining Prompt 的定義與背景，說明如何將複雜任務分解為多個子任務，逐步執行。
- 優勢包含：
  - ✅ 分解複雜性
  - ✅ 提升準確性
  - ✅ 增加可解釋性
  - ✅ 增強可靠性
- 技術類型：
  - ➡️ 順序鏈接 (Sequential Chaining)
  - ➡️ 條件鏈接 (Conditional Chaining)
  - ➡️ 循環鏈接 (Looping Chaining)
- 實務範例與演練：
  - `web_search.py`：結合 AOAI 與 Tavily API 進行多步驟資料搜尋、摘要與分析
  - `calculator.py`：將中文數學題目轉為標準運算式並計算結果
  - `stock_api.py`：串接證交所 API 進行股票查詢並生成專業報告


### 🔸 lab03. Embedding

- 介紹什麼是 Embedding 及其在 LLM 應用中的重要角色。
- 功能重點：
  - ✅ 使用 Azure OpenAI 將文字轉為向量（embedding）。
  - ✅ 多執行緒加速大量文字資料處理。
  - ✅ 計算餘弦相似度找出文本之間的關聯性。
  - ✅ 使用 t-SNE 技術將高維向量降維並進行視覺化。
  - ✅ 輸入自訂查詢後，呈現相似結果與視覺化分布圖。
- 範例檔案：
  - `embedding.py`：示範 embedding 計算、相似度搜尋及視覺化。
- 範例流程：
  1. 從網路下載書籍資料集。
  2. 使用多執行緒執行 embedding。
  3. 計算查詢句與資料集間的相似度。
  4. 視覺化結果，並在圖中標示出查詢點與相似點。
- 練習任務：
  - 修改計算相似度的方法，嘗試使用 L2 距離 (歐氏距離)。


---

## 🛠️ 待續課程
- lab04. Conversation
- lab05. RAG-1
- lab06. RAG-2
- lab07. RAG-3

---

## ⚙️ 環境需求
- Python 3.11.7
- Anaconda（建議使用）
- 必要安裝套件：
  - openai
  - jupyter
  - python-dotenv
  - requests
  - numexpr
  - threading
  - tavily-python

---

## 🚀 使用方式
1. 建立虛擬環境
2. 安裝需求套件
3. 設定 `.env` 檔案（包含 AOAI_KEY、AOAI_URL、AOAI_MODEL_VERSION、TAVILY_API_KEY 等）
4. 依序執行各 lab 的程式範例與練習

---

> ✅ 本專案適合希望掌握 LLM 應用、Prompt 設計及實務操作的開發者、數據科學家及 AI 工程師。