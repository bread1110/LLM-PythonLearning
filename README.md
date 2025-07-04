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

---

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

### 🔸 lab04. Conversation
- 深入探討如何建立智能對話系統，實現自然且連貫的人機對話互動。
- 核心概念：
  - ✅ 對話狀態管理
  - ✅ 上下文記憶與追蹤
  - ✅ 角色設定與人設維持
  - ✅ 對話流程控制
- 技術重點：
  - ➡️ 使用 Streamlit 建立互動式對話介面
  - ➡️ Azure OpenAI 對話串接整合
  - ➡️ 系統提示（System Prompt）設計
  - ➡️ 對話歷史管理機制
- 實作範例：
  - `chatbot_app.py`：完整的 AI 對話應用程式
  - `conversation_manager.py`：對話狀態管理工具
  - `prompt_templates.py`：系統提示模板集合
- 進階功能：
  - 🔄 打字動畫效果
  - 💾 對話記錄保存與讀取
  - 🔒 錯誤處理與重試機制
  - 🎯 角色切換與多重人設
- 練習任務：
  - 設計專業領域的對話助理
  - 實現多輪對話邏輯控制
  - 優化系統提示以提升回應品質

---

### 🔸 lab05. RAG (Retrieval Augmented Generation) & AI Agent
- 深入探討如何建立完整的 RAG 系統，結合 AI Agent 框架實現智能法律諮詢助手。
- 核心技術棧：
  - ✅ **Vector Database**：使用 PostgreSQL + pgvector 儲存向量資料
  - ✅ **Embedding**：Azure OpenAI text-embedding 向量化文本
  - ✅ **Reranker**：CrossEncoder 模型進行二次排序優化
  - ✅ **Function Calling**：AI Agent 自動選擇和調用工具
  - ✅ **Web Search**：整合 Tavily API 獲取即時資訊
- RAG 系統架構：
  - 📄 **資料處理**：PDF 解析、文本分割、向量化儲存
  - 🔍 **檢索優化**：向量搜索 + Reranker 二次排序
  - 🤖 **智能回答**：AI Agent 整合多種資料源生成回答
  - 🌐 **混合檢索**：結合本地知識庫與網路搜索
- AI Agent 功能：
  - ➡️ **自動工具選擇**：根據問題類型智能選擇合適工具
  - ➡️ **查詢改寫**：優化使用者問題以提升檢索準確性
  - ➡️ **多輪對話**：支援複雜問題的分步處理
  - ➡️ **錯誤處理**：完善的異常處理和降級機制
- 實作範例：
  - `rag.py`：RAG 系統核心實現
  - `process_data.py`：資料處理和向量化工具
  - `query_test.py`：AI Agent 查詢系統（主程式）
  - `embedding.py`：向量計算和相似度搜索
  - `create_cosine_similarity_function.sql`：資料庫函數設定
- 進階功能：
  - 🔄 **智能排序**：結合向量相似度和語義重排序
  - 📊 **統計分析**：資料庫統計和章節分析工具
  - 🌐 **即時資訊**：網路搜索補充最新政策和案例
  - 🎯 **專業問答**：針對勞動基準法的專業法律諮詢
- 技術亮點：
  - 🚀 **Function Calling**：OpenAI 最新功能實現智能工具調用
  - 🔧 **Reranker 優化**：使用 CrossEncoder 大幅提升檢索準確性
  - 💬 **自然語言介面**：支援口語化問題自動理解和處理
  - 📈 **效能監控**：Token 使用統計和處理時間追蹤
- 練習任務：
  - 擴展到其他法律領域的知識庫
  - 實現更多專業工具（如法條比較、案例分析）
  - 優化 Reranker 模型以提升特定領域的檢索效果

---

## ⚙️ 環境需求
- Python 3.11.7
- Anaconda（建議使用）
- 必要安裝套件：
  - **核心套件**：
    - openai
    - python-dotenv
    - requests
    - numpy
    - pandas
  - **機器學習**：
    - scikit-learn
    - sentence-transformers
    - transformers
  - **資料庫**：
    - psycopg2-binary
    - pgvector
  - **網路搜索**：
    - tavily-python
  - **文件處理**：
    - PyPDF2
    - langchain
  - **介面開發**：
    - jupyter
    - streamlit
  - **其他工具**：
    - threading
    - numexpr
---

## 🚀 使用方式

### 基本設定
1. **建立虛擬環境**
   ```bash
   conda create -n llm-learning python=3.11.7
   conda activate llm-learning
   ```

2. **安裝需求套件**
   ```bash
   pip install openai python-dotenv sentence-transformers psycopg2-binary tavily-python
   ```

3. **設定 `.env` 檔案**
   ```env
   # Azure OpenAI 設定
   AOAI_KEY=your_azure_openai_key
   AOAI_URL=your_azure_openai_endpoint
   AOAI_MODEL_VERSION=your_model_deployment_name
   
   # Embedding 設定
   EMBEDDING_API_KEY=your_embedding_key
   EMBEDDING_URL=your_embedding_endpoint
   EMBEDDING_MODEL=your_embedding_model
   
   # 網路搜索設定
   TAVILY_API_KEY=your_tavily_api_key
   
   # 資料庫設定（適用於 lab05）
   PG_HOST=localhost
   PG_PORT=5432
   PG_DATABASE=labor_law_rag
   PG_USER=postgres
   PG_PASSWORD=your_password
   ```

### Lab05 額外設定
4. **PostgreSQL + pgvector 設定**（僅適用於 lab05）
   - 安裝 PostgreSQL
   - 執行 `process_data.py` 處理法條資料

5. **執行程式**
   ```bash
   # 一般 lab 範例
   python basic.py
   
   # Lab05 AI Agent 系統
   cd lab05_RAG
   python query_test.py
   ```

---

> ✅ 本專案適合希望掌握 LLM 應用、Prompt 設計及實務操作的開發者、數據科學家及 AI 工程師。