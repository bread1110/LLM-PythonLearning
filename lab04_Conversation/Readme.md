# AI Chatbot Application

## 簡介
這是一個基於 Azure OpenAI 服務的智能聊天機器人應用程式，使用 Streamlit 框架建立友善的網頁介面。此應用程式允許使用者與 AI 助理進行自然對話，並提供即時的智能回應。

## 功能特點
- 🌐 直觀的網頁聊天介面
- 🤖 Azure OpenAI GPT 模型整合
- 💾 對話歷史記錄管理
- ⌨️ 真實的打字動畫效果
- 🔄 自動重試機制
- 🔒 安全的環境變數管理

## 安裝需求
### 必要套件
```bash
pip install streamlit openai python-dotenv
```

### 環境設定
在專案根目錄建立 `.env` 檔案，並設定以下變數：
```ini
AOAI_KEY=您的_Azure_OpenAI_API_金鑰
AOAI_URL=您的_Azure_OpenAI_API_端點
AOAI_MODEL_VERSION=您的_GPT_模型版本
```

## 使用方法
1. **啟動應用程式**
   ```bash
   streamlit run chatbot_app.py
   ```

2. **開始對話**
   - 在瀏覽器中開啟顯示的網址
   - 在底部輸入框輸入訊息
   - 按下 Enter 鍵送出訊息

## 系統需求
- Python 3.11.7 或更高版本
- 有效的 Azure OpenAI API 訂閱

## 自訂設定
### 修改 AI 助理角色
可以透過修改系統提示來改變 AI 助理的行為：
```python
st.session_state.messages = [
    {"role": "system", "content": "你是一個專業的技術顧問。"}
]
```

### 調整回應特性
在 `chat_with_aoai_gpt` 函數中修改 temperature 參數：
- 0.0：更確定且一致的回應
- 1.0：更具創造性的回應
```python
temperature = 0.7  # 預設值，可依需求調整
```

## 錯誤處理
- 自動檢查環境變數設定
- API 呼叫失敗時自動重試（最多3次）
- 清晰的錯誤訊息顯示

## 技術架構
- **前端框架**：Streamlit
- **AI 服務**：Azure OpenAI
- **狀態管理**：Streamlit Session State
- **設定管理**：python-dotenv

## 注意事項
1. 確保 `.env` 檔案中的所有設定值都已正確填寫
2. 檢查 Azure OpenAI 服務配額是否足夠
3. 不要將 API 金鑰直接寫入程式碼中
4. 定期監控 API 使用量
