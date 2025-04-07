# 導入必要的套件
import streamlit as st              # 用於建立網頁介面
import os                          # 用於處理環境變數
from openai import AzureOpenAI     # Azure OpenAI API 客戶端
from dotenv import load_dotenv     # 用於載入環境變數
import time                        # 用於模擬打字效果

# 載入環境變數
load_dotenv()

# 從環境變數中獲取 Azure OpenAI 服務所需的配置
aoai_key = os.getenv("AOAI_KEY")             # Azure OpenAI API 金鑰
aoai_url = os.getenv("AOAI_URL")             # Azure OpenAI 服務端點 URL
aoai_model_version = os.getenv("AOAI_MODEL_VERSION")    # 使用的模型版本

# 檢查必要的環境變數是否都已設置
if not all([aoai_key, aoai_url, aoai_model_version]):
    st.error("請確保 .env 檔案中已設定 AOAI_KEY, AOAI_URL, 和 AOAI_MODEL_VERSION")
    st.stop()

def chat_with_aoai_gpt(messages: list[dict]) -> tuple[str, int, int]:
    """與 Azure OpenAI 服務互動的核心函數
    
    Args:
        messages: 包含對話歷史的列表，每個元素是包含 role 和 content 的字典
    
    Returns:
        tuple: 包含三個元素：
            - AI的回應內容 (str)
            - 輸入消息的 token 數量 (int)
            - 輸出回應的 token 數量 (int)
    """
    error_time = 0     # 記錄重試次數
    temperature = 0.7  # 控制回應的創造性/隨機性，0為最保守，1為最創造性
    
    while error_time <= 2:  # 最多重試3次
        error_time += 1
        try:
            # 初始化 Azure OpenAI 客戶端
            client = AzureOpenAI(
                api_key=aoai_key,
                azure_endpoint=aoai_url,
            )

            # 發送請求到 Azure OpenAI 服務
            aoai_response = client.chat.completions.create(
                model=aoai_model_version,
                messages=messages,
                temperature=temperature,
            )

            # 提取 AI 的回應
            assistant_message = aoai_response.choices[0].message.content

            # 返回 AI 回應及相關的 token 使用統計
            return (
                assistant_message,
                aoai_response.usage.prompt_tokens,
                aoai_response.usage.total_tokens - aoai_response.usage.prompt_tokens,
            )
        except Exception as e:
            print(f"錯誤：{str(e)}")
            return "", 0, 0  # 發生錯誤時返回空值

# 設置網頁標題和說明
st.title("💬 我的第一個 LLM Chatbot")
st.caption("🚀 使用 Streamlit 和 LLM API 建立")

# 初始化對話歷史
if "messages" not in st.session_state:
    # 設置系統角色的初始提示
    st.session_state.messages = [{"role": "system", "content": "你是一個友善且樂於助人的 AI 助理。"}]
    print("Session state 初始化完成")

# 顯示歷史對話內容
for message in st.session_state.messages:
    if message["role"] != "system":  # 不顯示系統提示
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 處理用戶輸入
if prompt := st.chat_input("在這裡輸入你的訊息..."):
    # 將用戶輸入添加到對話歷史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    print(f"User: {prompt}")

    # 顯示 AI 助理的回應
    with st.chat_message("assistant"):
        message_placeholder = st.empty()  # 創建空白容器用於顯示回應
        message_placeholder.markdown("思考中...")  # 顯示載入提示

        # 獲取 AI 的回應
        assistant_response, prompt_tokens, completion_tokens = chat_with_aoai_gpt(st.session_state.messages)

        # 模擬打字效果顯示回應
        full_response = ""
        for chunk in assistant_response:
            full_response += chunk
            time.sleep(0.01)  # 添加延遲以創造打字效果
            message_placeholder.markdown(full_response + "▌")  # 顯示打字游標
        message_placeholder.markdown(full_response)  # 顯示完整回應

    # 將 AI 的回應添加到對話歷史
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    print(f"Assistant: {assistant_response}")

    # 重新加載頁面以更新對話
    st.rerun()