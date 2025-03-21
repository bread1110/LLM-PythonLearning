"""
Stock Information Analysis System
-------------------------------
整合 Azure OpenAI 與台灣證券交易所 API 的股票資訊分析系統。
實現股票資訊提取、數據獲取和專業分析的完整流程。
"""

# 導入必要的套件
import os
import json
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv
from datetime import datetime

# 初始化環境設定
load_dotenv()  # 從 .env 檔案載入環境變數

def chat_with_aoai_gpt(prompt_messages: list[dict], user_json_format: bool = False) -> tuple[str, int, int]:
    """與 Azure OpenAI 服務互動的核心函數
    
    Args:
        prompt_messages: 對話歷史訊息列表，包含系統提示和用戶輸入
        user_json_format: 是否要求 AI 以 JSON 格式回應
    
    Returns:
        tuple: (AI回應內容, 輸入token數, 輸出token數)
    """
    retry_count = 0  # 重試計數器
    temperature = 0.7  # AI 回應的創造性程度 (0-1)
    
    while retry_count <= 2:  # 最多重試 2 次
        retry_count += 1
        try:
            # 從環境變數獲取 Azure OpenAI 服務配置
            aoai_key = os.getenv("AOAI_KEY")
            aoai_url = os.getenv("AOAI_URL")
            aoai_model_version = os.getenv("AOAI_MODEL_VERSION")

            # 初始化 Azure OpenAI 客戶端
            client = AzureOpenAI(
                api_key=aoai_key,
                azure_endpoint=aoai_url,
            )

            # 建立 API 請求並獲取回應
            aoai_response = client.chat.completions.create(
                model=aoai_model_version,
                messages=prompt_messages,
                temperature=temperature,
                response_format={"type": "json_object"} if user_json_format else None
            )

            # 提取 AI 回應內容
            assistant_message = aoai_response.choices[0].message.content

            # 返回 AI 回應及 token 使用統計
            return (
                assistant_message,
                aoai_response.usage.prompt_tokens,
                aoai_response.usage.completion_tokens,
            )
        except Exception as e:
            print(f"錯誤：{str(e)}")
            return "", 0, 0  # 發生錯誤時返回空值

if __name__ == "__main__":
    # 模擬用戶查詢
    user_query = "請問2025年的2月27號 台積電的股價表現如何?"

    # 步驟 1: 股票資訊提取
    # 建立提示訊息列表，包含系統角色定義和用戶查詢
    stock_extraction_messages = []
    stock_extraction_messages.append({"role": "system", "content": f"""你是一個專業的股票資訊提取專家。請依照以下規則處理用戶的查詢：

1. 提取規則：
   - 從用戶問題中提取日期和台灣股票資訊
   - 將公司名稱轉換為正確的股票代號（例如：台積電 = 2330）
   - 將日期轉換為 YYYYMMDD 格式

2. 股票代號規則：
   - 必須是台灣股票代號（4位數字）
   - 如果只提供公司名稱，需轉換為對應代號
   - 常見對照：
     台積電 = 2330
     鴻海 = 2317
     聯發科 = 2454
     台塑 = 1301

3. 日期格式規則：
   - 輸入格式可能為：2025年2月27號、2025-02-27、2025/02/27
   - 統一轉換為 YYYYMMDD 格式（例：20250227）
   - 未來日期也要正常處理

4. 錯誤處理：
   - 缺少日期：{{ "error": "未提供日期資訊" }}
   - 缺少股票資訊：{{ "error": "未提供股票資訊" }}
   - 格式錯誤：{{ "error": "日期格式錯誤" }}
   - 無效股票：{{ "error": "無效的股票代號或公司名稱" }}

5. 回傳格式：
{{
  "date": "20250227",
  "stock_code": "2330"
}}

請只回傳 JSON 格式的結果，不要包含任何其他說明文字。"""})
    stock_extraction_messages.append({"role": "user", "content": user_query})

    # 呼叫 AI 進行股票資訊提取
    extraction_response, extraction_prompt_tokens, extraction_completion_tokens = chat_with_aoai_gpt(
        stock_extraction_messages, 
        user_json_format=True  # 要求 JSON 格式回應
    )

    # 解析 AI 回應，獲取查詢參數
    extracted_data = json.loads(extraction_response)
    query_date = extracted_data["date"]  # 擷取查詢日期
    query_stock_code = extracted_data["stock_code"]  # 擷取股票代碼

    # 步驟 2: 從證交所 API 獲取股票資料
    stock_api_url = 'https://www.twse.com.tw/exchangeReport/STOCK_DAY'
    stock_api_response = requests.get(
        f'{stock_api_url}?response=json&date={query_date}&stockNo={query_stock_code}'
    )
    stock_trading_data = json.loads(stock_api_response.text)  # 解析 API 回應
    
    # 步驟 3: 股票數據分析
    # 建立分析提示訊息列表
    stock_analysis_messages = []
    stock_analysis_messages.append({"role": "system", "content": f"""你是一位專業的股票分析師，需要根據提供的股票交易資料回答用戶問題。

1. 資料解讀規則：
   - 仔細分析 context 中的股票交易資料
   - 如果查詢日期沒有資料，請說明原因（例如：非交易日、未來日期）
   - 數據包含：日期、成交股數、成交金額、開盤價、最高價、最低價、收盤價、漲跌價差、成交筆數

2. 回答格式要求：
   - 使用清晰的中文回答
   - 數據要包含具體數字
   - 價格相關數據需要標示單位（元）
   - 漲跌需標示正負號和百分比

3. 回答內容要求：
   - 優先回答用戶直接詢問的內容
   - 如果是未來日期，請明確說明這是未來日期，無法提供實際交易資料
   - 如果當日為非交易日，請明確說明

4. 專業分析：
   - 提供當日交易概況
   - 說明價格波動情況
   - 提供成交量資訊
   - 比較開盤價和收盤價的差異

5. 注意事項：
   - 保持客觀專業的語氣
   - 不要提供投資建議
   - 只使用提供的數據進行分析
   - 如果數據不完整或有異常，請明確指出

當前日期：{datetime.now().strftime("%Y-%m-%d")}
"""})
    stock_analysis_messages.append({
        "role": "user", 
        "content": f"""
用戶問題: {user_query}
交易資料: {stock_trading_data}
"""
    })

    # 呼叫 AI 進行股票數據分析
    analysis_response, analysis_prompt_tokens, analysis_completion_tokens = chat_with_aoai_gpt(
        stock_analysis_messages
    )
    # 輸出分析結果
    print(analysis_response)

    # TODO: 練習 - 整合 Tavily API 獲取相關新聞
   