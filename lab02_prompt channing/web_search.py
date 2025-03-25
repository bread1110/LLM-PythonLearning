"""
Web Search and Analysis System
-----------------------------
這是一個基於 Azure OpenAI 和 Tavily 的智能搜尋與分析系統。
實現了完整的 prompt chaining 流程，包含關鍵字提取、網頁搜尋、內容摘要和最終分析。
"""

# 導入必要的套件
import os
import threading
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
from tavily import TavilyClient
from datetime import datetime
from queue import Queue

# 初始化設定
load_dotenv()  # 載入環境變數
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))  # 初始化 Tavily 客戶端

def chat_with_aoai_gpt(messages: list[dict], user_json_format: bool = False) -> tuple[str, int, int]:
    """與 Azure OpenAI 服務互動的核心函數
    
    Args:
        messages: 對話歷史列表
        user_json_format: 是否要求 JSON 格式回應
    
    Returns:
        tuple: (AI回應內容, 輸入token數, 輸出token數)
    """
    error_time = 0 # 錯誤次數
    temperature=0.7 # 溫度
    while error_time <= 2: # 如果錯誤次數小於2次，則繼續嘗試
        error_time += 1
        try:
            aoai_key = os.getenv("AOAI_KEY")                        # 取得AOAI金鑰
            aoai_url = os.getenv("AOAI_URL")                        # 取得AOAI URL
            aoai_model_version = os.getenv("AOAI_MODEL_VERSION")    # 取得AOAI模型版本

            # 初始化 API 客戶端
            client = AzureOpenAI(
                api_key=aoai_key,           # 設置AOAI金鑰
                azure_endpoint=aoai_url,    # 設置AOAI URL
            )

            # 發送請求給 API

            # 如果response_format存在，則設置response_format
            aoai_response = client.chat.completions.create(
                model=aoai_model_version,
                messages=messages,          # 設置聊天訊息，包含系統角色和使用者輸入
                temperature=temperature,    # 設置溫度
                response_format={"type": "json_object"} if user_json_format else None  # 設置回應格式
            )

            # 透過AOAI SDK取得AOAI的回答
            assistant_message = aoai_response.choices[0].message.content

            # 回傳Tuple物件(AOAI的回答、提示token數、回答token數)
            return (
                assistant_message,
                aoai_response.usage.prompt_tokens,
                aoai_response.usage.total_tokens - aoai_response.usage.prompt_tokens,
            )
        except Exception as e: # 如果發生錯誤，則回傳空字串、0、0
            print(f"錯誤：{str(e)}")
            return "", 0, 0

if __name__ == "__main__":
    # 步驟 1: 關鍵字提取
    # 設定當前日期格式
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    # 定義關鍵字提取的 prompt
    keyword_extraction_prompt = f"""你是一個專業的關鍵字提取專家。你的任務是：
1. 從用戶的查詢中提取多組關鍵搜尋字詞
2. 將這些關鍵字組合成多個簡潔的英文搜尋查詢
3. 以 JSON 格式返回搜尋查詢列表

輸出格式要求：
{{
    "search_queries": [
        "query1 here",
        "query2 here",
        "query3 here"
    ]
}}

關鍵字格式要求：
- 使用英文
- 關鍵字之間用空格分隔
- 重要詞組可以用引號包圍
- 不要使用特殊符號或運算符
- 每組查詢應該聚焦於不同的主題面向

當前日期：{current_date}
"""
    
    # 模擬使用者輸入
    user_query = """
    您好，我是一位剛開始關注科技股的投資新手。我最近一直在研究人工智能相關的投資機會，聽說Nvidia是這個領域的領導者。
Nvidia的股價表現如何？有沒有遇到什麼重大波動？另外，人工智能熱潮對Nvidia股價有什麼具體影響？
如果可以的話，也希望您能簡單說明一下影響Nvidia股價的主要因素有哪些。謝謝您的解答！
"""
    
    # 組建訊息並呼叫 AI 進行關鍵字提取
    messages_for_keywords = [
        {"role": "system", "content": keyword_extraction_prompt},
        {"role": "user", "content": user_query}
    ]
    keyword_response, keyword_prompt_tokens, keyword_completion_tokens = chat_with_aoai_gpt(
        messages_for_keywords, 
        user_json_format=True
    )
    
    # 解析關鍵字JSON
    search_queries = json.loads(keyword_response)

    # 步驟 2: 多執行緒網頁搜尋
    search_threads = []
    web_search_queue = Queue()  # 用於存儲搜尋結果的隊列

    def search_tavily(query):
        """執行 Tavily 搜尋的輔助函數"""
        search_result = tavily_client.search(query)
        web_search_queue.put(search_result)

    # 為每個搜尋查詢創建並啟動執行緒
    for query in search_queries["search_queries"]:
        search_thread = threading.Thread(target=search_tavily, args=(query,))
        search_threads.append(search_thread)
        search_thread.start()

    # 等待所有搜尋完成
    for search_thread in search_threads:
        search_thread.join()

    # 整合所有搜尋結果
    all_search_results = []
    while not web_search_queue.empty():
        search_result = web_search_queue.get()
        all_search_results.extend(search_result["results"])

    # 步驟 3: 內容摘要處理
    summary_prompt = f"""你是一個專業的財經分析師，負責整理和摘要網路上的財經資訊。你需要針對以下使用者問題，從搜尋結果中提取相關資訊：

使用者問題：
{user_query}

請遵循以下準則進行摘要：
1. 從提供的文章標題、摘要中提取與使用者問題相關的重要信息
2. 摘要要求：
   - 以簡潔的中文總結關鍵信息
   - 確保信息準確性，摘要的結果必須包含日期和數據本身
   - 保持客觀，不要添加個人觀點
   - 總結長度控制在100-150字之間
   - 優先提取與使用者問題直接相關的資訊

當前日期：{current_date}
"""
    
    summary_threads = []
    summary_queue = Queue()  # 用於存儲摘要結果的隊列

    def get_summary(summary_messages, summary_queue):
        """處理單個搜尋結果的摘要"""
        summary_response = chat_with_aoai_gpt(summary_messages)
        summary_queue.put(summary_response)

    # 為每個搜尋結果創建摘要處理執行緒
    for search_item in all_search_results:
        summary_messages = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": search_item["title"] + "\n" + search_item["content"]}
        ]
        summary_thread = threading.Thread(target=get_summary, args=(summary_messages, summary_queue))
        summary_threads.append(summary_thread)
        summary_thread.start()

    # 等待所有摘要處理完成
    for summary_thread in summary_threads:
        summary_thread.join()

    # 步驟 4: 最終回答生成
    final_prompt = f"""你是一位專業的財經分析師，需要根據提供的參考資料回答投資者的問題。
回答要求：
1. 資料使用原則：
   - 僅使用提供的參考資料內容進行回答
   - 如果某個問題在參考資料中找不到答案，請明確說明「根據現有資料無法回答該問題」
   - 清楚標示資料的時間點，特別是股價和市值相關數據

2. 回答格式：
   - 使用結構化的方式分點回答問題
   - 對於數據變化，提供精確的數字和百分比
   - 使用通俗易懂的語言，適合投資新手理解

3. 內容準則：
   - 保持客觀，不加入個人投資建議
   - 提供事實性的市場資訊和數據
   - 說明資訊的時效性
   - 如遇到預測性內容，需標明是分析師觀點而非確定性結論

當前日期：{current_date}
"""
    final_messages = [{"role": "system", "content": final_prompt}]
    
    # 收集所有摘要並加入最終訊息
    while not summary_queue.empty():
        final_messages.append({"role": "user", "content": f"參考資料:\n{summary_queue.get()}"})

    # 加入原始問題
    final_messages.append({"role": "user", "content": f"原始問題:\n{user_query}"})

    # 生成最終回答
    final_response, final_prompt_tokens, final_completion_tokens = chat_with_aoai_gpt(final_messages)
    print(final_response)

    # TODO: 練習- 實現 Rerank 機制來提升相關性