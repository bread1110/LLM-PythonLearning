import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

def get_response(user_message: str) -> tuple[str, int, int]:
    """取得AOAI的回答
    
        Args:
        user_message (str): 使用者輸入
        
    Returns:
        tuple[str, int, int]: (情感分析結果, 提示詞token數, 回應token數)
    """
    # 用於儲存聊天對話的list
    messages = []
    # 這次對話中，請LLM扮演一個樂於助人的助理
    messages.append({"role": "system", "content": """請用 JSON 格式回傳，用以下格式:
[
  "user1": {
    "name": "string", // 請用台灣常見姓名
    "age": "integer", // 年紀
    "bio": "text", // 請用台灣繁體中文
    "avatar_url": "url", // 個人圖像，請用真實可以連結的圖片
    "isSubscriber": "boolean", // 是否訂閱
}]
"""})
    # 使用者輸入
    messages.append({"role": "user", "content": user_message})
    return chat_with_aoai_gpt(messages) # 呼叫AOAI服務取得回應

def chat_with_aoai_gpt(messages: list[dict]) -> tuple[str, int, int]:
    """呼叫AOAI服務取得回應
        
    Args:
        messages (list[dict]): 包含對話歷史的訊息列表
        
    Returns:
        tuple[str, int, int]: (AI回應, 提示詞token數, 回應token數)
    """
    # 以下為新版指定response_format，可以指定json_schema，但我們的AOAI Model Version不支援，response_format 作為 json_schema 僅在 api 版本 2024-08-01-preview 及更高版本中啟用
    #response_format={
    #    "type": "json_schema",
    #    "json_schema": {
    #        "name": "user_data_response",
    #        "schema": {
    #            "type": "object",
    #            "properties": {
    #                "name": {
    #                    "type": "string",
    #                    "description": "請用台灣常見姓名"
    #                },
    #                "age": {
    #                    "type": "integer",
    #                    "description": "年紀"
    #                },
    #                "bio": {
    #                    "type": "string",
    #                    "description": "請用台灣繁體中文"
    #                },
    #                "avatar_url": {
    #                    "type": "string",
    #                    "description": "個人圖像，請用真實可以連結的圖片"
    #                },
    #                "isSubscriber": {
    #                    "type": "boolean",
    #                    "description": "是否訂閱"
    #                }
    #            },
    #            "required": [
    #                "name",
    #                "age",
    #                "bio",
    #                "avatar_url",
    #                "isSubscriber"
    #            ],
    #            "additionalProperties": False
    #        },
    #        "strict": True
    #    }
    #}
    
    response_format = { "type": "json_object" }

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
            response = client.chat.completions.create(
                model=aoai_model_version,
                messages=messages,                  # 設置聊天訊息，包含系統角色和使用者輸入
                temperature=temperature,            # 設置溫度
                response_format=response_format,    # 設置回傳格式
            )

            # 透過AOAI SDK取得AOAI的回答
            assistant_message = response.choices[0].message.content

            # 回傳Tuple物件(AOAI的回答、提示token數、回答token數)
            return (
                assistant_message,
                response.usage.prompt_tokens,
                response.usage.total_tokens - response.usage.prompt_tokens,
            )
        except Exception as e: # 如果發生錯誤，則回傳空字串、0、0
            print(f"錯誤：{str(e)}")
            return "", 0, 0

if __name__ == "__main__":
    assistant_response ,prompt_tokens, completion_tokens = get_response("請隨機產生三個 user 資料")
    if assistant_response:
        try:
            json_data = json.loads(assistant_response)
            print(json_data)
        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤：{str(e)}")
        print(f"生成User資料, 提示token數: {prompt_tokens}, 回答使用token數: {completion_tokens}\n: {assistant_response}\n")

    # 以下為練習，請透過改寫prompt，生成三本書籍資料，必須要包含以下欄位:
    # title: 字串，書籍標題（台灣繁體中文）
    # genre: 字串，書籍類型
    # price: 整數，價格（單位：新台幣）
    # inStock: 布林值，是否庫存中有貨
    assistant_response ,prompt_tokens, completion_tokens = get_response("請隨機產生三本書籍資料")
    if assistant_response:
        try:
            json_data = json.loads(assistant_response)
            if json_data and "title" in json_data and "genre" in json_data and "price" in json_data and "inStock" in json_data:
                for book in json_data:
                    print(f"書籍標題: {book['title']} 書籍類型: {book['genre']} 價格: {book['price']} 是否庫存中有貨: {book['inStock']}")
        except json.JSONDecodeError as e:
            print(f"JSON 解析錯誤：{str(e)}")
        print(f"生成書籍資料, 提示token數: {prompt_tokens}, 回答使用token數: {completion_tokens}\n: {assistant_response}\n")
