"""
Math Expression Calculator System
-------------------------------
整合 Azure OpenAI 與 numexpr 的數學計算系統。
將中文數學題轉換為標準運算表達式並計算結果。
"""

# 導入必要的套件
import os
import numexpr  # 用於高效能數學運算
from openai import AzureOpenAI
from dotenv import load_dotenv

# 初始化環境設定
load_dotenv()  # 從 .env 檔案載入環境變數

def chat_with_aoai_gpt(messages: list[dict], user_json_format: bool = False) -> tuple[str, int, int]:
    """與 Azure OpenAI 服務互動的核心函數
    
    Args:
        messages: 對話歷史訊息列表，包含系統提示和用戶輸入
        user_json_format: 是否要求 AI 以 JSON 格式回應
    
    Returns:
        tuple: (AI回應內容, 輸入token數, 輸出token數)
    """
    error_time = 0  # 重試計數器
    temperature = 0.7  # AI 回應的創造性程度 (0-1)
    
    while error_time <= 2:  # 最多重試 2 次
        error_time += 1
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
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"} if user_json_format else None
            )

            # 提取 AI 回應內容
            assistant_message = aoai_response.choices[0].message.content

            # 返回 AI 回應及 token 使用統計
            return (
                assistant_message,
                aoai_response.usage.prompt_tokens,
                aoai_response.usage.total_tokens - aoai_response.usage.prompt_tokens,
            )
        except Exception as e:
            print(f"錯誤：{str(e)}")
            return "", 0, 0  # 發生錯誤時返回空值

if __name__ == "__main__":
    # 定義數學問題
    question = "我想要計算以下的國中數學: 請計算 64 乘以 2 再扣掉 8，以上結果再除100後，再指數 1.234"

    # 設計 prompt 引導 AI 將中文數學題轉換為 Python 表達式
    prompt = f"""
Translate this math problem into a Python numexpr-compatible expression.
Follow these rules:
1. Pay attention to the order of operations as described in the question
2. Use parentheses to ensure correct calculation order
3. For sequential calculations, wrap each step in parentheses
4. Return ONLY the expression with no additional text

Examples:
Question: Calculate 10 plus 5, then multiply by 2
Expression: ((10 + 5) * 2)

Question: Calculate 100 times 2, then subtract 50, finally divide by 10
Expression: (((100 * 2) - 50) / 10)

Question: Calculate 25 divided by 5, then raise to power of 2
Expression: ((25 / 5)**2)

Question: {question}
Expression:
"""

    # 建立訊息列表並呼叫 AI 進行轉換
    messages = [{"role": "user", "content": prompt }]
    expression, prompt_tokens, completion_tokens = chat_with_aoai_gpt(messages)

    # 輸出生成的表達式
    print(f"表達式: {expression}")

    # 使用 numexpr 計算表達式結果
    answer = numexpr.evaluate(expression)

    # 輸出計算結果
    print(f"計算結果: {answer}")

    # TODO: 練習 - 讓 AI 說明計算過程