import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

# One-shot 實體提取
def get_one_shot_entity_extraction(user_message):
    messages = [
        {"role": "system", "content": "從給定的段落中提取個人身份資訊（PII）實體。"},
        {"role": "user", "content": "我從紐約的銀行通過手機提領了 $100。電話號碼 (345) 123-7867。問候，Raj"},
        {"role": "assistant", "content": "1. 金額: $100\\n2. 地點: 紐約\\n3. 電話號碼: (345) 123-7867\\n4. 姓名: Raj"},
        {"role": "user", "content": user_message}
    ]
    return chat_with_aoai_gpt(messages)

def chat_with_aoai_gpt(messages):
    error_time = 0
    temperature=0.7
    while error_time <= 2:
        error_time += 1
        try:
            aoai_key = os.getenv("AOAI_KEY")
            aoai_url = os.getenv("AOAI_URL")
            aoai_model_version = os.getenv("AOAI_MODEL_VERSION")

            client = AzureOpenAI(
                api_key=aoai_key,
                azure_endpoint=aoai_url,
            )
            response = client.chat.completions.create(
                model=aoai_model_version,
                # 設置聊天訊息，包含系統角色和使用者輸入
                messages=messages,
                temperature=temperature,
            )

            assistant_message = response.choices[0].message.content

            return (
                assistant_message,
                response.usage.prompt_tokens,
                response.usage.total_tokens - response.usage.prompt_tokens,
            )
        except Exception as e:      
            print(f"錯誤：{str(e)}")
            return "", 0, 0

if __name__ == "__main__":
    # One-shot 實體提取
    test_input = "嗨，我是 Ravi Dube。我在 2023 年 3 月 30 日的信用卡對帳單上注意到一筆 $1,000 的費用。該交易是在紐約的一家餐廳進行的。請通過 (123)456-7890 或 ravi.dube@email.com 聯繫我。"
    response = get_one_shot_entity_extraction(test_input)
    print(f"提取的實體:\\n{response[0]}\\n")