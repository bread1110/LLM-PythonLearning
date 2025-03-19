import os
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
    messages.append({"role": "system", "content": "您是一個樂於助人的助理。"})
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
                messages=messages,          # 設置聊天訊息，包含系統角色和使用者輸入
                temperature=temperature,    # 設置溫度
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
    # 呼叫AOAI服務取得回應
    assistant_response ,prompt_tokens, completion_tokens = get_response("描述三具關於猴子的事實")
    if assistant_response:
        print(f"三個關於猴子的事實, 提示token數: {prompt_tokens}, 回答使用token數: {completion_tokens}\n: {assistant_response}\n")

    # 呼叫AOAI服務取得回應
    assistant_response ,prompt_tokens, completion_tokens = get_response("草是什麼顏色？")
    if assistant_response:
        print(f"草是甚麼顏色, 提示token數: {prompt_tokens}, 回答使用token數: {completion_tokens}\n: {assistant_response}\n")

    story_1 = """總結以下的故事：\n司馬光生於光州光山，有一次，他跟幾個小夥伴在後院玩耍。 有一個孩子淘氣，他
爬到一口大水缸上，結果失足掉進去了。 水缸深，孩子小，眼看小夥伴就要淹死了，
其他的孩子都嚇傻了，有的孩子嚇得大哭，有的孩子嚇得去找大人。 就在此時，司
馬光急中生智，從地上撿起一塊大石頭，使勁向水缸擊去。 通過司馬光的砸缸行為，
水湧出來，小夥伴因此得救了。"""
    assistant_response ,prompt_tokens, completion_tokens = get_response(story_1)
    if assistant_response:
        print(f"故事總結, 提示token數: {prompt_tokens}, 回答使用token數: {completion_tokens}\n: {assistant_response}\n")
    
    story_2 = """你是一名醫生。 請閱讀這份病例史並預測患者的風險：
2005 年 1 ⽉ 1 ⽇：打籃球時左臂⻣折。 戴上石膏進行治療。
2016 年 3 ⽉ 15 ⽇：被診斷為高血壓。 開了利辛普利的處方。
2017 年 9 ⽉ 10 ⽇：患上肺炎。 用抗生素治療並完全康復。"""
    assistant_response ,prompt_tokens, completion_tokens = get_response(story_2)
    if assistant_response:
        print(f"閱讀病例結論, 提示token數: {prompt_tokens}, 回答使用token數: {completion_tokens}\n: {assistant_response}\n")


    # 下為練習，請透過改寫prompt，從句子中擷取出主詞
    test_sentences = [
    "小明在公園裡開心地跑步。",
    "那隻可愛的貓咪正在窗台上曬太陽。",
    "台北101是台灣最高的建築物。",
    "昨天下午王老師在教室裡講解數學題目。",
    "這台新手機的電池續航力非常優秀。",
    "櫻花季節的陽明山吸引了許多遊客。",
    "我的姐姐在美國西雅圖工作三年了。",
    "這家餐廳的牛肉麵聞名全台灣。",
    "張醫生在醫院急診室值大夜班。",
    "那部電影的特效讓觀眾印象深刻。"
]

