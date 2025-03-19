import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

def get_few_shot_response(user_message: str) -> tuple[str, int, int]:
    """使用 Few-Shot Learning 方式進行情感分析
    
    透過提供多個情感分析的範例，讓 AI 學習如何判斷文字的情感傾向。
    
    Args:
        user_message (str): 要進行情感分析的文字
        
    Returns:
        tuple[str, int, int]: (情感分析結果, 提示詞token數, 回應token數)
    """
    # 初始化對話紀錄列表
    messages = []
    
    # 設定系統提示詞，說明任務目標和背景
    messages.append({
        "role": "system", 
        "content": """twitter 是一個社交媒體平台，用戶可以發佈推文。 推文可以是積極的或消極的，我
們希望能夠將推文分類成積極或消極。 以下是一些積極和消極推文的例子。 請確保
正確分類最後一個推文是積極的還是消極的"""
    })
    
    # Few-Shot 示例 1：積極範例
    messages.append({"role": "user", "content": "今天真是開心的一天"})
    messages.append({"role": "assistant", "content": "積極的"})
    
    # Few-Shot 示例 2：消極範例
    messages.append({"role": "user", "content": "我討厭這個班級"})
    messages.append({"role": "assistant", "content": "消極的"})
    
    # Few-Shot 示例 3：積極範例
    messages.append({"role": "user", "content": "我喜歡這杯咖啡"})
    messages.append({"role": "assistant", "content": "積極的"})
    
    # 添加使用者要分析的文字
    messages.append({"role": "user", "content": user_message})
    
    # 呼叫 API 進行分析
    return chat_with_aoai_gpt(messages)

def chat_with_aoai_gpt(messages: list[dict]) -> tuple[str, int, int]:
    """與 Azure OpenAI API 進行通訊
    
    Args:
        messages (list[dict]): 包含對話歷史的訊息列表
        
    Returns:
        tuple[str, int, int]: (AI回應, 提示詞token數, 回應token數)
    """
    error_time = 0  # 錯誤計數器
    temperature = 0.7  # 控制回應的隨機性，0為最確定，1為最隨機
    
    # 最多重試 2 次
    while error_time <= 2:
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
            
        except Exception as e:
            # 發生錯誤時輸出錯誤訊息
            print(f"錯誤：{str(e)}")
            return "", 0, 0

if __name__ == "__main__":
    # 測試範例：分析一句正面的句子
    assistant_response = get_few_shot_response("今天工作進度都滿順利的")
    if assistant_response:
        print(f"推文語意判斷: {assistant_response[0]}\n")

    # 下為練習，請透過改寫prompt，從句子中擷取出主詞
    # - 設計一個分類新聞類型的系統（政治/體育/科技/娛樂）
    # - 提供適當的示例讓模型學習分類規則
    test_sentences = [
    "政府宣布新稅收政策，預計明年起實施",
    "本地足球隊在決賽中以3:2擊敗對手，奪得冠軍",
    "最新智能手機搭載AI晶片，效能提升50%",
    "知名歌手發行新專輯，粉絲熱烈期待",
    "總理將於下週與外國領袖會晤，討論貿易合作",
    "奧運選手刷新100米短跑紀錄，創歷史新高",
    "科學家發現新型電池技術，可延長續航時間",
    "電影續集票房破億，刷新年度紀錄",
    "議會通過新法案，引發民眾抗議",
    "籃球明星因傷退出本季比賽，粉絲震驚",
    "虛擬現實技術革新遊戲體驗，預計年底上市",
    "電視劇大結局播出，收視率創新高",
    "選舉結果揭曉，新領導人即將上任",
    "網球公開賽爆冷門，頭號種子意外出局",
    "太空公司成功發射新型火箭，成本大幅降低",
    "明星夫婦公開婚訊，社群媒體瘋傳",
    "國際氣候協議達成，各國承諾減排",
    "馬拉松比賽吸引數千人參與，場面熱烈",
    "量子計算機原型亮相，運算速度驚人",
    "實境秀新一季開播，話題性十足",
]
    