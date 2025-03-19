import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# 載入 .env 檔案中的環境變數
load_dotenv()

def get_response(user_message):
    messages = []
    messages.append({"role": "system", "content": "您是一個樂於助人的助理。"})
    messages.append({"role": "user", "content": user_message})
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
    # Zero-shot 分類
    complaint = """以下段落是一則消費者投訴。投訴內容涉及以下選項之一：信用卡、信用報告、抵押貸款與貸款、零售銀行業務或債務追討。請閱讀以下段落並判斷投訴屬於哪個選項。
    我多年來一直在富國銀行（Wells Fargo）持有抵押貸款。每個月我都會提前7-10天付款。在 XX/XX/XXXX 至 XX/XX/XXXX 期間，我每月支付 $3000.00。在 XXXX 年，我接到富國銀行的電話，說我的月付款金額不正確。經過長時間討論，我同意額外支付 $750.00 以使帳戶恢復正常，並從此支付 $XXXX。在 XX/XX/XXXX，我收到一封來自 XXXX 的信，稱我的抵押貸款已違約，並建議我立即採取行動。經過長時間討論，我終於發現，在 XX/XX/XXXX，銀行如常收到我的付款，但因為金額低於他們的要求，他們沒有將這筆錢用於支付我的抵押貸款，而是將全部金額應用於本金。他們從未通知我。他們一直向信用機構報告我，還威脅要沒收我的房子，聲稱我未付款，而事實上我從未漏付或遲交。他們這樣對待我，卻連通知都沒有。為什麼他們不打電話給我？他們檔案中有兩個電話號碼，其中一個已經停用20年，他們從未撥打另一個號碼。我注意到我在 XXXX 年與一位年輕人通話時，他能通過電話聯繫到我。為什麼不寄信？他們為什麼這樣對我？他們說是電腦造成的。為什麼他們不能回溯修復？他們說時間太久了。我必須在今年支付第13次抵押貸款，否則將面臨沒收。他們在欺騙我。你會以為可以信任銀行管理你的帳戶，但現在我明白這不是真的。我已經 XXXX 歲了，也許這就是他們採取這種政策的理由。"""
    response = get_response(complaint)
    print(f"投訴類別:\\n{response[0]}\\n")