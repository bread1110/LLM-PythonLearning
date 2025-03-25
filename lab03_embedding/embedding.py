"""
文本 Embedding 與相似度視覺化程式

這個程式實現以下功能：
1. 從網路獲取書籍資料集
2. 使用 Azure OpenAI 服務將文本轉換為 embedding 向量
3. 使用多執行緒加速 embedding 處理
4. 計算文本之間的相似度
5. 使用 t-SNE 進行降維視覺化
6. 支援使用者查詢並顯示相似結果

主要依賴套件：
- openai: Azure OpenAI API 客戶端
- pandas: 資料處理
- numpy: 數值計算
- sklearn: t-SNE 降維
- matplotlib: 資料視覺化
- threading: 多執行緒處理
"""

# 導入必要的套件
import requests
import os
import threading
import pandas as pd
from queue import Queue, Empty
from io import StringIO
from openai import AzureOpenAI
from dotenv import load_dotenv
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# 初始化設定
load_dotenv()  # 載入環境變數

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

def query_aoai_embedding(content: str) -> list[float]:
    """從 Azure OpenAI 服務獲取文本的 embedding 向量
    
    Args:
        content (str): 要進行 embedding 的文本內容
    
    Returns:
        list[float]: 返回 embedding 向量，如果發生錯誤則返回空列表
    """
    try_cnt = 2
    while try_cnt > 0:
        try_cnt -= 1
        api_key = os.getenv("EMBEDDING_API_KEY")
        api_base = os.getenv("EMBEDDING_URL")
        embedding_model = os.getenv("EMBEDDING_MODEL")

        try:
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_base,
            )
            embedding = client.embeddings.create(
                input=content,
                model=embedding_model,
            )
            return embedding.data[0].embedding
        except Exception as e:
            print(f"get_embedding_resource error | err_msg={e}")

    return []

def worker(queue: Queue, df: pd.DataFrame) -> None:
    """工作執行緒函數，從佇列中取出任務並處理 embedding
    
    Args:
        queue (Queue): 包含待處理任務的佇列，每個任務是 (index, data) 的元組
        df (pd.DataFrame): 要更新的資料框，包含 'embeddings' 欄位
    
    Returns:
        None
    """
    while True:
        try:
            # 從 queue 中取得任務
            index, data = queue.get(block=False)
            # 處理 embedding
            embedding = query_aoai_embedding(data)
            # 更新 DataFrame
            df.at[index, 'embeddings'] = embedding
            # 標記任務完成
            queue.task_done()
        except Empty:
            break

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """計算兩個向量之間的餘弦相似度
    
    Args:
        a (list[float]): 第一個向量
        b (list[float]): 第二個向量
    
    Returns:
        float: 兩個向量的餘弦相似度，範圍在 [-1, 1] 之間
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_similar(df: pd.DataFrame, query: str, n: int = 3) -> pd.DataFrame:
    """搜尋與查詢文本最相似的 n 筆資料
    
    Args:
        df (pd.DataFrame): 包含 embeddings 的資料框
        query (str): 查詢文本
        n (int, optional): 要返回的結果數量. 預設為 3
    
    Returns:
        pd.DataFrame: 包含最相似的 n 筆資料，並依相似度排序
    """
    query_embedding = query_aoai_embedding(query)
    df["similarity"] = df['embeddings'].apply(lambda x: cosine_similarity(x, query_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
    )
    return results

if __name__ == "__main__":
    """主程式流程：
    1. 從網路獲取書籍資料集
    2. 使用多執行緒處理每本書的 embedding
    3. 使用 t-SNE 進行降維視覺化，並標示使用者查詢和相似結果
    4. 執行相似度搜尋示例
    """
    # 取得書籍資料(Dataset)
    url = "https://ihower.tw/data/books-dataset-33.csv"
    response = requests.get(url)
    print(response.text)

    df = pd.read_csv(StringIO(response.text))
    df['embeddings'] = None

    # 建立任務佇列
    task_queue = Queue()
    
    # 將所有任務加入佇列
    for index, row in df.iterrows():
        data = f"{row['title']} {row['description']}"
        task_queue.put((index, data))
    
    # 建立工作執行緒
    num_threads = 8  # 可以根據需求調整執行緒數量
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(task_queue, df))
        t.start()
        threads.append(t)
    
    # 等待所有任務完成
    task_queue.join()
    
    # 等待所有執行緒結束
    for t in threads:
        t.join()

    print(df)

    # 將 DataFrame 中的 embeddings 轉換為 numpy 陣列
    matrix = np.array(df["embeddings"].to_list())

    # 定義使用者查詢文本並取得其 embedding
    query = "怎樣用 Python 做資料分析"
    query_embedding = query_aoai_embedding(query)
    
    # 將查詢向量垂直堆疊（vstack）到原始矩陣中
    # 這樣可以同時對所有向量（包含查詢）進行 t-SNE 降維
    matrix_with_query = np.vstack([matrix, query_embedding])
    
    # 使用 t-SNE 進行降維
    # n_components=2: 降至2維以便視覺化
    # perplexity=15: 用於平衡局部和全局結構的參數
    # random_state=42: 確保結果可重現
    # init='random': 初始化方式
    # learning_rate=200: 學習率參數
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
    vis_dims_with_query = tsne.fit_transform(matrix_with_query)
    
    # 分離降維後的結果為原始數據點和查詢點
    vis_dims = vis_dims_with_query[:-1]  # 取出除了最後一個點之外的所有點（原始數據）
    query_point = vis_dims_with_query[-1]  # 取出最後一個點（查詢點）
    
    # 使用相似度搜尋找出最相似的 n 個文件
    n = 5
    results = search_similar(df, query, n)
    similar_indices = results.index.tolist()  # 獲取相似文件的索引
    
    # 設置圖形大小
    plt.figure(figsize=(10, 8))
    
    # 繪製一般文件點（藍色）
    # 使用 mask 來過濾出非相似點的索引
    mask = ~np.isin(range(len(vis_dims)), similar_indices)
    plt.scatter(vis_dims[mask, 0], vis_dims[mask, 1], 
               c='blue', marker='o', label='Other documents', alpha=0.6)
    
    # 繪製相似文件點（橘色）
    # s=100 設置點的大小
    similar_points = vis_dims[similar_indices]
    plt.scatter(similar_points[:, 0], similar_points[:, 1], 
               c='orange', marker='o', label='Similar documents', s=100)
    
    # 繪製查詢點（紅色星號）
    # s=200 設置點的大小，使用星號標記
    plt.scatter(query_point[0], query_point[1], 
               c='red', marker='*', s=200, label='Query')
    
    # 添加網格線
    plt.grid(True)
    # 添加圖例
    plt.legend()
    # 設置圖表標題
    plt.title('t-SNE Visualization of Documents with Query')
    # 顯示圖表
    plt.show()

    # 印出最相似文件的標題和相似度分數
    print("\n最相似的文件：")
    print(results[['title', 'similarity']])

    # TODO: 練習- 請嘗試改成用L2距離來計算相似度