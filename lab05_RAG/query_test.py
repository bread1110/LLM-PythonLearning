"""
勞動基準法RAG查詢測試工具 - AI Agent 版本

此工具使用 function calling 實現 AI agent，能自動選擇適當的工具來回答用戶問題
支援向量搜索、統計查詢等多種功能
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Callable
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient
from openai import AzureOpenAI
from sentence_transformers import CrossEncoder

# 載入環境變數
load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))  # 初始化 Tavily 客戶端

def json_serializer(obj):
    """自定義 JSON 序列化器，處理 datetime 和其他不可序列化的物件"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class LaborLawAgent:
    """勞動基準法 AI Agent 系統"""
    
    def __init__(self):
        """初始化 AI Agent 系統"""
        # PostgreSQL連接配置
        self.db_config = {
            'host': os.getenv('PG_HOST', 'localhost'),
            'port': os.getenv('PG_PORT', '5432'),
            'database': os.getenv('PG_DATABASE', 'labor_law_rag'),
            'user': os.getenv('PG_USER', 'postgres'),
            'password': os.getenv('PG_PASSWORD', 'your_password')
        }
        
        # 初始化 Reranker 模型
        print("🔧 正在初始化 Reranker 模型...")
        try:
            # 使用多語言 Cross-Encoder 模型進行重新排序
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
            print("✅ Reranker 模型初始化成功")
        except Exception as e:
            print(f"❌ Reranker 模型初始化失敗: {e}")
            print("🔄 將使用原始向量搜索結果")
            self.reranker = None
        
        # 確保自定義函數存在
        self._ensure_similarity_function()
        
        # 初始化工具系統
        self._setup_tools()
    
    def _ensure_similarity_function(self):
        """確保cosine_similarity函數存在"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            
            # 創建cosine_similarity函數（如果不存在）
            create_function_sql = """
            CREATE OR REPLACE FUNCTION cosine_similarity(a double precision[], b double precision[])
            RETURNS double precision AS $$
            DECLARE
                dot_product double precision := 0;
                norm_a double precision := 0;
                norm_b double precision := 0;
                i integer;
            BEGIN
                -- 檢查向量長度是否相同
                IF array_length(a, 1) != array_length(b, 1) THEN
                    RETURN 0;
                END IF;
                
                -- 計算點積和範數
                FOR i IN 1..array_length(a, 1) LOOP
                    dot_product := dot_product + (a[i] * b[i]);
                    norm_a := norm_a + (a[i] * a[i]);
                    norm_b := norm_b + (b[i] * b[i]);
                END LOOP;
                
                -- 避免除以零
                IF norm_a = 0 OR norm_b = 0 THEN
                    RETURN 0;
                END IF;
                
                -- 返回餘弦相似度
                RETURN dot_product / (sqrt(norm_a) * sqrt(norm_b));
            END;
            $$ LANGUAGE plpgsql;
            """
            
            cur.execute(create_function_sql)
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            print(f"創建cosine_similarity函數時發生錯誤: {e}")
    
    def _setup_tools(self):
        """設置可用的工具和函數定義"""
        self.tools = {
            "vector_search": {
                "function": self._tool_vector_search,
                "description": "使用向量相似度搜索相關的勞動基準法條文。系統會先檢索15個相關結果，然後使用Reranker模型進行二次排序，最終返回前5個最相關的結果。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查詢，描述用戶想了解的法律問題"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "初始檢索結果數量限制，默認為15（最終會通過Reranker排序返回前5個）",
                            "default": 15
                        }
                    },
                    "required": ["query"]
                }
            },
            "web_search": {
                "function": self._tool_web_search,
                "description": "使用網路搜索獲取最新的法律資訊、相關新聞或其他補充資料",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索查詢，用於在網路上搜索相關資訊"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "返回結果數量限制，默認為5",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        }
        
        # 轉換為 OpenAI function calling 格式
        self.tool_definitions = []
        for name, tool in self.tools.items():
            self.tool_definitions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
    
    def _tool_vector_search(self, query: str, limit: int = 15) -> Dict[str, Any]:
        """工具：向量搜索"""
        print(f"🔍 執行向量搜索: '{query}'")
        
        # 生成查詢embedding
        query_embedding = self.query_aoai_embedding(query)
        if not query_embedding:
            return {"error": "無法生成查詢embedding"}
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 使用新版資料表結構進行向量搜索
            search_sql = """
            SELECT 
                id,
                content,
                created_at,
                -- 計算餘弦相似度
                cosine_similarity(embedding_vector, %s::double precision[]) as similarity,
                -- 計算文本長度
                length(content) as char_count
            FROM embeddings
            WHERE embedding_vector IS NOT NULL
            ORDER BY similarity DESC
            LIMIT %s;
            """
            
            cur.execute(search_sql, (query_embedding, limit))
            results = cur.fetchall()
            
            cur.close()
            conn.close()
            
            search_results = [dict(row) for row in results]
            print(f"✅ 找到 {len(search_results)} 個相關結果")
            
            # 使用 Reranker 進行二次排序，取前5名
            if search_results:
                print(f"🔄 對 {len(search_results)} 個結果進行 Reranker 二次排序...")
                reranked_results = self.rerank_results(query, search_results, top_k=5)
                print(f"✅ Reranker 完成，最終返回 {len(reranked_results)} 個結果")
                
                return {
                    "success": True,
                    "results": reranked_results,
                    "count": len(reranked_results),
                    "original_count": len(search_results),
                    "reranked": True
                }
            else:
                return {
                    "success": True,
                    "results": search_results,
                    "count": len(search_results),
                    "reranked": False
                }
            
        except Exception as e:
            error_msg = f"向量搜索時發生錯誤: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    
    def _tool_web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """工具：網路搜索"""
        print(f"🌐 執行網路搜索: '{query}'")
        
        try:
            # 使用 Tavily 客戶端進行搜索
            search_result = tavily_client.search(query, max_results=max_results)
            
            # 提取有用的搜索結果
            if search_result and 'results' in search_result:
                results = []
                for item in search_result['results']:
                    result_item = {
                        'title': item.get('title', ''),
                        'content': item.get('content', ''),
                        'url': item.get('url', ''),
                        'score': item.get('score', 0)
                    }
                    results.append(result_item)
                
                print(f"✅ 網路搜索找到 {len(results)} 個結果")
                
                return {
                    "success": True,
                    "results": results,
                    "count": len(results),
                    "query": query
                }
            else:
                return {"error": "網路搜索未返回有效結果"}
                
        except Exception as e:
            error_msg = f"網路搜索時發生錯誤: {e}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

     
    def rerank_results(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        使用 Reranker 模型對搜索結果進行二次排序
        
        Args:
            query (str): 查詢文字
            results (List[Dict]): 原始搜索結果
            top_k (int): 返回前k個結果
            
        Returns:
            List[Dict]: 重新排序後的前k個結果
        """
        if not self.reranker or not results:
            print("⚠️ Reranker 不可用或無搜索結果，返回原始結果")
            return results[:top_k]
        
        print(f"🔄 使用 Reranker 對 {len(results)} 個結果進行二次排序...")
        
        try:
            # 準備查詢-文檔對
            query_doc_pairs = []
            for result in results:
                content = result.get('content', '')
                # 限制文檔長度以提高效率
                if len(content) > 512:
                    content = content[:512] + "..."
                query_doc_pairs.append([query, content])
            
            # 使用 Reranker 計算相關性分數
            rerank_scores = self.reranker.predict(query_doc_pairs)
            
            # 將分數添加到結果中
            for i, result in enumerate(results):
                result['rerank_score'] = float(rerank_scores[i])
            
            # 根據 rerank 分數重新排序
            reranked_results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
            
            # 返回前k個結果
            top_results = reranked_results[:top_k]
            
            print(f"✅ Reranker 排序完成，返回前 {len(top_results)} 個結果")
            
            # 顯示排序結果摘要
            print("📊 Reranker 排序結果摘要:")
            for i, result in enumerate(top_results, 1):
                original_sim = result.get('similarity', 0)
                rerank_score = result.get('rerank_score', 0)
                print(f"  {i}. ID:{result.get('id', 'N/A')} | 原始相似度:{original_sim:.4f} | Rerank分數:{rerank_score:.4f}")
            
            return top_results
            
        except Exception as e:
            print(f"❌ Reranker 處理失敗: {e}")
            print("🔄 返回原始向量搜索結果")
            return results[:top_k]

    def chat_with_aoai_gpt(self, messages: list[dict], user_json_format: bool = False, 
                          tools: list = None, tool_choice: str = "auto"):
        """與 Azure OpenAI 服務互動的核心函數，支援 function calling

        Args:
            messages: 對話歷史列表
            user_json_format: 是否要求 JSON 格式回應
            tools: 可用的工具列表
            tool_choice: 工具選擇策略 ("auto", "none", 或指定工具名稱)

        Returns:
            tuple: (AI回應物件, 輸入token數, 輸出token數)
        """
        error_time = 0 # 錯誤次數
        temperature = 0.7 # 溫度
        
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

                # 準備 API 請求參數
                api_params = {
                    "model": aoai_model_version,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                # 設置回應格式
                if user_json_format:
                    api_params["response_format"] = {"type": "json_object"}
                
                # 設置工具調用
                if tools and len(tools) > 0:
                    api_params["tools"] = tools
                    api_params["tool_choice"] = tool_choice

                # 發送請求給 API
                aoai_response = client.chat.completions.create(**api_params)

                # 回傳完整的回應物件和token統計
                return (
                    aoai_response.choices[0].message,  # 返回完整的 message 物件，包含 tool_calls
                    aoai_response.usage.prompt_tokens,
                    aoai_response.usage.total_tokens - aoai_response.usage.prompt_tokens,
                )
                
            except Exception as e: # 如果發生錯誤
                print(f"❌ Azure OpenAI API 錯誤：{str(e)}")
                if error_time > 2:
                    # 如果重試次數用完，返回空的 message 物件
                    class EmptyMessage:
                        def __init__(self):
                            self.content = ""
                            self.tool_calls = None
                    return EmptyMessage(), 0, 0
                continue
        
        # 如果所有重試都失敗
        class EmptyMessage:
            def __init__(self):
                self.content = ""
                self.tool_calls = None
        return EmptyMessage(), 0, 0

    def query_aoai_embedding(self, content: str) -> list[float]:
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
                print(f"❌ Embedding API 錯誤：{e}")

        return []
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """執行指定的工具"""
        if tool_name not in self.tools:
            return {"error": f"未知的工具: {tool_name}"}
        
        try:
            tool_function = self.tools[tool_name]["function"]
            return tool_function(**kwargs)
        except Exception as e:
            return {"error": f"執行工具 {tool_name} 時發生錯誤: {e}"}
    
    def rewrite_query(self, original_query: str) -> str:
        """
        改寫和完善使用者的查詢，使其更適合進行向量搜索和工具調用
        
        Args:
            original_query (str): 原始使用者查詢
            
        Returns:
            str: 改寫後的查詢
        """
        print(f"✏️ 開始改寫查詢: '{original_query}'")
        
        rewrite_prompt = """你是一個專業的查詢改寫專家，專門處理勞動基準法相關問題。

你的任務是將使用者的原始查詢改寫成更清晰、更具體、更適合進行法條搜索的查詢。

改寫原則：
1. 保持原始意圖不變
2. 使查詢更具體和精確
3. 加入相關的法律術語和關鍵詞
4. 如果查詢模糊，可以將其拆分為多個明確的問題
5. 如果涉及多個概念，請明確說明
6. 使用繁體中文

請只回傳改寫後的查詢，不要添加任何解釋。

範例：
原始查詢：「加班費怎麼算？」
改寫後：「勞動基準法中加班費的計算方式和標準是什麼？包括平日加班、假日加班的費率規定。」

原始查詢：「可以隨便開除員工嗎？」
改寫後：「雇主解僱員工的法定程序和條件是什麼？勞動基準法對於資遣和解僱有哪些規定？」

現在請改寫以下查詢："""

        messages = [
            {"role": "system", "content": rewrite_prompt},
            {"role": "user", "content": original_query}
        ]
        
        try:
            # 調用 LLM 進行查詢改寫
            response, input_tokens, output_tokens = self.chat_with_aoai_gpt(
                messages, 
                user_json_format=False,
                tools=None,  # 改寫不需要工具
                tool_choice="none"
            )
            
            rewritten_query = response.content.strip() if hasattr(response, 'content') else original_query
            
            print(f"✅ 改寫完成")
            print(f"📝 原始查詢: {original_query}")
            print(f"🔄 改寫後: {rewritten_query}")
            print(f"📊 Token 使用 - 輸入: {input_tokens}, 輸出: {output_tokens}")
            
            return rewritten_query
            
        except Exception as e:
            print(f"❌ 查詢改寫失敗: {e}")
            print("🔄 使用原始查詢繼續處理")
            return original_query

    def generate_agent_response(self, user_question: str, max_iterations: int = 3) -> str:
        """
        使用 AI agent 生成回答，支援 function calling
        
        Args:
            user_question (str): 使用者問題
            max_iterations (int): 最大迭代次數
            
        Returns:
            str: AI agent 生成的回答
        """
        print(f"🤖 AI Agent 開始處理問題: '{user_question}'")
        
        # 步驟1：改寫和完善查詢
        print("\n📝 步驟1: 查詢改寫與完善")
        improved_query = self.rewrite_query(user_question)
        
        # 構建system prompt
        system_prompt = """你是一個專業的勞動基準法 AI 助手。你可以使用以下工具來回答用戶問題：

1. vector_search - 使用向量相似度搜索相關法條，系統會自動：
   - 先檢索15個相關結果
   - 使用Reranker模型進行二次排序
   - 返回前5個最相關的結果
   
2. web_search - 使用網路搜索獲取最新的法律資訊、相關新聞或其他補充資料


請根據用戶問題，選擇適當的工具獲取資訊，然後提供準確、實用的回答。

回答要求：
- 基於獲取的資料回答，不要添加沒有的內容
- 回答要清晰、具體、實用
- 使用專業但易懂的語言
- 回答以繁體中文進行
- 如果需要搜索法條，請使用 vector_search（已包含智能排序）
- 如果需要最新資訊、新聞或補充資料，請使用 web_search

工具使用策略：
- 對於法條條文查詢，優先使用 vector_search（系統已優化排序準確性）
- 對於最新政策、修法動態、實務案例，使用 web_search
- 可以組合使用多個工具來提供完整的回答"""

        # 初始化對話 - 使用改寫後的查詢
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用戶原始問題：{user_question}\n\n完善後的問題：{improved_query}"}
        ]
        
        print(f"\n🔧 步驟2: 開始工具調用和回答生成")
        
        for iteration in range(max_iterations):
            print(f"🔄 第 {iteration + 1} 次迭代...")
            
            try:
                # 調用 LLM，可能包含工具調用
                response, input_tokens, output_tokens = self.chat_with_aoai_gpt(
                    messages, 
                    user_json_format=False,
                    tools=self.tool_definitions,
                    tool_choice="auto"
                )
                
                print(f"📊 Token 使用 - 輸入: {input_tokens}, 輸出: {output_tokens}")
                
                # 檢查是否有工具調用
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"🔧 AI 選擇調用 {len(response.tool_calls)} 個工具")
                    
                    # 添加 assistant 消息
                    messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                }
                            } for tool_call in response.tool_calls
                        ]
                    })
                    
                    # 執行每個工具調用
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            tool_args = {}
                        
                        print(f"⚙️ 執行工具: {tool_name} 參數: {tool_args}")
                        
                        # 執行工具
                        tool_result = self.execute_tool(tool_name, **tool_args)
                        
                        # 添加工具結果到對話
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result, ensure_ascii=False, default=json_serializer)
                        })
                    
                    # 繼續下一次迭代，讓 AI 基於工具結果生成最終回答
                    continue
                    
                else:
                    # 沒有工具調用，返回最終回答
                    print("✅ AI Agent 完成回答")
                    return response.content if hasattr(response, 'content') else str(response)
                    
            except Exception as e:
                error_msg = f"AI Agent 處理錯誤: {e}"
                print(f"❌ {error_msg}")
                return f"抱歉，處理您的問題時發生錯誤：{error_msg}"
        
        return "抱歉，AI Agent 達到最大迭代次數，無法完成回答。"
    
    def display_llm_response(self, llm_response: str):
        """
        顯示LLM生成的回答
        
        Args:
            llm_response (str): LLM回答內容
        """
        if llm_response:
            print(f"\n{'🤖 AI Agent 回答':=^60}")
            print(llm_response)
            print("=" * 60)
        else:
            print("\n❌ 未能生成AI回答")
    
def main():
    """主程式 - AI Agent 互動式查詢介面"""
    print("🤖 勞動基準法 AI Agent 系統")
    print("=" * 50)
    print("本系統使用 AI Agent 技術，能自動選擇適當的工具來回答您的問題")
    print("支援功能：法條搜索、網路搜索、Reranker二次排序、智能問答")
    print("=" * 50)
    
    # 檢查並提示依賴套件
    print("🔧 正在初始化系統...")
    try:
        agent = LaborLawAgent()
    except ImportError as e:
        print(f"❌ 缺少必要的 Python 套件: {e}")
        print("💡 請執行以下命令安裝所需套件:")
        print("   pip install -r requirements.txt")
        print("   或手動安裝: pip install sentence-transformers")
        return
    except Exception as e:
        print(f"❌ 系統初始化失敗: {e}")
        print("💡 請檢查環境變數設定和資料庫連接")
        return
    
    while True:
        print("\n" + "="*60)
        print("🎯 AI Agent 查詢系統")
        print("💡 您可以詢問任何關於勞動基準法的問題")
        print("📚 法條查詢：工時規定、加班費計算、資遣相關法條等")
        print("🌐 網路搜索：最新修法動態、政策解釋、實務案例等")
        print("輸入 'exit' 以退出程式")
        print("=" * 60)
        
        query = input("請輸入您的問題: ").strip()
        if query.lower() == 'exit':
            print("感謝使用勞動基準法 AI Agent 系統！再見！")
            break
        
        if query:
            print("\n🚀 AI Agent 開始處理您的問題...")
            print("-" * 40)
            
            # 使用 AI Agent 處理問題
            try:
                # 使用 AI Agent 處理問題
                response = agent.generate_agent_response(query)
                # 顯示 AI Agent 的回答
                agent.display_llm_response(response)
            except Exception as e:
                print(f"❌ 處理問題時發生錯誤: {e}")
                print("💡 建議：請重新表述您的問題或稍後再試")

if __name__ == "__main__":
    main() 