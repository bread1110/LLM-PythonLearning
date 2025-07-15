"""
勞動基準法RAG查詢測試工具 - 簡化版本
只使用向量搜索 + 強力繁體中文Reranker模型
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
from utils.database_config import get_database_config
from utils.ai_client import get_embedding_for_content, chat_with_azure_openai
from sentence_transformers import CrossEncoder
import concurrent.futures
import time

# 載入環境變數
load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class ChineseReranker:
    """繁體中文專用 Reranker 模型"""
    
    def __init__(self):
        """初始化繁體中文Reranker模型"""
        self.model = None
        
        # 使用BAAI的BGE Reranker base - 平衡速度與質量，支援繁體中文
        model_config = {
            'name': 'bge-reranker-base',
            'model_path': 'BAAI/bge-reranker-base'
        }
        
        try:
            print(f"🔧 正在加載 {model_config['name']} 繁體中文Reranker模型...")
            self.model = CrossEncoder(model_config['model_path'])
            print(f"✅ {model_config['name']} 模型加載成功")
        except Exception as e:
            print(f"❌ {model_config['name']} 模型加載失敗: {e}")
            print("🔄 嘗試備用模型...")
            try:
                # 備用模型1：更小的BGE模型
                print("🔧 嘗試加載 bge-reranker-base 備用模型...")
                self.model = CrossEncoder('BAAI/bge-reranker-base')
                print("✅ BGE base 備用模型加載成功")
            except Exception as e2:
                print(f"❌ BGE 備用模型加載失敗: {e2}")
                try:
                    # 備用模型2：輕量級模型
                    print("🔧 嘗試加載輕量級備用模型...")
                    self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                    print("✅ 輕量級備用模型加載成功")
                except Exception as e3:
                    print(f"❌ 所有備用模型都無法加載: {e3}")
                    print("⚠️ 將使用無rerank模式")
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """使用繁體中文Reranker進行排序"""
        if not self.model or not results:
            return results[:top_k]
        
        print(f"🎯 使用繁體中文Reranker對 {len(results)} 個結果進行重排序...")
        
        # 開始計時
        start_time = time.time()
        
        try:
            # 準備查詢-文檔對
            prep_start = time.time()
            query_doc_pairs = []
            for result in results:
                content = result.get('content', '')
                # 限制文本長度以提升性能
                if len(content) > 512:
                    content = content[:512] + "..."
                query_doc_pairs.append([query, content])
            prep_time = time.time() - prep_start
            
            # 使用模型評分
            predict_start = time.time()
            scores = self.model.predict(query_doc_pairs)
            predict_time = time.time() - predict_start
            
            # 添加rerank分數到結果並排序
            sort_start = time.time()
            for i, result in enumerate(results):
                result['rerank_score'] = float(scores[i]) if i < len(scores) else 0.0
            
            # 根據rerank分數排序
            sorted_results = sorted(results, key=lambda x: x.get('rerank_score', 0), reverse=True)
            top_results = sorted_results[:top_k]
            sort_time = time.time() - sort_start
            
            # 總計時
            total_time = time.time() - start_time
            
            print(f"✅ Reranker完成，返回前 {len(top_results)} 個結果")
            print(f"⏱️ Reranker 計時統計:")
            print(f"   - 數據準備: {prep_time:.3f}秒")
            print(f"   - 模型推理: {predict_time:.3f}秒") 
            print(f"   - 結果排序: {sort_time:.3f}秒")
            print(f"   - 總處理時間: {total_time:.3f}秒")
            print(f"   - 平均每個結果: {total_time/len(results):.4f}秒")
            
            self._display_results(top_results)
            
            return top_results
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ Reranker失敗: {e} (耗時: {total_time:.3f}秒)")
            return results[:top_k]
    
    def _display_results(self, results: List[Dict]):
        """顯示rerank結果摘要"""
        print("📊 Reranker結果摘要:")
        for i, result in enumerate(results, 1):
            rerank_score = result.get('rerank_score', 0)
            print(f"  {i}. ID:{result.get('id', 'N/A')} | Rerank分數:{rerank_score:.4f}")

class LaborLawAgent:
    """勞動基準法 AI Agent 系統 - 簡化版"""
    
    def __init__(self):
        """初始化 AI Agent 系統"""
        # PostgreSQL連接配置
        self.db_config = get_database_config()
        
        # 初始化繁體中文 Reranker 系統
        print("🔧 正在初始化繁體中文 Reranker 系統...")
        self.reranker = ChineseReranker()
        
        # 初始化工具系統
        self._setup_tools()
    
    def _setup_tools(self):
        """設置可用的工具和函數定義"""
        self.tools = {
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
            },
            "vector_search": {
                "function": self._tool_vector_search,
                "description": "使用語義向量搜索查找相關的勞動基準法條文和規定，自動使用繁體中文Reranker模型重新排序結果",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "自然語言查詢，描述您想了解的勞基法相關問題"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "返回結果數量，默認為15",
                            "default": 15
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
        """工具：向量搜索 + 繁體中文Reranker"""
        print(f"🔍 執行向量搜索: '{query}'")
        
        # 生成查詢embedding
        query_embedding = self.query_aoai_embedding(query)
        if not query_embedding:
            return {"error": "無法生成查詢embedding"}
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # 向量搜索SQL
            search_sql = """
            SELECT 
                id,
                content,
                created_at,
                cosine_similarity(embedding_vector, %s::double precision[]) as similarity,
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
            
            # 使用繁體中文 Reranker 進行重排序
            if search_results:
                print(f"🔄 對 {len(search_results)} 個結果進行繁體中文Reranker排序...")
                
                reranked_results = self.reranker.rerank(query, search_results, top_k=5)
                
                print(f"✅ 繁體中文Reranker完成，最終返回 {len(reranked_results)} 個結果")
                
                return {
                    "success": True,
                    "results": reranked_results,
                    "count": len(reranked_results),
                    "original_count": len(search_results),
                    "reranked": True,
                    "reranking_method": "chinese_reranker"
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

    def chat_with_aoai_gpt(self, messages: List[Dict], tools: List[Dict] = None) -> tuple:
        """與 Azure OpenAI GPT 進行對話"""
        return chat_with_azure_openai(messages, tools)

    def query_aoai_embedding(self, content: str) -> list[float]:
        """從 Azure OpenAI 服務獲取文本的 embedding 向量"""
        return get_embedding_for_content(content)
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """執行指定的工具"""
        if tool_name not in self.tools:
            return {"error": f"未知的工具: {tool_name}"}
        
        try:
            tool_function = self.tools[tool_name]["function"]
            return tool_function(**kwargs)
        except Exception as e:
            return {"error": f"工具執行失敗: {e}"}
    
    def execute_tools_concurrently(self, tool_calls: List) -> List[Dict[str, Any]]:
        """並行執行多個工具（同步版本）"""
        print(f"🚀 開始並行執行 {len(tool_calls)} 個工具...")
        
        # 使用 ThreadPoolExecutor 來並行執行工具
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # 準備任務
            futures = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # 提交任務到執行器
                future = executor.submit(self.execute_tool, function_name, **function_args)
                futures.append((tool_call, future))
                
                print(f"📋 已提交工具任務: {function_name} 參數: {function_args}")
            
            # 等待所有任務完成並收集結果
            results = []
            for tool_call, future in futures:
                try:
                    result = future.result(timeout=60)  # 60秒超時
                    results.append({
                        'tool_call': tool_call,
                        'result': result,
                        'success': True
                    })
                    print(f"✅ 工具 {tool_call.function.name} 執行完成")
                except concurrent.futures.TimeoutError:
                    error_result = {"error": f"工具 {tool_call.function.name} 執行超時"}
                    results.append({
                        'tool_call': tool_call,
                        'result': error_result,
                        'success': False
                    })
                    print(f"⏱️ 工具 {tool_call.function.name} 執行超時")
                except Exception as e:
                    error_result = {"error": f"工具執行異常: {e}"}
                    results.append({
                        'tool_call': tool_call,
                        'result': error_result,
                        'success': False
                    })
                    print(f"❌ 工具 {tool_call.function.name} 執行失敗: {e}")
            
            print(f"🎯 所有工具執行完成，成功: {sum(1 for r in results if r['success'])}/{len(results)}")
            return results

    def rewrite_query(self, user_question: str) -> str:
        """改寫和完善用戶查詢"""
        rewrite_messages = [
            {
                "role": "system", 
                "content": """你是一個專業的查詢改寫專家。請將用戶的問題改寫成更適合搜索的完整查詢。

改寫原則：
1. 保持原意不變
2. 補充相關的法律術語
3. 使查詢更具體和準確
4. 適合向量搜索和語義理解

範例：
用戶問題：「加班費怎麼算？」
改寫結果：「勞動基準法加班費計算方式 平日延長工時費率 假日工作報酬標準」

請只返回改寫後的查詢，不要包含其他說明。"""
            },
            {"role": "user", "content": f"請改寫這個問題：{user_question}"}
        ]
        
        try:
            message, _, _ = self.chat_with_aoai_gpt(rewrite_messages)
            improved_query = message.content.strip() if message.content else user_question
            print(f"📝 查詢改寫: '{user_question}' → '{improved_query}'")
            return improved_query
        except Exception as e:
            print(f"⚠️ 查詢改寫失敗: {e}")
            return user_question

    def generate_agent_response(self, user_question: str, conversation_history: List[Dict[str, str]] = None) -> str:
        """生成 AI Agent 回應"""
        print(f"🤖 AI Agent 開始處理問題: '{user_question}'")
        
        # 步驟1：改寫和完善查詢
        print("\n📝 步驟1: 查詢改寫與完善")
        improved_query = self.rewrite_query(user_question)
        
        # 構建system prompt
        system_prompt = """你是一個專業的勞動基準法 AI 助手。你可以使用以下工具來回答用戶問題：

1. vector_search - 向量搜索功能（主要工具）：
   - 使用語義理解查找相關的法條和規定
   - 自動使用繁體中文Reranker模型重新排序結果
   - 適用於所有法條相關查詢
   
2. web_search - 網路搜索功能：
   - 使用網路搜索獲取最新的法律資訊
   - 查找相關新聞、政策解釋、實務案例

回答要求：
1. 優先使用vector_search查找法條依據
2. 如需要最新資訊才使用web_search
3. 回答要準確、專業、易懂
4. 引用具體法條條文
5. 提供實務建議
6. 根據對話歷史提供連貫的回答"""

        # 初始化對話
        messages = [{"role": "system", "content": system_prompt}]
        
        # 加入對話歷史
        if conversation_history:
            print(f"📚 載入 {len(conversation_history)} 條對話歷史")
            for msg in conversation_history:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # 加入當前問題
        messages.append({"role": "user", "content": improved_query})
        
        # AI Agent 迭代處理
        max_iterations = 5
        for iteration in range(max_iterations):
            print(f"\n🔄 AI Agent 迭代 {iteration + 1}/{max_iterations}")
            
            try:
                # 呼叫 GPT 並傳遞工具定義
                message, input_tokens, output_tokens = self.chat_with_aoai_gpt(
                    messages, 
                    self.tool_definitions
                )
                
                print(f"📊 Token使用: 輸入={input_tokens}, 輸出={output_tokens}")
                
                # 添加助手回應到對話歷史
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [tool_call.__dict__ if hasattr(tool_call, '__dict__') else tool_call 
                                 for tool_call in message.tool_calls] if message.tool_calls else None
                })
                
                # 檢查是否需要執行工具
                if message.tool_calls:
                    print(f"🔧 需要執行 {len(message.tool_calls)} 個工具")
                    
                    if len(message.tool_calls) > 1:
                        # 多個工具 - 使用並行執行
                        print("🚀 檢測到多個工具，使用並行執行模式...")
                        
                        # 運行並行工具執行
                        tool_results = self.execute_tools_concurrently(message.tool_calls)
                        
                        # 將結果添加到對話歷史
                        for tool_result_info in tool_results:
                            tool_call = tool_result_info['tool_call']
                            tool_result = tool_result_info['result']
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(tool_result, ensure_ascii=False, default=str)
                            })
                    else:
                        # 單個工具 - 使用傳統順序執行
                        print("🔧 單個工具，使用順序執行模式...")
                        tool_call = message.tool_calls[0]
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        print(f"⚙️ 執行工具: {function_name} 參數: {function_args}")
                        
                        # 執行工具
                        tool_result = self.execute_tool(function_name, **function_args)
                        
                        # 添加工具結果到對話
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_result, ensure_ascii=False, default=str)
                        })
                        
                        print(f"✅ 工具 {function_name} 執行完成")
                else:
                    # 沒有工具調用，返回最終回答
                    if message.content:
                        print(f"🎯 AI Agent 完成回答")
                        return message.content
                    
            except Exception as e:
                error_msg = f"AI Agent 處理錯誤: {e}"
                print(f"❌ {error_msg}")
                return f"抱歉，處理您的問題時發生錯誤：{error_msg}"
        
        return "抱歉，AI Agent 達到最大迭代次數，無法完成回答。"
    
    def display_llm_response(self, llm_response: str):
        """顯示LLM生成的回答"""
        if llm_response:
            print("\n" + "="*60)
            print("🤖 AI Agent 回答:")
            print("="*60)
            print(llm_response)
            print("="*60)
        else:
            print("❌ 未獲得有效回答")

def main():
    """主程序"""
    print("🚀 勞動基準法 AI Agent 系統 (簡化版) 啟動中...")
    
    try:
        # 初始化 AI Agent
        agent = LaborLawAgent()
        print("✅ 系統初始化完成")
    except ImportError as e:
        print(f"❌ 缺少必要的 Python 套件: {e}")
        print("💡 請執行以下命令安裝所需套件:")
        print("   pip install sentence-transformers")
        return
    except Exception as e:
        print(f"❌ 系統初始化失敗: {e}")
        print("💡 請檢查環境變數設定和資料庫連接")
        return
    
    while True:
        print("\n" + "="*60)
        print("🎯 AI Agent 查詢系統 (向量搜索 + 繁體中文Reranker)")
        print("💡 您可以詢問任何關於勞動基準法的問題")
        print("📚 法條查詢：工時規定、加班費計算、資遣相關法條等")
        print("🎯 特色功能：語義向量搜索 + 繁體中文Reranker排序")
        print("🌐 網路搜索：最新修法動態、政策解釋、實務案例等")
        print()
        print("輸入 'exit' 以退出程式")
        print("=" * 60)
        
        query = input("請輸入您的問題: ").strip()
        if query.lower() == 'exit':
            print("感謝使用勞動基準法 AI Agent 系統！再見！")
            break
        
        if query:
            print("\n🚀 AI Agent 開始處理您的問題...")
            print("-" * 40)
            
            try:
                response = agent.generate_agent_response(query)
                agent.display_llm_response(response)
            except Exception as e:
                print(f"❌ 處理問題時發生錯誤: {e}")
                print("💡 建議：請重新表述您的問題或稍後再試")

if __name__ == "__main__":
    main()