"""
勞動基準法 RAG API Server
使用 FastAPI 提供 RESTful API 服務
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# 導入現有的 RAG 系統
from query_test import LaborLawAgent
from utils.tracking_utils import execute_query_with_tracking

# 全局變數
labor_agent: Optional[LaborLawAgent] = None

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    # 啟動時初始化
    global labor_agent
    print("🚀 正在初始化勞動基準法 RAG 系統...")
    try:
        labor_agent = LaborLawAgent()
        print("✅ RAG 系統初始化完成")
    except Exception as e:
        print(f"❌ RAG 系統初始化失敗: {e}")
        raise
    
    yield
    
    # 關閉時清理
    print("🔄 正在關閉 RAG 系統...")
    labor_agent = None

# 創建 FastAPI 應用
app = FastAPI(
    title="勞動基準法 RAG API",
    description="提供勞動基準法智能查詢服務的 RESTful API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 設置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React 開發服務器
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Pydantic 模型定義 ===

class ChatMessage(BaseModel):
    """對話訊息模型"""
    role: str = Field(..., description="訊息角色：user 或 assistant")
    content: str = Field(..., description="訊息內容")

class QueryRequest(BaseModel):
    """查詢請求模型"""
    question: str = Field(..., description="用戶問題", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="會話 ID")
    include_technical_details: bool = Field(True, description="是否包含技術細節")
    messages: Optional[List[ChatMessage]] = Field([], description="對話歷史")

class SearchResult(BaseModel):
    """搜索結果模型"""
    id: int
    content: str
    hybrid_score: Optional[float] = None
    ensemble_score: Optional[float] = None
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    source: Optional[str] = None

class UsedChunk(BaseModel):
    """使用的chunk資訊模型"""
    id: int
    content: str
    full_content: str
    rerank_score: Optional[float] = None
    similarity: Optional[float] = None
    hybrid_score: Optional[float] = None
    ensemble_score: Optional[float] = None
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    source: str
    used_in_response: bool = True

class TechnicalDetails(BaseModel):
    """技術細節模型"""
    search_metadata: Optional[Dict[str, Any]] = None
    hybrid_results: Optional[List[SearchResult]] = None
    web_results: Optional[List[Dict[str, Any]]] = None
    used_chunks: Optional[List[UsedChunk]] = None
    token_usage: Optional[Dict[str, int]] = None

class QueryResponse(BaseModel):
    """查詢回應模型"""
    answer: str = Field(..., description="AI 回答")
    session_id: str = Field(..., description="會話 ID")
    timestamp: str = Field(..., description="回應時間戳")
    technical_details: Optional[TechnicalDetails] = None
    processing_time: float = Field(..., description="處理時間（秒）")

class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str
    timestamp: str
    version: str
    system_info: Dict[str, Any]

class ErrorResponse(BaseModel):
    """錯誤回應模型"""
    error: str
    detail: Optional[str] = None
    timestamp: str

# === WebSocket 連接管理 ===

class ConnectionManager:
    """WebSocket 連接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔗 WebSocket 連接建立，總連接數: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"🔌 WebSocket 連接斷開，總連接數: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message, ensure_ascii=False))

manager = ConnectionManager()

# 使用共用的技術細節追蹤器

async def query_with_technical_details(agent, query: str, conversation_history: List[Dict[str, str]] = None) -> Tuple[str, TechnicalDetails]:
    """執行查詢並收集技術細節"""
    
    # 使用共用的追蹤功能
    response, technical_details_dict = execute_query_with_tracking(agent, query, conversation_history)
    
    # 轉換為 API 所需的格式
    technical_details = TechnicalDetails(
        search_metadata=technical_details_dict.get('search_metadata'),
        hybrid_results=[
            SearchResult(
                id=result.get('id', 0),
                content=result.get('content', ''),
                hybrid_score=result.get('hybrid_score'),
                ensemble_score=result.get('ensemble_score'),
                vector_score=result.get('vector_score'),
                keyword_score=result.get('keyword_score'),
                source=result.get('source')
            ) for result in (technical_details_dict.get('hybrid_results', []))
        ] if technical_details_dict.get('hybrid_results') else None,
        web_results=technical_details_dict.get('web_results'),
        used_chunks=[
            UsedChunk(
                id=chunk.get('id', 0),
                content=chunk.get('content', ''),
                full_content=chunk.get('full_content', ''),
                rerank_score=chunk.get('rerank_score'),
                similarity=chunk.get('similarity'),
                hybrid_score=chunk.get('hybrid_score'),
                ensemble_score=chunk.get('ensemble_score'),
                vector_score=chunk.get('vector_score'),
                keyword_score=chunk.get('keyword_score'),
                source=chunk.get('source', ''),
                used_in_response=chunk.get('used_in_response', True)
            ) for chunk in technical_details_dict.get('used_chunks', [])
        ] if technical_details_dict.get('used_chunks') else None,
        token_usage=technical_details_dict.get('token_usage', {})
    )
    
    return response, technical_details

# === API 路由 ===

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路徑"""
    return {
        "message": "勞動基準法 RAG API Server",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    global labor_agent
    
    system_info = {
        "agent_initialized": labor_agent is not None,
        "python_version": os.sys.version,
        "api_server": "FastAPI"
    }
    
    if labor_agent:
        try:
            # 簡單測試查詢以確認系統運作
            # 檢查reranker類型並獲取相關資訊
            if hasattr(labor_agent, 'reranker_ensemble'):
                system_info["reranker_models"] = len(labor_agent.reranker_ensemble.models)
            elif hasattr(labor_agent, 'reranker'):
                system_info["reranker_models"] = 1  # 單一reranker模型
                system_info["reranker_type"] = "chinese_reranker"
            else:
                system_info["reranker_models"] = 0
            system_info["available_tools"] = list(labor_agent.tools.keys())
        except Exception as e:
            system_info["system_error"] = str(e)
    
    return HealthResponse(
        status="healthy" if labor_agent else "unhealthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        system_info=system_info
    )

@app.post("/query", response_model=QueryResponse)
async def query_labor_law(request: QueryRequest):
    """查詢勞動基準法"""
    global labor_agent
    
    if not labor_agent:
        raise HTTPException(status_code=503, detail="RAG 系統未初始化")
    
    start_time = datetime.now()
    
    try:
        # 生成會話 ID
        session_id = request.session_id or f"session_{int(start_time.timestamp())}"
        
        # 執行查詢
        print(f"🔍 處理查詢: {request.question[:50]}...")
        
        # 轉換前端傳來的對話歷史格式
        conversation_history = []
        if request.messages:
            for msg in request.messages:
                conversation_history.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # 如果需要技術細節，使用追蹤器
        if request.include_technical_details:
            answer, technical_details = await query_with_technical_details(labor_agent, request.question, conversation_history)
        else:
            answer = labor_agent.generate_agent_response(request.question, conversation_history)
            technical_details = None
        
        # 計算處理時間
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 構建回應
        response = QueryResponse(
            answer=answer,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            technical_details=technical_details
        )
        
        print(f"✅ 查詢完成，處理時間: {processing_time:.2f}秒")
        return response
        
    except Exception as e:
        print(f"❌ 查詢處理失敗: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"查詢處理失敗: {str(e)}"
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端點，支援即時查詢"""
    await manager.connect(websocket)
    
    try:
        while True:
            # 接收客戶端消息
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                question = message.get("question", "")
                
                if not question:
                    await manager.send_personal_message({
                        "error": "問題不能為空"
                    }, websocket)
                    continue
                
                # 發送處理中狀態
                await manager.send_personal_message({
                    "type": "processing",
                    "message": "正在處理您的問題..."
                }, websocket)
                
                # 處理查詢
                start_time = datetime.now()
                
                # 處理對話歷史
                conversation_history = []
                if "messages" in message and message["messages"]:
                    for msg in message["messages"]:
                        conversation_history.append({
                            "role": msg.get("role", ""),
                            "content": msg.get("content", "")
                        })
                
                # 檢查是否需要技術細節
                include_details = message.get("include_technical_details", False)
                
                if include_details:
                    answer, technical_details = await query_with_technical_details(labor_agent, question, conversation_history)
                    # 將技術細節轉換為可序列化的格式
                    details_dict = technical_details.dict() if technical_details else None
                else:
                    answer = labor_agent.generate_agent_response(question, conversation_history)
                    details_dict = None
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                # 發送結果
                response_data = {
                    "type": "response",
                    "answer": answer,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                if details_dict:
                    response_data["technical_details"] = details_dict
                
                await manager.send_personal_message(response_data, websocket)
                
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "error": "無效的 JSON 格式"
                }, websocket)
            except Exception as e:
                await manager.send_personal_message({
                    "error": f"處理失敗: {str(e)}"
                }, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 異常處理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """通用異常處理器"""
    print(f"🚨 未處理的異常: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="內部服務器錯誤",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# === 主程序 ===

def main():
    """啟動 API 服務器"""
    print("🚀 啟動勞動基準法 RAG API Server...")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()