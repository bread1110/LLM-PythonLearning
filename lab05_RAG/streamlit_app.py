"""
勞動基準法 RAG 系統 Streamlit UI
基於 lab04 優化的進階版本，整合 AI Agent 功能
"""

import streamlit as st

# 必須首先設定頁面配置
st.set_page_config(
    page_title="勞動基準法 RAG 查詢系統",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 完整的深色主題CSS配色
st.markdown("""
<style>
    /* === 全局基礎樣式 === */
    .stApp {
        background-color: #141414 !important;
        color: #f0f0f0 !important;
    }
    
    /* 主要內容區域 */
    .main .block-container {
        background-color: #141414 !important;
        color: #f0f0f0 !important;
        padding-top: 1rem !important;
    }
    
    /* === 側邊欄樣式 === */
    .css-1d391kg, .css-1y4p8pa, .css-17eq0hr, section[data-testid="stSidebar"] {
        background-color: #262626 !important;
    }
    
    .css-1d391kg .css-1y4p8pa, section[data-testid="stSidebar"] > div {
        background-color: #262626 !important;
        color: #f0f0f0 !important;
    }
    
    /* 側邊欄文字 */
    section[data-testid="stSidebar"] * {
        color: #f0f0f0 !important;
    }
    
    /* === 按鈕樣式 === */
    .stButton > button,
    button[kind="secondary"],
    button[kind="primary"],
    .css-1cpxqw2,
    .css-19rxjzo,
    .css-bva2kr,
    .css-1x8cf1d,
    button {
        background-color: #262626 !important;
        color: #d9d9d9 !important;
        border: 1px solid #404040 !important;
        border-radius: 6px !important;
    }
    
    .stButton > button:hover,
    button[kind="secondary"]:hover,
    button[kind="primary"]:hover,
    .css-1cpxqw2:hover,
    .css-19rxjzo:hover,
    .css-bva2kr:hover,
    .css-1x8cf1d:hover,
    button:hover {
        background-color: #404040 !important;
        border-color: #595959 !important;
        color: #ffffff !important;
    }
    
    /* 強制覆蓋白底黑字的反白效果 */
    .stButton > button:hover *,
    button:hover *,
    button[kind="secondary"]:hover *,
    button[kind="primary"]:hover *,
    button:hover span,
    button:hover div,
    .stButton:hover *,
    .stButton:hover span,
    .stButton:hover div {
        background-color: transparent !important;
        color: #ffffff !important;
    }
    
    /* 修正按鈕反白時的文字對比度問題 */
    button:hover,
    .stButton:hover button,
    button[data-testid]:hover {
        background-color: #404040 !important;
        color: #ffffff !important;
    }
    
    /* 確保按鈕點擊狀態也保持良好對比度 */
    button:active,
    button:focus,
    .stButton:active button,
    .stButton:focus button {
        background-color: #595959 !important;
        color: #ffffff !important;
    }
    
    .stButton > button:active, 
    .stButton > button:focus,
    button[kind="secondary"]:active,
    button[kind="secondary"]:focus,
    button[kind="primary"]:active,
    button[kind="primary"]:focus,
    .css-1cpxqw2:active,
    .css-1cpxqw2:focus,
    .css-19rxjzo:active,
    .css-19rxjzo:focus,
    .css-bva2kr:active,
    .css-bva2kr:focus,
    .css-1x8cf1d:active,
    .css-1x8cf1d:focus,
    button:active,
    button:focus {
        background-color: #404040 !important;
        border-color: #595959 !important;
        color: #ffffff !important;
        box-shadow: none !important;
    }
    
    /* === 輸入框樣式 === */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        background-color: #262626 !important;
        color: #d9d9d9 !important;
        border: 1px solid #404040 !important;
        border-radius: 6px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #1677ff !important;
        box-shadow: 0 0 0 2px rgba(22, 119, 255, 0.2) !important;
    }
    
    /* 聊天輸入框 - 所有可能的選擇器 */
    .stChatInput textarea,
    .stChatInput > div textarea,
    .stChatInput > div > div textarea,
    .stChatInput > div > div > div textarea,
    .stChatInput > div > div > div > div textarea,
    .stChatInput > div > div > div > div > div > textarea,
    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] > div textarea,
    div[data-testid="stChatInput"] > div > div textarea {
        background-color: #262626 !important;
        color: #d9d9d9 !important;
        border: 1px solid #404040 !important;
        border-radius: 6px !important;
    }
    
    /* 聊天輸入框容器 */
    .stChatInput,
    div[data-testid="stChatInput"],
    .stChatInput > div,
    div[data-testid="stChatInput"] > div {
        background-color: #141414 !important;
        border: none !important;
    }
    
    /* === 聊天訊息樣式 === */
    .stChatMessage {
        background-color: #1f1f1f !important;
        border: 1px solid #303030 !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
    }
    
    .stChatMessage .stMarkdown {
        color: #f0f0f0 !important;
    }
    
    .stChatMessage .stMarkdown p {
        color: #d9d9d9 !important;
    }
    
    /* === Metric 樣式 === */
    .metric-container,
    .css-1r6slb0,
    div[data-testid="metric-container"] {
        background-color: #1f1f1f !important;
        padding: 12px !important;
        border-radius: 6px !important;
        border: 1px solid #303030 !important;
    }
    
    div[data-testid="metric-container"] > div {
        color: #f0f0f0 !important;
    }
    
    div[data-testid="metric-container"] > div > div {
        color: #d9d9d9 !important;
    }
    
    /* === Expander 樣式 === */
    .streamlit-expanderHeader {
        background-color: #1f1f1f !important;
        color: #f0f0f0 !important;
        border: 1px solid #303030 !important;
        border-radius: 6px !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1f1f1f !important;
        border: 1px solid #303030 !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }
    
    /* === 標題和文字樣式 === */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    .stMarkdown, .stText, p {
        color: #d9d9d9 !important;
    }
    
    strong {
        color: #ffffff !important;
    }
    
    /* === Success/Info/Warning/Error 樣式 === */
    .stSuccess {
        background-color: #1f4c2e !important;
        color: #73d13d !important;
        border: 1px solid #389e0d !important;
    }
    
    .stInfo {
        background-color: #003a8c !important;
        color: #69c0ff !important;
        border: 1px solid #1677ff !important;
    }
    
    .stWarning {
        background-color: #614700 !important;
        color: #faad14 !important;
        border: 1px solid #d48806 !important;
    }
    
    .stError {
        background-color: #5c0011 !important;
        color: #ff4d4f !important;
        border: 1px solid #cf1322 !important;
    }
    
    /* === Checkbox 樣式 === */
    .stCheckbox > label {
        color: #d9d9d9 !important;
    }
    
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] {
        color: #d9d9d9 !important;
    }
    
    /* === Divider 樣式 === */
    .stDivider > div {
        border-color: #404040 !important;
    }
    
    hr {
        border-color: #404040 !important;
    }
    
    /* === DataFrame/Table 樣式 === */
    .stDataFrame {
        background-color: #1f1f1f !important;
    }
    
    .stDataFrame table {
        background-color: #1f1f1f !important;
        color: #d9d9d9 !important;
    }
    
    .stDataFrame th {
        background-color: #262626 !important;
        color: #ffffff !important;
        border-bottom: 1px solid #404040 !important;
    }
    
    .stDataFrame td {
        background-color: #1f1f1f !important;
        color: #d9d9d9 !important;
        border-bottom: 1px solid #404040 !important;
    }
    
    /* === Spinner 樣式 === */
    .stSpinner > div > div {
        border-top-color: #69c0ff !important;
    }
    
    /* === Progress Bar 樣式 === */
    .stProgress > div > div > div {
        background-color: #69c0ff !important;
    }
    
    /* === 滾動條樣式 === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1f1f1f;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #404040;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #595959;
    }
    
    /* === 隱藏Streamlit品牌 === */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {visibility: hidden;}
    
    /* === 容器和卡片樣式 === */
    .element-container {
        background-color: transparent !important;
    }
    
    .stContainer {
        background-color: #1f1f1f !important;
        border: 1px solid #303030 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    
    /* === 修正特定白色背景元素 === */
    
    /* 聊天輸入區域底部容器 */
    .stBottom,
    .stBottom > div,
    div[data-testid="stBottom"],
    div[data-testid="stBottom"] > div,
    .css-1544g2n,
    .css-1y4p8pa {
        background-color: #141414 !important;
    }
    
    /* 範例按鈕和容器 */
    .stButton,
    .stButton > div,
    .element-container div {
        background-color: transparent !important;
    }
    
    /* 修正input和textarea的父容器 */
    .stTextInput,
    .stTextArea,
    .stTextInput > div,
    .stTextArea > div,
    .stChatInput,
    .stChatInput > div {
        background-color: transparent !important;
    }
    
    /* 特殊處理聊天輸入框的所有層級 */
    .stChatInput *,
    div[data-testid="stChatInput"] *,
    .css-1y4p8pa * {
        background-color: transparent !important;
    }
    
    /* 但保留實際的輸入框背景 */
    .stChatInput textarea,
    div[data-testid="stChatInput"] textarea {
        background-color: #262626 !important;
    }
    
    /* === 響應式調整 === */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

# 導入本地模塊
from query_test import LaborLawAgent
from utils.tracking_utils import execute_query_with_tracking

# 載入環境變數
load_dotenv()

class RAGStreamlitApp:
    """RAG 系統的 Streamlit 應用程式"""
    
    def __init__(self):
        """初始化應用程式"""
        self.initialize_session_state()
        self.load_agent()
    
    def initialize_session_state(self):
        """初始化 session state"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        if "show_technical_details" not in st.session_state:
            st.session_state.show_technical_details = True  # 始終顯示技術細節
        
        if "total_input_tokens" not in st.session_state:
            st.session_state.total_input_tokens = 0
        
        if "total_output_tokens" not in st.session_state:
            st.session_state.total_output_tokens = 0
        
        if "query_count" not in st.session_state:
            st.session_state.query_count = 0
    
    def load_agent(self):
        """載入 RAG Agent"""
        try:
            if "agent" not in st.session_state:
                with st.spinner("🔧 正在初始化 AI Agent 系統..."):
                    st.session_state.agent = LaborLawAgent()
                st.success("✅ AI Agent 系統初始化完成！")
        except Exception as e:
            st.error(f"❌ AI Agent 初始化失敗: {e}")
            st.error("請檢查環境變數設定和資料庫連接")
            st.stop()
    
    def render_sidebar(self):
        """渲染側邊欄 - 仿照React布局"""
        with st.sidebar:
            # Header區域 - 仿照React的Header
            st.markdown("""
            <div style="background-color: #1a1a1a; padding: 16px; margin: -16px -16px 16px -16px; border-bottom: 1px solid #303030;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 24px;">🤖</span>
                    <span style="color: #ffffff; font-size: 16px; font-weight: bold;">勞動基準法 RAG 查詢系統</span>
                </div>
                <div style="margin-top: 8px;">
                    <span style="background: #003a8c; color: #69c0ff; padding: 2px 8px; border-radius: 4px; font-size: 12px;">v2.0</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 系統狀態卡片 - 仿照React的SystemStatus
            st.markdown("### 📊 系統狀態")
            
            # 系統運行狀態
            if st.session_state.agent:
                st.markdown("✅ **系統狀態：** 正常運行")
                st.markdown("🔧 **Reranker模型：** 1個")
            else:
                st.markdown("❌ **系統狀態：** 初始化失敗")
            
            st.divider()
            
            # 查詢統計 - 仿照React的圖標 + 數字布局
            st.markdown("❓ **查詢次數**")
            st.markdown(f"<h2 style='color: #ffffff; margin: 0;'>{st.session_state.query_count}</h2>", unsafe_allow_html=True)
            
            st.divider()
            
            # Token使用統計 - 三欄布局仿照React
            if st.session_state.total_input_tokens > 0 or st.session_state.total_output_tokens > 0:
                st.markdown("### 📈 Token 使用統計")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**輸入**")
                    st.markdown(f"<h3 style='color: #69c0ff; margin: 0;'>{st.session_state.total_input_tokens:,}</h3>", unsafe_allow_html=True)
                    st.markdown("<small style='color: #d9d9d9;'>Tokens</small>", unsafe_allow_html=True)
                    
                with col2:
                    st.markdown("**輸出**") 
                    st.markdown(f"<h3 style='color: #73d13d; margin: 0;'>{st.session_state.total_output_tokens:,}</h3>", unsafe_allow_html=True)
                    st.markdown("<small style='color: #d9d9d9;'>Tokens</small>", unsafe_allow_html=True)
                    
                with col3:
                    total_tokens = st.session_state.total_input_tokens + st.session_state.total_output_tokens
                    st.markdown("**總計**")
                    st.markdown(f"<h3 style='color: #faad14; margin: 0;'>{total_tokens:,}</h3>", unsafe_allow_html=True)
                    st.markdown("<small style='color: #d9d9d9;'>Tokens</small>", unsafe_allow_html=True)
                
                st.divider()
            
            # 範例查詢 - 仿照React的範例查詢區塊
            st.markdown("### 🎯 範例查詢")
            
            example_queries = [
                "加班費如何計算？包括平日加班和假日加班的費率規定",
                "工作時間有什麼限制？正常工時和延長工時的規定", 
                "雇主資遣員工需要遵循什麼程序？資遣費如何計算？",
                "2025年勞基法有哪些重要的修正內容？"
            ]
            
            for i, query in enumerate(example_queries, 1):
                if st.button(f"❓ {query[:20]}...", key=f"example_{i}", help=query, use_container_width=True):
                    self.handle_example_query(query)
    
    def render_main_interface(self):
        """渲染主介面 - 仿照React布局"""
        # Header區域 - 仿照React的Header
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown("""
            <div style="background-color: #1a1a1a; padding: 16px 24px; margin: -16px -16px 24px -16px; border-bottom: 1px solid #303030;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 24px; color: #69c0ff;">🤖</span>
                    <span style="color: #ffffff; font-size: 20px; font-weight: bold;">勞動基準法 RAG 查詢系統</span>
                    <span style="background: #003a8c; color: #69c0ff; padding: 2px 8px; border-radius: 4px; font-size: 12px;">v2.0</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #1a1a1a; padding: 16px 8px; margin: -16px -16px 24px -16px; border-bottom: 1px solid #303030;">
            </div>
            """, unsafe_allow_html=True)
            # API 文檔連結按鈕 - 直接用 HTML 實現新分頁開啟
            st.markdown("""
            <a href="http://localhost:8000/docs" target="_blank" style="
                display: inline-block;
                background-color: #262626;
                color: #d9d9d9;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 8px 16px;
                text-decoration: none;
                font-size: 14px;
                transition: all 0.3s ease;
                margin-bottom: 8px;
            " onmouseover="this.style.backgroundColor='#404040'; this.style.color='#ffffff';" 
               onmouseout="this.style.backgroundColor='#262626'; this.style.color='#d9d9d9';">
                📖 API 文檔
            </a>
            """, unsafe_allow_html=True)
        
        # 緊湊版功能說明 - 仿照React的設計
        show_description = st.session_state.get('show_system_description', False)
        
        # 緊湊版提示條
        col1, col2 = st.columns([9, 1])
        with col1:
            if st.button("💡 系統功能說明 - 點擊查看詳細功能", key="system_desc_toggle", use_container_width=True):
                st.session_state.show_system_description = not show_description
        with col2:
            st.markdown(f"{'🔼' if show_description else '🔽'}")
        
        # 展開的功能說明
        if st.session_state.get('show_system_description', False):
            st.markdown("""
            <div style="background: #1f1f1f; border: 1px solid #303030; border-radius: 6px; padding: 16px; margin-bottom: 16px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                    <div>
                        <strong style="color: #73d13d; font-size: 14px;">🔍 智能搜索功能:</strong><br>
                        <span style="color: #d9d9d9; line-height: 1.6;">
                            • 語義理解與精確匹配<br>
                            • 智能重排序優化<br>
                            • 適用於所有法條查詢
                        </span>
                    </div>
                    <div>
                        <strong style="color: #69c0ff; font-size: 14px;">🌐 網路搜索功能:</strong><br>
                        <span style="color: #d9d9d9; line-height: 1.6;">
                            • 最新修法動態查詢<br>
                            • 政策解釋和實務案例<br>
                            • 相關新聞和時事資訊
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
    
    def handle_example_query(self, query: str):
        """處理範例查詢"""
        st.session_state.example_query = query
        st.rerun()
    
    def show_system_status_popup(self):
        """顯示系統狀態彈窗"""
        with st.expander("🔍 系統詳細狀態", expanded=True):
            if st.session_state.agent:
                st.success("✅ **RAG 系統：** 正常運行")
                st.info("🔧 **Reranker模型：** 1個 (bge-reranker-base)")
                st.info("🗄️ **資料庫連接：** 正常")
                st.info("🧠 **AI模型：** Azure OpenAI GPT-4")
            else:
                st.error("❌ **RAG 系統：** 初始化失敗")
            
            # 顯示環境資訊
            st.markdown("### 📊 系統資訊")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("查詢次數", st.session_state.query_count)
                st.metric("輸入 Tokens", f"{st.session_state.total_input_tokens:,}")
            with col2:
                st.metric("總 Tokens", f"{st.session_state.total_input_tokens + st.session_state.total_output_tokens:,}")
                st.metric("輸出 Tokens", f"{st.session_state.total_output_tokens:,}")
    
    def render_chat_interface(self):
        """渲染對話介面 - 仿照React布局"""
        
        # 聊天內容區域 - 使用容器來模擬固定高度
        chat_container = st.container()
        
        with chat_container:
            # 如果沒有對話，顯示歡迎訊息 - 仿照React
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align: center; padding: 40px; color: #8c8c8c;">
                    <div style="font-size: 48px; margin-bottom: 16px; color: #69c0ff;">🤖</div>
                    <div style="font-size: 16px; font-weight: bold; color: #ffffff; margin-bottom: 8px;">
                        歡迎使用勞動基準法 RAG 查詢系統
                    </div>
                    <div style="color: #8c8c8c;">
                        請輸入您的問題，或點擊左側的範例查詢開始對話
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # 顯示對話歷史
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        with st.chat_message("user", avatar="👤"):
                            st.markdown(f"**您的問題:** {message['content']}")
                    
                    elif message["role"] == "assistant":
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(message["content"])
                            
                            # 顯示技術細節 - 默認顯示
                            if "technical_details" in message:
                                self.render_technical_details(message["technical_details"])
        
        # 處理範例查詢
        if hasattr(st.session_state, 'example_query'):
            query = st.session_state.example_query
            del st.session_state.example_query
            self.process_query(query)
        
        # 輸入區域 - 仿照React的固定底部設計
        st.markdown("""
        <div style="border-top: 1px solid #303030; padding-top: 16px; margin-top: 16px;">
        </div>
        """, unsafe_allow_html=True)
        
        # 用戶輸入 - 使用chat_input模擬固定在底部
        if prompt := st.chat_input("請輸入您關於勞動基準法的問題..."):
            self.process_query(prompt)
            
        # 輸入提示 - 仿照React的提示文字
        st.markdown("""
        <div style="text-align: center; color: #8c8c8c; font-size: 12px; margin-top: 8px;">
            按 Enter 發送，Shift + Enter 換行
        </div>
        """, unsafe_allow_html=True)
    
    def generate_agent_response_with_tokens(self, query: str, conversation_history: List[Dict[str, str]] = None):
        """
        使用 AI agent 生成回答並追蹤 token 使用和技術細節
        
        Args:
            query: 用戶查詢
            conversation_history: 對話歷史列表
        
        Returns:
            tuple: (response, total_input_tokens, total_output_tokens, technical_details)
        """
        # 使用共用的追蹤功能
        response, technical_details = execute_query_with_tracking(st.session_state.agent, query, conversation_history)
        
        # 從技術細節中提取 token 使用量
        token_usage = technical_details.get('token_usage', {})
        total_input_tokens = token_usage.get('input', 0)
        total_output_tokens = token_usage.get('output', 0)
        
        return response, total_input_tokens, total_output_tokens, technical_details
    
    def process_query(self, query: str):
        """處理用戶查詢"""
        # 添加用戶消息
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.query_count += 1
        
        with st.chat_message("user"):
            st.markdown(f"**您的問題:** {query}")
        
        # 顯示 AI 處理狀態
        with st.chat_message("assistant"):
            with st.spinner("🤖 AI Agent 正在分析您的問題..."):
                
                # 創建狀態顯示區域
                status_container = st.container()
                result_container = st.container()
                
                with status_container:
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                
                # 執行查詢
                technical_details = {}
                total_input_tokens = 0
                total_output_tokens = 0
                
                try:
                    # 更新狀態
                    status_placeholder.info("📝 正在改寫和完善查詢...")
                    progress_bar.progress(20)
                    
                    # 準備對話歷史
                    conversation_history = []
                    for msg in st.session_state.messages:
                        if msg["role"] in ["user", "assistant"]:
                            conversation_history.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                    
                    # 獲取 AI Agent 回應（使用修改後的方法來追蹤token和技術細節）
                    response, input_tokens, output_tokens, technical_details = self.generate_agent_response_with_tokens(query, conversation_history)
                    
                    # 更新token統計
                    total_input_tokens = input_tokens
                    total_output_tokens = output_tokens
                    st.session_state.total_input_tokens += input_tokens
                    st.session_state.total_output_tokens += output_tokens
                    
                    progress_bar.progress(100)
                    status_placeholder.success("✅ 查詢完成！")
                    
                    # 清除狀態顯示
                    time.sleep(1)
                    status_container.empty()
                    
                    # 顯示結果
                    with result_container:
                        st.markdown(response)
                        
                        # 添加token使用資訊到技術細節
                        technical_details["token_usage"] = {
                            "input": total_input_tokens,
                            "output": total_output_tokens,
                            "total": total_input_tokens + total_output_tokens
                        }
                        
                        # 添加回應到歷史
                        assistant_message = {
                            "role": "assistant", 
                            "content": response,
                            "technical_details": technical_details,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.messages.append(assistant_message)
                
                except Exception as e:
                    status_placeholder.error(f"❌ 處理查詢時發生錯誤: {e}")
                    st.error("請稍後再試或重新表述您的問題")
        
        # 重新運行以更新界面
        st.rerun()
    
    def render_technical_details(self, details: Dict[str, Any]):
        """渲染技術細節"""
        if not details:
            return
        
        # 生成唯一的標識符以避免重複的key
        import time
        import random
        unique_id = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        with st.expander("🔧 技術細節", expanded=False):
            
            # 回答參考的文檔片段
            if "used_chunks" in details and details["used_chunks"]:
                st.subheader("📚 回答參考的文檔片段")
                st.info("📌 AI回答時參考的文檔片段及相關性分數 (✅ 表示被使用)")
                
                for i, chunk in enumerate(details["used_chunks"], 1):
                    with st.container():
                        # 標題顯示使用狀態
                        used_indicator = "✅ 已使用" if chunk.get("used_in_response", True) else "❌ 未使用"
                        
                        # 使用折疊式顯示，但不使用nested expander
                        st.markdown(f"**📄 片段 {i} (ID: {chunk.get('id', 'N/A')}) - {used_indicator}**")
                        
                        # 內容預覽
                        st.write("**內容預覽:**")
                        st.write(chunk.get("content", ""))
                        
                        # 分數顯示 - 只顯示rerank和相似度分數
                        col1, col2 = st.columns(2)
                        with col1:
                            if chunk.get("rerank_score") is not None:
                                st.metric("Rerank 分數", f"{chunk['rerank_score']:.4f}")
                        
                        with col2:
                            if chunk.get("similarity") is not None:
                                st.metric("相似度分數", f"{chunk['similarity']:.4f}")
                        
                        # 完整內容 - 使用checkbox來控制顯示
                        if chunk.get("full_content"):
                            chunk_id = chunk.get('id', f'chunk_{i}')
                            show_full = st.checkbox(f"顯示完整內容", key=f"show_full_{unique_id}_{chunk_id}_{i}")
                            if show_full:
                                st.text_area("完整內容:", value=chunk["full_content"], height=150, disabled=True, key=f"chunk_content_{unique_id}_{chunk_id}_{i}")
                        
                        st.divider()
                
                st.divider()
            
            # Token 使用統計
            if "token_usage" in details:
                token_info = details["token_usage"]
                st.markdown(f'''
                <div class="token-usage">
                    <strong>Token 使用統計:</strong><br>
                    輸入: {token_info.get("input", 0)} | 輸出: {token_info.get("output", 0)} | 總計: {token_info.get("total", 0)}
                </div>
                ''', unsafe_allow_html=True)
    
    def run(self):
        """運行應用程式"""
        self.render_sidebar()
        self.render_main_interface()
        st.divider()
        self.render_chat_interface()

def main():
    """主程式入口"""
    app = RAGStreamlitApp()
    app.run()

if __name__ == "__main__":
    main()