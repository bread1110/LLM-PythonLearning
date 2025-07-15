"""
Technical details tracking utilities for Lab05 RAG system
Provides centralized tracking for token usage and technical details
"""

from typing import Dict, List, Any, Optional, Tuple
from .ai_client import chat_with_azure_openai


class TokenAndDetailsTracker:
    """
    Unified tracker for token usage and technical details across different components
    """
    
    def __init__(self, original_agent):
        """
        Initialize tracker with reference to original agent
        
        Args:
            original_agent: The original agent instance to track
        """
        self.agent = original_agent
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.web_search_results = None
        self.hybrid_search_results = None
        self.vector_search_results = None
        self.search_metadata = {}
        self.used_chunks = []
    
    def track_tokens(self, input_tokens: int, output_tokens: int):
        """
        Track token usage
        
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
    
    def track_tool_result(self, tool_name: str, tool_result: Dict[str, Any]):
        """
        Track tool execution results
        
        Args:
            tool_name (str): Name of the executed tool
            tool_result (Dict[str, Any]): Result from tool execution
        """
        if tool_name == "web_search" and tool_result.get("success"):
            self.web_search_results = tool_result.get("results", [])
            self.search_metadata["web_search"] = {
                "count": tool_result.get("count", 0),
                "query": tool_result.get("query", "")
            }
        elif tool_name == "vector_search" and tool_result.get("success"):
            self._track_vector_search(tool_result)
        elif tool_name == "hybrid_search" and tool_result.get("success"):
            self._track_hybrid_search(tool_result)
    
    def _track_vector_search(self, tool_result: Dict[str, Any]):
        """Track vector search results"""
        self.vector_search_results = tool_result.get("results", [])
        self.search_metadata["vector_search"] = {
            "count": tool_result.get("count", 0),
            "original_count": tool_result.get("original_count", 0),
            "reranked": tool_result.get("reranked", False),
            "reranking_method": tool_result.get("reranking_method", "")
        }
        
        # Track used chunks from vector search
        if self.vector_search_results:
            for result in self.vector_search_results:
                chunk_info = {
                    "id": result.get("id"),
                    "content": self._truncate_content(result.get("content", "")),
                    "full_content": result.get("content", ""),
                    "rerank_score": result.get("rerank_score"),
                    "similarity": result.get("similarity"),
                    "source": "vector_search",
                    "used_in_response": True
                }
                self.used_chunks.append(chunk_info)
    
    def _track_hybrid_search(self, tool_result: Dict[str, Any]):
        """Track hybrid search results"""
        self.hybrid_search_results = tool_result.get("results", [])
        self.search_metadata["hybrid_search"] = {
            "count": tool_result.get("count", 0),
            "search_type": tool_result.get("search_type", "hybrid"),
            "vector_weight": tool_result.get("vector_weight", 0.7),
            "keyword_weight": tool_result.get("keyword_weight", 0.3),
            "vector_results_count": tool_result.get("vector_results_count", 0),
            "keyword_results_count": tool_result.get("keyword_results_count", 0)
        }
        
        # Track used chunks from hybrid search
        if self.hybrid_search_results:
            for result in self.hybrid_search_results:
                chunk_info = {
                    "id": result.get("id"),
                    "content": self._truncate_content(result.get("content", "")),
                    "full_content": result.get("content", ""),
                    "hybrid_score": result.get("hybrid_score"),
                    "ensemble_score": result.get("ensemble_score"),
                    "vector_score": result.get("vector_score"),
                    "keyword_score": result.get("keyword_score"),
                    "source": result.get("source", "hybrid"),
                    "used_in_response": True
                }
                self.used_chunks.append(chunk_info)
    
    def _truncate_content(self, content: str, max_length: int = 200) -> str:
        """
        Truncate content for display
        
        Args:
            content (str): Original content
            max_length (int): Maximum length
            
        Returns:
            str: Truncated content
        """
        if len(content) > max_length:
            return content[:max_length] + "..."
        return content
    
    def get_token_usage(self) -> Dict[str, int]:
        """
        Get token usage summary
        
        Returns:
            Dict[str, int]: Token usage statistics
        """
        return {
            "input": self.total_input_tokens,
            "output": self.total_output_tokens,
            "total": self.total_input_tokens + self.total_output_tokens
        }
    
    def get_technical_details(self) -> Dict[str, Any]:
        """
        Get comprehensive technical details
        
        Returns:
            Dict[str, Any]: Technical details dictionary
        """
        details = {}
        
        if self.search_metadata:
            details["search_metadata"] = self.search_metadata
        
        if self.hybrid_search_results:
            details["hybrid_results"] = self.hybrid_search_results
        
        if self.web_search_results:
            details["web_results"] = self.web_search_results
        
        if self.vector_search_results:
            details["vector_results"] = self.vector_search_results
        
        if self.used_chunks:
            details["used_chunks"] = self.used_chunks
        
        details["token_usage"] = self.get_token_usage()
        
        return details


def create_tracking_wrapper(agent, tracker: TokenAndDetailsTracker) -> Tuple[callable, callable]:
    """
    Create wrapped methods for tracking token usage and tool execution
    
    Args:
        agent: The agent instance to wrap
        tracker: The tracker instance
        
    Returns:
        Tuple[callable, callable]: (wrapped_chat_method, wrapped_execute_tool)
    """
    # Store original methods
    original_chat_method = getattr(agent, 'chat_with_aoai_gpt', None)
    original_execute_tool = getattr(agent, 'execute_tool', None)
    
    def wrapped_chat_method(*args, **kwargs):
        """Wrapped chat method with token tracking"""
        if original_chat_method:
            result = original_chat_method(*args, **kwargs)
            if len(result) == 3:  # (message, input_tokens, output_tokens)
                message, input_tokens, output_tokens = result
                tracker.track_tokens(input_tokens, output_tokens)
            return result
        else:
            # Fallback to direct Azure OpenAI call
            return chat_with_azure_openai(*args, **kwargs)
    
    def wrapped_execute_tool(tool_name: str, **kwargs):
        """Wrapped execute tool method with result tracking"""
        if original_execute_tool:
            result = original_execute_tool(tool_name, **kwargs)
            tracker.track_tool_result(tool_name, result)
            return result
        else:
            raise AttributeError("execute_tool method not found on agent")
    
    return wrapped_chat_method, wrapped_execute_tool


def execute_query_with_tracking(agent, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Execute a query with comprehensive tracking
    
    Args:
        agent: The agent instance
        query: The query string
        conversation_history: Optional conversation history
        
    Returns:
        Tuple[str, Dict[str, Any]]: (response, technical_details)
    """
    # Create tracker
    tracker = TokenAndDetailsTracker(agent)
    
    # Create wrapped methods
    wrapped_chat_method, wrapped_execute_tool = create_tracking_wrapper(agent, tracker)
    
    # Store original methods
    original_chat_method = getattr(agent, 'chat_with_aoai_gpt', None)
    original_execute_tool = getattr(agent, 'execute_tool', None)
    
    # Temporarily replace methods
    agent.chat_with_aoai_gpt = wrapped_chat_method
    agent.execute_tool = wrapped_execute_tool
    
    try:
        # Execute query
        response = agent.generate_agent_response(query, conversation_history)
        
        # Get technical details
        technical_details = tracker.get_technical_details()
        
        return response, technical_details
        
    finally:
        # Restore original methods
        if original_chat_method:
            agent.chat_with_aoai_gpt = original_chat_method
        if original_execute_tool:
            agent.execute_tool = original_execute_tool