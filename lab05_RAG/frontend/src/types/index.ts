// API 請求和回應類型定義

export interface QueryRequest {
  question: string
  session_id?: string
  include_technical_details?: boolean
  messages?: Array<{
    role: string
    content: string
  }>
}

export interface SearchResult {
  id: number
  content: string
  hybrid_score?: number
  ensemble_score?: number
  vector_score?: number
  keyword_score?: number
  source?: string
}

export interface UsedChunk {
  id: number
  content: string
  full_content: string
  rerank_score?: number
  similarity?: number
  hybrid_score?: number
  ensemble_score?: number
  vector_score?: number
  keyword_score?: number
  source: string
  used_in_response: boolean
}

export interface TechnicalDetails {
  search_metadata?: Record<string, any>
  hybrid_results?: SearchResult[]
  web_results?: Array<{
    title: string
    content: string
    url: string
    score: number
  }>
  used_chunks?: UsedChunk[]
  token_usage?: {
    input: number
    output: number
    total: number
  }
}

export interface QueryResponse {
  answer: string
  session_id: string
  timestamp: string
  technical_details?: TechnicalDetails
  processing_time: number
}

export interface HealthResponse {
  status: string
  timestamp: string
  version: string
  system_info: SystemInfo
}

export interface SystemInfo {
  agent_initialized: boolean
  python_version: string
  api_server: string
  reranker_models?: number
  available_tools?: string[]
  system_error?: string
}

export interface ErrorResponse {
  error: string
  detail?: string
  timestamp: string
}

// 前端組件類型

export interface ChatMessage {
  id: string
  type: 'user' | 'assistant' | 'error'
  content: string
  timestamp: string
  technicalDetails?: TechnicalDetails
  processingTime?: number
}

export interface SearchMetadata {
  count: number
  search_type: string
  vector_weight?: number
  keyword_weight?: number
  vector_results_count?: number
  keyword_results_count?: number
  query?: string
}