import axios from 'axios'
import type { QueryRequest, QueryResponse, HealthResponse } from '../types'

// 創建 axios 實例
const api = axios.create({
  baseURL: '/api',  // 透過 Vite proxy 轉發到後端
  timeout: 60000,   // 60秒超時
  headers: {
    'Content-Type': 'application/json',
  }
})

// 請求攔截器
api.interceptors.request.use(
  (config) => {
    console.log('🚀 API Request:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('❌ Request Error:', error)
    return Promise.reject(error)
  }
)

// 回應攔截器
api.interceptors.response.use(
  (response) => {
    console.log('✅ API Response:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('❌ Response Error:', error.response?.status, error.response?.data)
    
    // 處理常見錯誤
    if (error.response?.status === 503) {
      error.message = 'RAG 系統暫時無法使用，請稍後再試'
    } else if (error.response?.status === 500) {
      error.message = '服務器內部錯誤，請聯繫管理員'
    } else if (error.code === 'ECONNABORTED') {
      error.message = '請求超時，請檢查網路連接'
    } else if (!error.response) {
      error.message = '無法連接到服務器，請檢查後端服務是否運行'
    }
    
    return Promise.reject(error)
  }
)

export class QueryService {
  /**
   * 查詢勞動基準法
   */
  static async query(request: QueryRequest): Promise<QueryResponse> {
    const response = await api.post<QueryResponse>('/query', request)
    return response.data
  }

  /**
   * 檢查系統健康狀態
   */
  static async checkHealth(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>('/health')
    return response.data
  }

  /**
   * 獲取根路徑信息
   */
  static async getInfo(): Promise<{ message: string; version: string; docs: string }> {
    const response = await api.get('/')
    return response.data
  }
}

// WebSocket 服務
export class WebSocketService {
  private ws: WebSocket | null = null
  private url: string
  private onMessage: (data: any) => void
  private onError: (error: Event) => void
  private onClose: () => void

  constructor(
    url: string = 'ws://localhost:8000/ws',
    onMessage: (data: any) => void = () => {},
    onError: (error: Event) => void = () => {},
    onClose: () => void = () => {}
  ) {
    this.url = url
    this.onMessage = onMessage
    this.onError = onError
    this.onClose = onClose
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url)
        
        this.ws.onopen = () => {
          console.log('🔗 WebSocket 連接已建立')
          resolve()
        }
        
        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            this.onMessage(data)
          } catch (error) {
            console.error('❌ WebSocket 消息解析失敗:', error)
          }
        }
        
        this.ws.onerror = (error) => {
          console.error('❌ WebSocket 錯誤:', error)
          this.onError(error)
          reject(error)
        }
        
        this.ws.onclose = () => {
          console.log('🔌 WebSocket 連接已關閉')
          this.onClose()
        }
        
      } catch (error) {
        reject(error)
      }
    })
  }

  sendMessage(message: { question: string; include_technical_details?: boolean }): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.error('❌ WebSocket 未連接')
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

export default api