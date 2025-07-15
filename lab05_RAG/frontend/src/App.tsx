import React, { useState, useEffect } from 'react'
import { 
  Layout, 
  Typography, 
  Card, 
  Button, 
  Input, 
  Space, 
  Divider, 
  Alert,
  Spin,
  Row,
  Col,
  Tag,
  notification
} from 'antd'
import { 
  SendOutlined, 
  QuestionCircleOutlined, 
  RobotOutlined,
  ClockCircleOutlined,
  ApiOutlined,
  DownOutlined,
  UpOutlined
} from '@ant-design/icons'
import ChatInterface from './components/ChatInterface'
import SystemStatus from './components/SystemStatus'
import TechnicalDetails from './components/TechnicalDetails'
import { QueryService } from './services/api'
import type { ChatMessage, SystemInfo } from './types'

const { Header, Content, Footer, Sider } = Layout
const { Title, Paragraph, Text } = Typography
const { TextArea } = Input

const App: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [currentQuestion, setCurrentQuestion] = useState('')
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null)
  const [collapsed, setCollapsed] = useState(false)
  const [totalQueries, setTotalQueries] = useState(0)
  const [showSystemDescription, setShowSystemDescription] = useState(false)

  // 初始化時檢查系統健康狀態
  useEffect(() => {
    checkSystemHealth()
  }, [])

  const checkSystemHealth = async () => {
    try {
      const health = await QueryService.checkHealth()
      setSystemInfo(health.system_info)
      
      if (health.status !== 'healthy') {
        notification.warning({
          message: '系統狀態異常',
          description: '後端服務可能未正常運行，請檢查服務狀態',
        })
      }
    } catch (error) {
      notification.error({
        message: '無法連接到後端服務',
        description: '請確認 API 服務器正在運行（端口 8000）',
      })
    }
  }

  const handleSubmitQuestion = async () => {
    if (!currentQuestion.trim()) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: currentQuestion,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setLoading(true)
    
    // 準備對話歷史（包含當前使用者訊息）
    const conversationHistory = [
      ...messages.map(msg => ({
        role: msg.type === 'user' ? 'user' : 'assistant',
        content: msg.content
      })),
      {
        role: 'user',
        content: currentQuestion
      }
    ]
    
    setCurrentQuestion('')

    try {
      const response = await QueryService.query({
        question: currentQuestion,
        include_technical_details: true,
        messages: conversationHistory
      })

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.answer,
        timestamp: response.timestamp,
        technicalDetails: response.technical_details,
        processingTime: response.processing_time
      }

      setMessages(prev => [...prev, assistantMessage])
      setTotalQueries(prev => prev + 1)

    } catch (error: any) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'error',
        content: `查詢失敗：${error.response?.data?.detail || error.message}`,
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMessage])
      
      notification.error({
        message: '查詢失敗',
        description: error.response?.data?.detail || '請稍後再試'
      })
    } finally {
      setLoading(false)
    }
  }

  const handleExampleQuery = (query: string) => {
    setCurrentQuestion(query)
  }


  const exampleQueries = [
    "加班費如何計算？包括平日加班和假日加班的費率規定",
    "工作時間有什麼限制？正常工時和延長工時的規定",
    "雇主資遣員工需要遵循什麼程序？資遣費如何計算？",
    "2025年勞基法有哪些重要的修正內容？"
  ]

  return (
    <Layout className="app-container">
      <Header style={{ 
        background: '#1a1a1a', 
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #303030'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <RobotOutlined style={{ fontSize: '24px', color: '#69c0ff' }} />
          <Title level={3} style={{ color: '#ffffff', margin: 0 }}>
            勞動基準法 RAG 查詢系統
          </Title>
          <Tag color="cyan" style={{ background: '#003a8c', borderColor: '#1677ff', color: '#69c0ff' }}>v2.0</Tag>
        </div>
        <Button 
          type="text" 
          icon={<ApiOutlined />} 
          style={{ color: '#d9d9d9' }}
          href="http://localhost:8000/docs" 
          target="_blank"
        >
          API 文檔
        </Button>
      </Header>

      <Layout>
        <Sider 
          width="25%" 
          theme="dark" 
          collapsible 
          collapsed={collapsed} 
          onCollapse={setCollapsed}
          style={{ 
            background: '#262626',
            minWidth: collapsed ? '80px' : '300px',
            maxWidth: collapsed ? '80px' : '400px'
          }}
        >
          <div style={{ padding: '16px' }}>
            {!collapsed && (
              <>
                <SystemStatus systemInfo={systemInfo} totalQueries={totalQueries} />
                <Divider />
                <Card size="small" title="範例查詢">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {exampleQueries.map((query, index) => (
                      <Button
                        key={index}
                        type="link"
                        size="small"
                        onClick={() => handleExampleQuery(query)}
                        style={{ 
                          textAlign: 'left', 
                          height: 'auto', 
                          whiteSpace: 'normal',
                          padding: '4px 8px'
                        }}
                      >
                        <QuestionCircleOutlined /> {query.substring(0, 30)}...
                      </Button>
                    ))}
                  </Space>
                </Card>
              </>
            )}
          </div>
        </Sider>

        <Layout style={{ padding: '0 2vw 0', background: '#141414', flex: 1 }}>
          <Content style={{ 
            padding: '2vw 2vw 0', 
            height: 'calc(100vh - 64px - 60px)', 
            background: '#141414',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              {/* 系統功能說明區域 - 緊湊版 */}
              <div 
                style={{ 
                  background: 'rgba(31, 31, 31, 0.6)',
                  border: '1px solid #303030',
                  borderRadius: '6px',
                  padding: '8px 16px',
                  marginBottom: '16px',
                  flexShrink: 0,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  transition: 'all 0.3s ease'
                }}
                onClick={() => setShowSystemDescription(!showSystemDescription)}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(31, 31, 31, 0.8)'
                  e.currentTarget.style.borderColor = '#404040'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'rgba(31, 31, 31, 0.6)'
                  e.currentTarget.style.borderColor = '#303030'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Text style={{ color: '#69c0ff', fontSize: '13px' }}>💡</Text>
                  <Text style={{ color: '#d9d9d9', fontSize: '13px' }}>
                    系統功能說明 - 點擊查看詳細功能
                  </Text>
                </div>
                <Button 
                  type="text" 
                  size="small"
                  icon={showSystemDescription ? <UpOutlined /> : <DownOutlined />}
                  style={{ color: '#8c8c8c', padding: '2px 4px' }}
                />
              </div>
              
              {showSystemDescription && (
                <Card className="dark-card" style={{ marginBottom: '16px', flexShrink: 0 }}>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Paragraph>
                        <Text strong style={{ color: '#73d13d', fontSize: '14px' }}>🔍 智能搜索功能:</Text>
                        <br />
                        <Text style={{ color: '#d9d9d9', lineHeight: '1.6' }}>
                          • 語義理解與精確匹配
                          <br />
                          • 智能重排序優化
                          <br />
                          • 適用於所有法條查詢
                        </Text>
                      </Paragraph>
                    </Col>
                    <Col span={12}>
                      <Paragraph>
                        <Text strong style={{ color: '#69c0ff', fontSize: '14px' }}>🌐 網路搜索功能:</Text>
                        <br />
                        <Text style={{ color: '#d9d9d9', lineHeight: '1.6' }}>
                          • 最新修法動態查詢
                          <br />
                          • 政策解釋和實務案例
                          <br />
                          • 相關新聞和時事資訊
                        </Text>
                      </Paragraph>
                    </Col>
                  </Row>
                </Card>
              )}

              {/* 聊天內容區域 - 可滾動 */}
              <Card className="dark-card chat-content-card" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                  <ChatInterface 
                    messages={messages}
                    loading={loading}
                  />
                </div>
              </Card>

              {/* 輸入區域 - 固定在底部 */}
              <Card className="dark-card input-area-card" style={{ 
                marginTop: '16px', 
                marginBottom: '2vw',
                flexShrink: 0 
              }}>
                <Space.Compact style={{ width: '100%' }}>
                  <TextArea
                    placeholder="請輸入您關於勞動基準法的問題..."
                    value={currentQuestion}
                    onChange={(e) => setCurrentQuestion(e.target.value)}
                    onPressEnter={(e) => {
                      if (!e.shiftKey) {
                        e.preventDefault()
                        handleSubmitQuestion()
                      }
                    }}
                    autoSize={{ minRows: 2, maxRows: 4 }}
                    disabled={loading}
                  />
                  <Button
                    type="primary"
                    className="send-button"
                    icon={<SendOutlined />}
                    onClick={handleSubmitQuestion}
                    loading={loading}
                    disabled={!currentQuestion.trim()}
                  >
                    發送
                  </Button>
                </Space.Compact>
                <div style={{ marginTop: '8px', color: '#8c8c8c', fontSize: '12px' }}>
                  按 Enter 發送，Shift + Enter 換行
                </div>
              </Card>
            </div>
          </Content>
        </Layout>
      </Layout>

      <Footer style={{ 
        textAlign: 'center', 
        background: '#1a1a1a', 
        color: '#8c8c8c', 
        borderTop: '1px solid #303030',
        padding: '1vh 2vw',
        height: 'auto',
        lineHeight: '1.5'
      }}>
        <Text style={{ color: '#8c8c8c', fontSize: 'clamp(11px, 1.5vw, 13px)' }}>
          勞動基準法 RAG 查詢系統 ©2025 - 基於 FastAPI + React 架構
        </Text>
      </Footer>
    </Layout>
  )
}

export default App