import React from 'react'
import { Card, Typography, Tag, Divider, Space, Alert } from 'antd'
import { UserOutlined, RobotOutlined, ClockCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons'
import Markdown from 'markdown-to-jsx'
import TechnicalDetails from './TechnicalDetails'
import type { ChatMessage } from '../types'

const { Text, Paragraph } = Typography

interface Props {
  messages: ChatMessage[]
  loading?: boolean
}

const ChatInterface: React.FC<Props> = ({ messages, loading }) => {
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('zh-TW', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const renderMessage = (message: ChatMessage) => {
    const isUser = message.type === 'user'
    const isError = message.type === 'error'

    return (
      <div key={message.id} style={{ marginBottom: '16px' }}>
        <Card
          size="small"
          style={{
            marginLeft: isUser ? '20%' : '0',
            marginRight: isUser ? '0' : '20%',
            background: isUser ? '#003a8c' : isError ? '#5c0011' : '#1f4c2e',
            borderColor: isUser ? '#1677ff' : isError ? '#cf1322' : '#389e0d'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
            <div style={{ fontSize: '16px', marginTop: '2px' }}>
              {isUser ? (
                <UserOutlined style={{ color: '#69c0ff' }} />
              ) : isError ? (
                <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
              ) : (
                <RobotOutlined style={{ color: '#73d13d' }} />
              )}
            </div>
            
            <div style={{ flex: 1 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <Text strong style={{ color: '#ffffff' }}>
                  {isUser ? '您' : isError ? '系統錯誤' : 'AI 助手'}
                </Text>
                <Text style={{ fontSize: '12px', color: '#bfbfbf' }}>
                  <ClockCircleOutlined /> {formatTime(message.timestamp)}
                </Text>
                {message.processingTime && (
                  <Tag style={{ 
                    fontSize: '11px', 
                    background: '#003a8c',
                    borderColor: '#1677ff',
                    color: '#69c0ff'
                  }}>
                    {message.processingTime.toFixed(2)}s
                  </Tag>
                )}
              </div>
              
              <div style={{ lineHeight: '1.6' }}>
                {isError ? (
                  <Alert message={message.content} type="error" showIcon />
                ) : isUser ? (
                  <Paragraph style={{ 
                    marginBottom: 0, 
                    whiteSpace: 'pre-wrap',
                    color: '#d9d9d9'
                  }}>
                    {message.content}
                  </Paragraph>
                ) : (
                  <div style={{ 
                    background: '#262626', 
                    padding: '20px 24px', 
                    borderRadius: '6px',
                    border: '1px solid #404040',
                    color: '#d9d9d9',
                    overflow: 'hidden'
                  }}>
                    <div style={{ 
                      paddingLeft: '8px',
                      lineHeight: '1.6'
                    }}>
                      <Markdown options={{ wrapper: 'div' }}>
                        {message.content}
                      </Markdown>
                    </div>
                  </div>
                )}
              </div>
              
              {message.technicalDetails && (
                <div style={{ marginTop: '12px' }}>
                  <TechnicalDetails details={message.technicalDetails} />
                </div>
              )}
            </div>
          </div>
        </Card>
      </div>
    )
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {messages.length === 0 ? (
        <div style={{ 
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column', 
          justifyContent: 'center', 
          alignItems: 'center',
          textAlign: 'center', 
          padding: '40px' 
        }}>
          <RobotOutlined style={{ 
            fontSize: '48px', 
            marginBottom: '16px',
            color: '#69c0ff'
          }} />
          <div>
            <Text strong style={{ 
              fontSize: '16px',
              color: '#ffffff'
            }}>歡迎使用勞動基準法 RAG 查詢系統</Text>
          </div>
          <div style={{ marginTop: '8px' }}>
            <Text style={{ color: '#8c8c8c' }}>請輸入您的問題，或點擊左側的範例查詢開始對話</Text>
          </div>
        </div>
      ) : (
        <div style={{ 
          flex: 1, 
          overflow: 'auto', 
          paddingRight: '8px',
          paddingBottom: '8px',
          minHeight: 0
        }}>
          {messages.map(renderMessage)}
          {loading && (
            <div style={{ textAlign: 'center', padding: '20px' }}>
              <Space>
                <RobotOutlined spin style={{ color: '#69c0ff' }} />
                <Text style={{ color: '#bfbfbf' }}>AI 正在思考中...</Text>
              </Space>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default ChatInterface