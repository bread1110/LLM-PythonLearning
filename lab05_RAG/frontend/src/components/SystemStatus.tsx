import React from 'react'
import { Card, Statistic, Row, Col, Tag, Typography } from 'antd'
import { 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  ApiOutlined,
  ThunderboltOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons'
import type { SystemInfo } from '../types'

const { Text } = Typography

interface Props {
  systemInfo: SystemInfo | null
  totalQueries: number
}

const SystemStatus: React.FC<Props> = ({ systemInfo, totalQueries }) => {
  const getStatusIcon = () => {
    if (!systemInfo) return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
    return systemInfo.agent_initialized ? 
      <CheckCircleOutlined style={{ color: '#73d13d' }} /> : 
      <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
  }

  const getStatusText = () => {
    if (!systemInfo) return '未知'
    return systemInfo.agent_initialized ? '正常運行' : '系統異常'
  }

  const getStatusColor = () => {
    if (!systemInfo) return 'default'
    return systemInfo.agent_initialized ? 'success' : 'error'
  }

  return (
    <Card size="small" title="系統狀態">
      <Row gutter={[0, 16]}>
        <Col span={24}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {getStatusIcon()}
            <Text strong style={{ color: '#ffffff' }}>{getStatusText()}</Text>
          </div>
        </Col>
        
        <Col span={24}>
          <Statistic
            title={<span style={{ color: '#f0f0f0' }}>查詢次數</span>}
            value={totalQueries}
            prefix={<QuestionCircleOutlined style={{ color: '#69c0ff' }} />}
            valueStyle={{ color: '#ffffff' }}
          />
        </Col>

        {systemInfo && (
          <>
            <Col span={24}>
              <div>
                <Text strong style={{ color: '#f0f0f0' }}>Reranker 模型: </Text>
                <Tag style={{
                  background: '#003a8c',
                  borderColor: '#1677ff',
                  color: '#69c0ff'
                }}>
                  {systemInfo.reranker_models || 0} 個
                </Tag>
              </div>
            </Col>

            <Col span={24}>
              <div>
                <Text strong style={{ color: '#f0f0f0' }}>可用工具:</Text>
                <div style={{ marginTop: '4px' }}>
                  {systemInfo.available_tools?.map(tool => (
                    <Tag 
                      key={tool} 
                      icon={<ApiOutlined style={{ color: '#73d13d' }} />} 
                      style={{ 
                        marginBottom: '4px',
                        background: '#1f4c2e',
                        borderColor: '#389e0d',
                        color: '#73d13d'
                      }}
                    >
                      {tool}
                    </Tag>
                  )) || <Text style={{ color: '#8c8c8c' }}>無資料</Text>}
                </div>
              </div>
            </Col>

            {systemInfo.system_error && (
              <Col span={24}>
                <Tag 
                  icon={<ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />}
                  style={{
                    background: '#5c0011',
                    borderColor: '#cf1322',
                    color: '#ff4d4f'
                  }}
                >
                  {systemInfo.system_error}
                </Tag>
              </Col>
            )}
          </>
        )}
      </Row>
    </Card>
  )
}

export default SystemStatus