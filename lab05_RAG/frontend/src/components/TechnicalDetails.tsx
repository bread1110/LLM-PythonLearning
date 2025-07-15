import React from 'react'
import { Collapse, Table, Tag, Typography, Card, Row, Col, Progress } from 'antd'
import { 
  SearchOutlined, 
  GlobalOutlined, 
  ThunderboltOutlined,
  BarChartOutlined,
  FileTextOutlined
} from '@ant-design/icons'
import type { TechnicalDetails, SearchResult, SearchMetadata, UsedChunk } from '../types'

const { Text, Paragraph } = Typography
const { Panel } = Collapse

interface Props {
  details: TechnicalDetails
}

const TechnicalDetailsComponent: React.FC<Props> = ({ details }) => {
  if (!details) return null


  const renderHybridResults = () => {
    if (!details.hybrid_results || details.hybrid_results.length === 0) return null

    const columns = [
      {
        title: 'ID',
        dataIndex: 'id',
        key: 'id',
        width: 60
      },
      {
        title: '內容預覽',
        dataIndex: 'content',
        key: 'content',
        render: (content: string) => (
          <Paragraph ellipsis={{ rows: 2 }} style={{ marginBottom: 0, color: '#d9d9d9' }}>
            {content}
          </Paragraph>
        )
      },
      {
        title: '相關性分數',
        key: 'relevance_score',
        width: 180,
        render: (record: SearchResult) => {
          // 優先使用 ensemble_score，其次是 hybrid_score
          const score = record.ensemble_score !== undefined ? record.ensemble_score : record.hybrid_score
          const scoreLabel = record.ensemble_score !== undefined ? '智能評分' : '混合評分'
          
          return (
            <div>
              {score !== undefined && (
                <div>
                  <Text strong style={{ color: '#73d13d' }}>{scoreLabel}: </Text>
                  <Progress 
                    percent={Math.round(score * 100)} 
                    size="small" 
                    strokeColor="#73d13d"
                    format={(percent) => `${score.toFixed(3)}`}
                  />
                </div>
              )}
              <div style={{ marginTop: '4px' }}>
                <Tag style={{
                  background: record.source === 'hybrid' ? '#531dab' : record.source === 'vector' ? '#003a8c' : '#1f4c2e',
                  borderColor: record.source === 'hybrid' ? '#722ed1' : record.source === 'vector' ? '#1677ff' : '#389e0d',
                  color: record.source === 'hybrid' ? '#b37feb' : record.source === 'vector' ? '#69c0ff' : '#73d13d'
                }}>
                  {record.source || 'unknown'}
                </Tag>
              </div>
            </div>
          )
        }
      }
    ]

    return (
      <Panel header="🔀 混合搜索結果" key="hybrid">
        <Text style={{ display: 'block', marginBottom: '12px', color: '#8c8c8c' }}>
          📌 這是結合向量搜索和關鍵字搜索的最終結果，已通過多模型集成排序
        </Text>
        <Table 
          dataSource={details.hybrid_results} 
          columns={columns} 
          pagination={false}
          size="small"
          rowKey="id"
        />
      </Panel>
    )
  }

  const renderWebResults = () => {
    if (!details.web_results || details.web_results.length === 0) return null

    return (
      <Panel header="🌐 網路搜索結果" key="web">
        <Row gutter={[16, 16]}>
          {details.web_results.map((result, index) => (
            <Col span={24} key={index}>
              <Card size="small">
                <Text strong>{result.title}</Text>
                <br />
                <Text type="secondary">來源: </Text>
                <a href={result.url} target="_blank" rel="noopener noreferrer">
                  {result.url}
                </a>
                <br />
                <Paragraph ellipsis={{ rows: 2 }} style={{ marginTop: '8px', marginBottom: 0 }}>
                  {result.content}
                </Paragraph>
                {result.score && (
                  <div style={{ marginTop: '8px' }}>
                    <Tag color="orange">評分: {result.score.toFixed(2)}</Tag>
                  </div>
                )}
              </Card>
            </Col>
          ))}
        </Row>
      </Panel>
    )
  }

  const renderUsedChunks = () => {
    if (!details.used_chunks || details.used_chunks.length === 0) return null

    const columns = [
      {
        title: 'ID',
        dataIndex: 'id',
        key: 'id',
        width: 80,
        render: (id: number, record: UsedChunk) => (
          <div>
            <Tag style={{
              background: '#003a8c',
              borderColor: '#1677ff',
              color: '#69c0ff'
            }}>
              {id}
            </Tag>
            {record.used_in_response && (
              <div style={{ marginTop: '4px' }}>
                <Tag style={{
                  background: '#1f4c2e',
                  borderColor: '#389e0d',
                  color: '#73d13d',
                  fontSize: '10px'
                }}>
                  ✅ 已使用
                </Tag>
              </div>
            )}
          </div>
        )
      },
      {
        title: '內容預覽',
        dataIndex: 'content',
        key: 'content',
        render: (content: string) => (
          <Paragraph ellipsis={{ rows: 2 }} style={{ marginBottom: 8, color: '#d9d9d9' }}>
            {content}
          </Paragraph>
        )
      },
      {
        title: '相關性分數',
        key: 'scores',
        width: 200,
        render: (record: UsedChunk) => (
          <div>
            {record.rerank_score !== undefined && (
              <div style={{ marginBottom: '4px' }}>
                <Text strong style={{ color: '#73d13d' }}>Rerank: </Text>
                <Progress 
                  percent={Math.round(record.rerank_score * 100)} 
                  size="small" 
                  strokeColor="#73d13d"
                  format={(percent) => `${record.rerank_score?.toFixed(3)}`}
                />
              </div>
            )}
            {record.similarity !== undefined && (
              <div>
                <Text strong style={{ color: '#69c0ff' }}>相似度: </Text>
                <Progress 
                  percent={Math.round(record.similarity * 100)} 
                  size="small" 
                  strokeColor="#69c0ff"
                  format={(percent) => `${record.similarity?.toFixed(3)}`}
                />
              </div>
            )}
          </div>
        )
      }
    ]

    return (
      <Panel header="📚 回答參考的文檔片段" key="used_chunks">
        <Text style={{ display: 'block', marginBottom: '12px', color: '#8c8c8c' }}>
          📌 AI回答時參考的文檔片段及相關性分數 (✅ 表示被使用)
        </Text>
        <Table 
          dataSource={details.used_chunks} 
          columns={columns} 
          pagination={false}
          size="small"
          rowKey="id"
          expandable={{
            expandedRowRender: (record: UsedChunk) => (
              <div style={{ 
                background: '#1a1a1a', 
                padding: '12px', 
                borderRadius: '4px',
                border: '1px solid #404040',
                color: '#d9d9d9',
                whiteSpace: 'pre-wrap'
              }}>
                <Text strong style={{ color: '#ffffff', display: 'block', marginBottom: '8px' }}>
                  完整內容：
                </Text>
                {record.full_content}
              </div>
            ),
            rowExpandable: (record: UsedChunk) => !!record.full_content
          }}
        />
      </Panel>
    )
  }

  const renderTokenUsage = () => {
    if (!details.token_usage) return null

    const { input, output, total } = details.token_usage

    return (
      <Panel header="📈 Token 使用統計" key="tokens">
        <Row gutter={16}>
          <Col span={8}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#69c0ff' }}>
                  {input.toLocaleString()}
                </div>
                <div style={{ color: '#d9d9d9' }}>輸入 Tokens</div>
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#73d13d' }}>
                  {output.toLocaleString()}
                </div>
                <div style={{ color: '#d9d9d9' }}>輸出 Tokens</div>
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#faad14' }}>
                  {total.toLocaleString()}
                </div>
                <div style={{ color: '#d9d9d9' }}>總計 Tokens</div>
              </div>
            </Card>
          </Col>
        </Row>
      </Panel>
    )
  }

  return (
    <div className="technical-details">
      <Collapse ghost>
        {renderUsedChunks()}
        {renderTokenUsage()}
      </Collapse>
    </div>
  )
}

export default TechnicalDetailsComponent