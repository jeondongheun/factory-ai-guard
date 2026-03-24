import axios from 'axios';

// 환경변수 우선, 없으면 상대경로(nginx 프록시) 사용
const API_BASE = process.env.REACT_APP_API_URL || '';

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 10000,
});

export const getSensorCurrent = () => api.get('/api/sensors/current');
export const getSensorHistory = (limit = 100) => api.get(`/api/sensors/history?limit=${limit}`);
export const analyzeDetection = (sensorData) => api.post('/api/detection/analyze', { sensor_data: sensorData });
export const getDetectionHistory = (limit = 50, offset = 0) => api.get(`/api/detection/history?limit=${limit}&offset=${offset}`);
export const getLLMDiagnosis = (detectionId, sensorStats) => api.post('/api/diagnosis/llm', { detection_id: detectionId, sensor_stats: sensorStats });
export const getStatsSummary = () => api.get('/api/stats/summary');
export const getStatsTrend = (days = 7) => api.get(`/api/stats/trend?days=${days}`);
export const getSensorAvg = () => api.get('/api/stats/sensor-avg');
export const getModelMetrics = () => api.get('/api/model/metrics');
export const getModelInfo = () => api.get('/api/model/info');
export const uploadCSV = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/api/upload/csv', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
};
export const getUploadResult = (jobId) => api.get(`/api/upload/result/${jobId}`);
export const getThresholds = () => api.get('/api/settings/thresholds');
export const updateThreshold = (data) => api.put('/api/settings/thresholds', data);

// 챗봇
export const sendChatMessage = (message, sessionId = null) =>
  api.post('/api/chat/message', { message, session_id: sessionId }, { timeout: 30000 });
export const clearChatSession = (sessionId) =>
  api.delete(`/api/chat/session/${sessionId}`);

// WebSocket URL: 환경변수 우선, 없으면 현재 호스트 기준
const _wsBase = process.env.REACT_APP_WS_URL ||
  (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host;
export const WS_URL = _wsBase + '/api/sensors/ws/realtime';