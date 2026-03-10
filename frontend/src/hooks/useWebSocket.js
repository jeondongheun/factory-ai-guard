import { useState, useEffect, useRef, useCallback } from 'react';
import { WS_URL } from '../utils/api';

export const useWebSocket = () => {
  const [sensorData, setSensorData] = useState(null);
  const [detection, setDetection] = useState(null);
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState([]);
  const wsRef = useRef(null);

  const connect = useCallback(() => {
    try {
      wsRef.current = new WebSocket(WS_URL);

      wsRef.current.onopen = () => {
        setConnected(true);
        console.log('✅ WebSocket 연결됨');
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setSensorData(data.sensor);
        setDetection(data.detection);
        setHistory(prev => [...prev, data.sensor].slice(-60));
      };

      wsRef.current.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000);
      };

      wsRef.current.onerror = (error) => {
        console.error('❌ WebSocket 오류:', error);
      };

    } catch (error) {
      console.error('WebSocket 연결 실패:', error);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return { sensorData, detection, connected, history };
};