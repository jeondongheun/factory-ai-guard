from fastapi import WebSocket
from typing import List
import json

class WebSocketManager:
    """WebSocket 연결 관리"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ WebSocket 연결 (+{len(self.active_connections)}개 활성)")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"👋 WebSocket 해제 ({len(self.active_connections)}개 활성)")

    async def broadcast(self, data: dict):
        """모든 연결된 클라이언트에게 데이터 전송"""
        if not self.active_connections:
            return

        message = json.dumps(data, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)

        # 끊어진 연결 정리
        for conn in disconnected:
            self.active_connections.remove(conn)

# 전역 인스턴스
ws_manager = WebSocketManager()
