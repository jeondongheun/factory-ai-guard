from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from app.services.websocket_manager import ws_manager
import asyncio

router = APIRouter()

@router.get("/current")
async def get_current_sensor(request: Request):
    """현재 센서값 반환"""
    simulator = request.app.state.simulator
    return simulator.get_current()

@router.get("/history")
async def get_sensor_history(request: Request, limit: int = 100):
    """최근 센서 버퍼 반환"""
    simulator = request.app.state.simulator
    buffer = list(simulator.buffer)[-limit:]
    return {"data": buffer, "count": len(buffer)}

@router.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """실시간 센서 데이터 WebSocket 스트림"""
    await ws_manager.connect(websocket)
    simulator = websocket.app.state.simulator
    ml_service = websocket.app.state.ml_service

    try:
        while True:
            # 새 센서값 생성
            sensor_data = simulator.get_next()

            # 윈도우로 ML 추론
            window = simulator.get_window()
            detection = ml_service.predict(window)

            # 합쳐서 전송
            payload = {
                "sensor": sensor_data,
                "detection": detection
            }

            await ws_manager.broadcast(payload)
            await asyncio.sleep(1)  # 1초마다 전송

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
