"""
server.py
─────────
Retail Store Intelligence — Central Analytics Broker

Receives telemetry from edge inference nodes over HTTP POST, persists it
to SQLite, and broadcasts live updates to dashboard clients over WebSocket.

Endpoints:
    WS  /ws                        Real-time broadcast to dashboard
    POST /telemetry                Ingest from edge nodes
    GET  /snapshot                 Latest state for all cameras (dashboard cold-start)
    GET  /history?camera=X&minutes=60   Time-series from SQLite
    GET  /cameras                  List of known active cameras

Run:
    python backend/server.py
"""

import asyncio
import json
import time
import os
from datetime import datetime, timezone
from collections import defaultdict
from contextlib import asynccontextmanager
import aio_pika
import motor.motor_asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from reid_manager import ReIDManager

reid_manager = ReIDManager()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
mongo_client = None
db = None
telemetry_collection = None

async def init_db() -> None:
    global mongo_client, db, telemetry_collection
    mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
    db = mongo_client["analytics_db"]
    telemetry_collection = db["telemetry"]
    
    await telemetry_collection.create_index([("camera", 1), ("timestamp", 1)])
    await telemetry_collection.create_index("created_at", expireAfterSeconds=7200)

async def db_insert(camera: str, timestamp: float, payload: dict) -> None:
    await telemetry_collection.insert_one({
        "camera": camera,
        "timestamp": timestamp,
        "payload": payload,
        "created_at": datetime.now(timezone.utc)
    })

async def db_fetch_history(camera: str, minutes: int) -> list:
    """Return time-series rows for a camera covering the last N minutes."""
    since = time.time() - minutes * 60
    cursor = telemetry_collection.find(
        {"camera": camera, "timestamp": {"$gt": since}}
    ).sort("timestamp", 1)
    
    rows = await cursor.to_list(length=None)
    return [{"timestamp": r["timestamp"], **r["payload"]} for r in rows]

class ConnectionManager:
    """
    Thread-safe WebSocket connection registry with async broadcast.
    Handles dead-client cleanup automatically on send failure.
    """

    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, data: dict) -> None:
        payload = json.dumps(data)
        async with self._lock:
            dead: set[WebSocket] = set()
            for client in self._clients:
                try:
                    await client.send_text(payload)
                except Exception:
                    dead.add(client)
            self._clients -= dead

    @property
    def count(self) -> int:
        return len(self._clients)

manager = ConnectionManager()

latest_state: dict[str, dict] = {}

import os
AMQP_URL = os.getenv("AMQP_URL", "amqp://guest:guest@localhost:5672/")

def process_reid_sync(camera: str, signatures: dict, track_zones: dict) -> dict:
    """Synchronous wrapper to run ReID logic in a separate thread."""
    if signatures:
        reid_manager.resolve_identities(camera, signatures)
    if track_zones:
        reid_manager.update_global_journeys(camera, track_zones)
    
    return {
        "global_funnel": reid_manager.get_global_funnel(),
        "global_transitions": reid_manager.get_global_transitions(),
        "cross_camera_count": reid_manager.get_cross_camera_count()
    }

async def handle_telemetry_message(data: dict):
    """Processes a telemetry payload from RabbitMQ."""
    camera    = data.get("camera", "unknown")
    timestamp = data.get("timestamp", time.time())

    signatures = data.get("signatures", {})
    track_zones = data.get("track_zones", {})

    # Offload heavy CPU-bound ReID calculations to a thread 
    # to avoid blocking FastAPI's WebSocket broadcasts
    reid_results = await asyncio.to_thread(process_reid_sync, camera, signatures, track_zones)

    data["global_funnel"]        = reid_results["global_funnel"]
    data["global_transitions"]   = reid_results["global_transitions"]
    data["cross_camera_count"]   = reid_results["cross_camera_count"]

    # Strip heavy data before DB insertion and WebSocket broadcast
    data.pop("signatures", None)
    data.pop("track_zones", None)

    latest_state[camera] = data

    await db_insert(camera, timestamp, data)

    await manager.broadcast({"type": "update", "camera": camera, **data})

async def consume_telemetry():
    while True:
        try:
            connection = await aio_pika.connect_robust(AMQP_URL)
            async with connection:
                channel = await connection.channel()
                queue = await channel.declare_queue("telemetry", durable=True)
                print(f"Connected to RabbitMQ at {AMQP_URL}")

                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            data = json.loads(message.body.decode())
                            await handle_telemetry_message(data)
        except Exception as e:
            print(f"RabbitMQ consumer error: {e}, retrying in 5s...")
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    consumer_task = asyncio.create_task(consume_telemetry())
    yield
    consumer_task.cancel()

app = FastAPI(title="Multi Cam Analysis API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Persistent WebSocket for dashboard clients.
    Immediately hydrates new connections with the latest known snapshot.
    """
    await manager.connect(ws)

    if latest_state:
        try:
            await ws.send_text(json.dumps({
                "type":    "snapshot",
                "cameras": latest_state,
            }))
        except Exception:
            pass

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(ws)


@app.get("/snapshot")
async def get_snapshot():
    """
    Returns the most recent telemetry payload for every known camera.
    Used by the dashboard on first load to instantly populate KPIs and cards.
    """
    return JSONResponse({"cameras": latest_state})

@app.get("/history")
async def get_history(camera: str, minutes: int = 60):
    """
    Returns historical time-series data for one camera.

    Query params:
        camera  — Camera name (must match config.yaml camera.name)
        minutes — Lookback window in minutes (default: 60, max: 120)
    """
    minutes = min(minutes, 120)
    rows    = await db_fetch_history(camera, minutes)
    return JSONResponse({"camera": camera, "minutes": minutes, "count": len(rows), "rows": rows})

@app.get("/cameras")
async def get_cameras():
    """Returns the list of cameras that have sent at least one telemetry payload."""
    return JSONResponse({"cameras": list(latest_state.keys())})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
