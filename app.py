# app.py
import os
import asyncio
import base64
import threading

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import your inference pipeline from the inference package.
from inference import InferencePipeline
# We'll use a custom on_prediction callback (instead of render_boxes) to send frames.
# (Adjust this as needed based on what type of frame data your pipeline returns.)

app = FastAPI()

# Allow CORS so your front end (even if hosted separately) can connect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global set to keep track of connected websocket clients.
clients = set()
# A global variable to hold the latest rendered frame (as base64 JPEG string).
latest_frame = None

# Custom on_prediction callback.
# Here we assume that each prediction returns a rendered frame as JPEG bytes.
def on_prediction(frame):
    global latest_frame
    # Convert the frame (assumed bytes) to a base64-encoded string.
    # (If your pipeline already returns a base64 string, you can skip encoding.)
    if isinstance(frame, bytes):
        latest_frame = base64.b64encode(frame).decode('utf-8')
    else:
        latest_frame = frame

    # Send the latest frame to all connected clients.
    asyncio.run(broadcast_frame(latest_frame))
    # Return if needed.
    return

# Asynchronous function to broadcast the frame to all websockets.
async def broadcast_frame(frame: str):
    to_remove = []
    for ws in clients:
        try:
            await ws.send_text(frame)
        except Exception as e:
            print("Error sending frame:", e)
            to_remove.append(ws)
    for ws in to_remove:
        clients.remove(ws)

# Initialize the inference pipeline with your model ID and video source.
# Replace video_reference with your RTSP URL or "0" for the default webcam.
pipeline = InferencePipeline.init(
    model_id="rock-paper-scissors-sxsw/11",
    video_reference="0",  # or use "rtsp://username:password@your-ip/..." for RTSP
    on_prediction=on_prediction,
)

# Start the inference pipeline in a background thread on startup.
@app.on_event("startup")
async def startup_event():
    def run_pipeline():
        pipeline.start()
        pipeline.join()  # This call will block until the pipeline stops.
    threading.Thread(target=run_pipeline, daemon=True).start()

# Create a WebSocket endpoint that clients can connect to.
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        # Keep the connection alive.
        while True:
            # Optionally wait for a ping or a message from the client.
            await websocket.receive_text()
            # Optionally, send the latest frame on demand.
            if latest_frame:
                await websocket.send_text(latest_frame)
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        clients.remove(websocket)

if __name__ == "__main__":
    # Use the port provided by Render (or default to 8000).
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
