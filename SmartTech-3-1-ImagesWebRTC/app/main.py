"""FastAPI app: serves the page, does WebRTC signaling (POST /offer) and hooks the data channel
up to a Session."""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import dicom_io
from .stream import Session

STATIC = Path(__file__).resolve().parent.parent / "static"
pcs = set()
series = None

@asynccontextmanager
async def lifespan(app):
    global series
    folder = os.environ.get("DICOM_DIR")
    if folder:
        series = dicom_io.load_series(folder)
        if series is None:
            print(f"[dicom] no DICOM slices found in {folder!r}, using the demo phantom instead")
    if series is None:
        series = dicom_io.make_demo_series()
    print(f"[dicom] {series.source}; {series.count} slices of {series.shape[1]}x{series.shape[0]}")
    yield
    await asyncio.gather(*(pc.close() for pc in pcs))
    pcs.clear()

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    # No STUN/TURN servers: enough for phone and laptop on the same network.
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("datachannel")
    def on_datachannel(channel):
        session = Session(channel, series)
        session.start()

        @channel.on("message")
        def on_message(message):
            if isinstance(message, str):
                session.handle(message)

        @channel.on("close")
        def on_close():
            session.close()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("[webrtc] connection state:", pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
