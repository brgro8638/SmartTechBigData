"""Rendering a slice to an encoded image, sending it over a data channel in chunks,
and the per-connection session that reacts to the phone's messages.

Protocol (all control messages are JSON text; image payloads are binary):

  phone -> server
    {"type": "hello"}                              ask for info + first image
    {"type": "set", "index": 12, "wc": 40, "ww": 400, "filter": "clahe",
                    "seg": "region", "seed": [x, y], "tol": 10, "fmt": "png"}
                                                   any subset of these keys
    {"type": "play", "fps": 8}  /  {"type": "pause"}

  server -> phone
    {"type": "info", "count": N, "w": W, "h": H, "wc": .., "ww": .., "filters": [...]}
    {"type": "image", "size": BYTES, "mime": "image/png", "index": .., "ms": ..}   header
    <binary chunks of at most CHUNK bytes, in order, until `size` bytes were sent>
"""
import asyncio
import json
import time
import traceback

import cv2

from . import dicom_io, filters, segmentation

CHUNK = 16 * 1024            # safe message size for every browser's data channel
HIGH_WATER = 1 * 1024 * 1024  # pause sending while this much is still queued

def render_frame(series, state):
    """Window -> filter -> segment/overlay -> encode. Returns (bytes, meta)."""
    t0 = time.perf_counter()
    gray = dicom_io.window(series.slices[state["index"]], state["wc"], state["ww"])
    gray = filters.apply_filter(gray, state["filter"])   # segmentation runs on the filtered image

    seg, seed = state["seg"], state["seed"]
    mask = None
    if seg == "region" and seed is not None:
        mask = segmentation.region_growing(gray, seed, state["tol"])
    elif seg == "watershed":
        mask = segmentation.watershed(gray, seed, state["tol"])

    if seg != "none":
        img = segmentation.overlay(gray, mask, seed)
    else:
        img = gray

    if state["fmt"] == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        mime = "image/jpeg"
    else:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        mime = "image/png"
    if not ok:
        raise RuntimeError("image encoding failed")

    h, w = gray.shape
    meta = {
        "mime": mime, "index": state["index"], "w": w, "h": h,
        "area": int((mask > 0).sum()) if mask is not None else 0,
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    return buf.tobytes(), meta

async def send_json(channel, obj):
    channel.send(json.dumps(obj))

async def send_image(channel, data, meta):
    """Header message, then the encoded image in CHUNK-sized binary messages."""
    channel.send(json.dumps({"type": "image", "size": len(data), **meta}))
    for i in range(0, len(data), CHUNK):
        while channel.bufferedAmount > HIGH_WATER: # don't pile up data if the phone is slow
            await asyncio.sleep(0.005)
        channel.send(data[i:i + CHUNK])

def apply_update(state, msg, count):
    """Merge the allowed keys from a 'set' message into the session state."""
    if "index" in msg:
        state["index"] = max(0, min(count - 1, int(msg["index"])))
    if "wc" in msg:
        state["wc"] = float(msg["wc"])
    if "ww" in msg:
        state["ww"] = max(1.0, float(msg["ww"]))
    if msg.get("filter") in filters.FILTERS:
        state["filter"] = msg["filter"]
    if msg.get("seg") in ("none", "region", "watershed"):
        state["seg"] = msg["seg"]
    if "seed" in msg:
        s = msg["seed"]
        state["seed"] = (int(s[0]), int(s[1])) if s else None
    if "tol" in msg:
        state["tol"] = max(1, min(100, int(msg["tol"])))
    if msg.get("fmt") in ("png", "jpeg"):
        state["fmt"] = msg["fmt"]

class Session:
    """One phone <-> server connection. Only the worker task ever sends on the channel,
    so image chunks can never interleave with other messages.

    Incoming messages only change the state and wake the worker. If several arrive while an
    image is being rendered (e.g. the user scrubs the slider), only the latest state is
    rendered next: stale intermediate frames are skipped.
    """

    def __init__(self, channel, series):
        self.channel = channel
        self.series = series
        self.state = {
            "index": series.count // 2, "wc": series.wc, "ww": series.ww,
            "filter": "none", "seg": "none", "seed": None, "tol": 10, "fmt": "png",
        }
        self.playing = False
        self.fps = 5.0
        self.need_info = False
        self.wake = asyncio.Event()
        self.task = None

    def start(self):
        self.task = asyncio.ensure_future(self._worker())

    def close(self):
        if self.task:
            self.task.cancel()

    def handle(self, text):
        try:
            msg = json.loads(text)
            kind = msg.get("type")
            if kind == "hello":
                self.need_info = True
            elif kind == "set":
                apply_update(self.state, msg, self.series.count)
            elif kind == "play":
                self.playing = True
                self.fps = max(1.0, min(30.0, float(msg.get("fps", 5))))
            elif kind == "pause":
                self.playing = False
            else:
                return
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            return   # ignore malformed messages
        self.wake.set()

    def _info(self):
        h, w = self.series.shape
        return {
            "type": "info", "count": self.series.count, "w": w, "h": h,
            "index": self.state["index"], "wc": self.series.wc, "ww": self.series.ww,
            "filters": list(filters.FILTERS), "source": self.series.source,
        }

    async def _worker(self):
        while True:
            if self.playing:
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=1.0 / self.fps)
                except asyncio.TimeoutError:   # no user input: advance the cine loop
                    self.state["index"] = (self.state["index"] + 1) % self.series.count
            else:
                await self.wake.wait()
            self.wake.clear()
            if self.channel.readyState != "open":
                return
            try:
                if self.need_info:
                    self.need_info = False
                    await send_json(self.channel, self._info())
                data, meta = await asyncio.to_thread(render_frame, self.series, dict(self.state))
                await send_image(self.channel, data, meta)
            except Exception:
                traceback.print_exc()
