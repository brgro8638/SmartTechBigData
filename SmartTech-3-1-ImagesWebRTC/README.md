# DICOM image streaming over WebRTC (minimal version)

A Python backend renders DICOM slices (window/level, filters, segmentation) and streams the
resulting images to a phone browser over a WebRTC **data channel**.

## Run

    pip install -r requirements.txt
    python run.py                         # synthetic demo phantom, no data needed
    python run.py --dicom path/to/dicoms  # your own DICOM folder

Open the printed `http://<laptop-ip>:8000` URL on the phone (same Wi-Fi or phone hotspot).
Allow Python through the laptop firewall on private networks.

## Files

    run.py               start script (prints the URL for the phone)
    app/main.py          FastAPI + POST /offer (WebRTC signaling) + data channel wiring
    app/stream.py        render_frame(), chunked image sending, per-connection Session + protocol
    app/dicom_io.py      DICOM loading, rescale to HU, window/level, demo phantom
    app/filters.py       gaussian, median, bilateral, CLAHE, unsharp, Sobel, Canny
    app/segmentation.py  region growing, watershed (seeded + automatic), overlay
    static/              index.html, app.js, style.css (phone UI)

## If the page loads but never connects

1. The network may block device-to-device traffic (eduroam and similar). Use your phone's hotspot.
2. Firewall: allow Python on private networks.
3. Phones often hide their IP behind an mDNS `.local` name. If it still fails on a normal LAN,
   try Chrome on the laptop first (localhost), then look at `[webrtc] connection state` in the
   server log.
