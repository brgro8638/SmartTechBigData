"""Start the server and print the URL to open on the phone.

    python run.py                       # synthetic demo dicoms
    python run.py --dicom path/to/dicoms
"""
import argparse
import os
import socket
import uvicorn

def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no packet is sent; just picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom", help="folder with DICOM files (default: synthetic demo phantom)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.dicom:
        os.environ["DICOM_DIR"] = args.dicom
    print(f"\n  Open on your phone (same Wi-Fi):  http://{lan_ip()}:{args.port}\n")
    uvicorn.run("app.main:app", host="0.0.0.0", port=args.port)
