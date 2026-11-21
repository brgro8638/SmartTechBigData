import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import numpy as np
import torch
import time
import colorsys
import tkinter as tk
from collections import deque
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

yolo_model = None

# Disse er tilgjengelige via dropdown
# For aktuelle klasser, se https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8
TARGET_CLASSES = ["car", "person", "bird"]

MOTION_SCALE_WIDTH = 160      # frames are downscaled to this width for motion detection
BLOCK_SIZE = 8                # block size (pixels, in the downscaled frame)
BLOCK_MOTION_FRACTION = 0.25  # fraction of a block's pixels that must be "changed" for the block to count as moving
MIN_MOVING_BLOCKS = 3         # blocks that must be moving to call the frame "motion"
MOTION_HOLD_FRAMES = 3        # keep running detection this many frames after motion stops

WEBCAM_INDEX = 0              # which camera cv2.VideoCapture opens

THEME = {
    "bg":       "#15171d",   # window background
    "panel":    "#1d2028",   # toolbar / status bar
    "field":    "#262a35",   # buttons, dropdown
    "hover":    "#313647",   # button hover
    "border":   "#323746",
    "text":     "#e8eaf0",
    "muted":    "#8d93a5",
    "accent":   "#2ec4b6",   # primary button, checkbox, focus
    "accent_h": "#4fd8cb",
    "danger":   "#e5484d",   # stop button
    "danger_h": "#f26a6e",
    "video_bg": "#0a0b0e",
}
UI_FONT = ("Segoe UI", 10)
MONO_FONT = ("Consolas", 10)

# Window / display
DEFAULT_WINDOW_W = 1200      # default window size; the video is scaled to fit the video area
DEFAULT_WINDOW_H = 800
DIFF_PANEL_WIDTH = 320        # width of each panel in the difference strip

# Differential frame display
DIFF_GAIN = 4                # brightness multiplier for the raw difference image (display only)
DIFF_MASK_THRESHOLD = 15      # pixel difference (0-255) above which a pixel counts as "changed" (used by both the mask display and motion detection)

DETECT_EVERY_N_FRAMES = 6     # run YOLO every N frames (while motion is active); LK carries the boxes in between.

LK_MAX_POINTS = 35            # feature points seeded inside each detected box
LK_MIN_POINTS = 3             # below this many surviving points, LK re-seeds the track
LK_FB_ERROR = 1.5             # forward-backward error (px) above which a point is discarded
LK_PARAMS = dict(winSize=(31, 31), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
IOU_MATCH_THRESHOLD = 0.2     # minimum overlap to link a new detection to an existing track
MAX_MISSED_DETECTIONS = 10    # drop a track after this many detection runs without a match
TRAIL_LENGTH = 30             # number of past centre points drawn per track
SHOW_LK_POINTS = True         # draw the tracked feature points
EDGE_TRIM = 0.35              # how much of the rectangle edge to trim when seeding features

def load_yolo_model():
    global yolo_model
    if yolo_model is not None:
        return yolo_model

    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
        print("Pretrained COCO model loaded")

        if torch.cuda.is_available():
            print("Using GPU")
            model.to('cuda')
        else:
            print("Using CPU")

        yolo_model = model
        return yolo_model

    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        return None

def compute_difference(prev_gray, curr_gray):
    """
    Frame differencing on two preprocessed (downscaled, blurred, grayscale) frames.

    Returns (diff, mask): the absolute difference image, and the binary change mask
    (255 where a pixel changed by more than DIFF_MASK_THRESHOLD).
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, mask = cv2.threshold(diff, DIFF_MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
    return diff, mask

def detect_motion(mask, bs=BLOCK_SIZE):
    """
    Decide whether there is motion from the change mask.

    The mask is divided into bs x bs blocks. A block is "moving" if at least
    BLOCK_MOTION_FRACTION of its pixels changed. The frame has motion when at least
    MIN_MOVING_BLOCKS blocks are moving, which ignores isolated noisy pixels.

    Returns (motion_detected: bool, moving_blocks: list of (x, y) block corners).
    """
    h, w = mask.shape
    gh, gw = h // bs, w // bs
    if gh == 0 or gw == 0:
        return False, []

    changed = (mask[:gh * bs, :gw * bs] > 0).reshape(gh, bs, gw, bs)
    block_fraction = changed.mean(axis=(1, 3))               # (gh, gw): share of changed pixels per block
    ys, xs = np.nonzero(block_fraction >= BLOCK_MOTION_FRACTION)
    moving = [(int(x) * bs, int(y) * bs) for y, x in zip(ys, xs)]

    return len(moving) >= MIN_MOVING_BLOCKS, moving

def preprocess_for_motion(frame):
    """Downscale, grayscale and blur a BGR frame for cheap, noise-tolerant motion search."""
    h, w = frame.shape[:2]
    scale = MOTION_SCALE_WIDTH / float(w)
    small = cv2.resize(frame, (MOTION_SCALE_WIDTH, max(1, int(h * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0), scale

def make_diff_display(diff, mask, panel_size):
    """
    Build the differential-frame view: the raw difference on the left and the
    thresholded change mask on the right, side by side as one RGB image.

    panel_size is (width, height) of each panel.
    """
    raw = cv2.convertScaleAbs(diff, alpha=DIFF_GAIN)   # amplified so faint changes are visible

    raw = cv2.resize(raw, panel_size, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, panel_size, interpolation=cv2.INTER_NEAREST)

    side_by_side = np.hstack([raw, mask])
    return cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2RGB)

def draw_motion_blocks(frame, moving_blocks, scale, bs=BLOCK_SIZE):
    """Overlay the moving blocks (in full-frame coordinates) for visual feedback."""
    inv = 1.0 / scale
    for bx, by in moving_blocks:
        x1, y1 = int(bx * inv), int(by * inv)
        x2, y2 = int((bx + bs) * inv), int((by + bs) * inv)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
    return frame

def detect_objects(frame, model, target_class, conf_threshold=0.5):
    """Run YOLO and return a list of {'box': [x1, y1, x2, y2], 'conf': float, 'name': str}."""
    results = model(frame)
    detections = results.pandas().xyxy[0]
    filtered = detections[(detections['name'] == target_class) & (detections['confidence'] > conf_threshold)]

    return [
        {'box': [float(r['xmin']), float(r['ymin']), float(r['xmax']), float(r['ymax'])],
         'conf': float(r['confidence']),
         'name': r['name']}
        for _, r in filtered.iterrows()
    ]

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _id_color(track_id):
    """Bright, fully saturated colour per ID. Golden-angle hue steps keep consecutive IDs far apart."""
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))

class LKTracker:
    """
    Tracks detected objects with pyramidal Lucas-Kanade optical flow.

    - Each track owns a set of corner points inside its box.
    - Every frame, LK moves those points and the box is shifted by their median motion.
    - Every time YOLO runs, detections are matched to tracks by IoU. Matched tracks keep
      their ID (and get re-seeded points); unmatched detections start new tracks.
    """
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def _seed_points(gray, box):
        h, w = gray.shape
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * EDGE_TRIM, (y2 - y1) * EDGE_TRIM     
        x1, y1 = int(max(0, x1 + mx)), int(max(0, y1 + my))
        x2, y2 = int(min(w, x2 - mx)), int(min(h, y2 - my))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return np.empty((0, 1, 2), np.float32)
        pts = cv2.goodFeaturesToTrack(gray[y1:y2, x1:x2], maxCorners=LK_MAX_POINTS,
                                      qualityLevel=0.01, minDistance=5)
        if pts is None:
            return np.empty((0, 1, 2), np.float32)
        return (pts + np.array([x1, y1], np.float32)).astype(np.float32)

    def _new_track(self, det, gray):
        tid = self.next_id
        self.next_id += 1
        box = list(det['box'])
        self.tracks[tid] = {
            'id': tid, 'name': det['name'], 'conf': det['conf'], 'box': box,
            'pts': self._seed_points(gray, box), 'missed': 0, 'detected': True,
            'trail': deque([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)], maxlen=TRAIL_LENGTH),
        }

    def update_flow(self, prev_gray, curr_gray):
        """Move every track with Lucas-Kanade optical flow (called every frame)."""
        h, w = curr_gray.shape
        for t in self.tracks.values():
            t['detected'] = False
            pts = t['pts']
            if len(pts) == 0:
                t['pts'] = self._seed_points(curr_gray, t['box'])
                continue

            new, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None, **LK_PARAMS)
            back, st_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, new, None, **LK_PARAMS)
            fb_err = np.linalg.norm(pts - back, axis=2).reshape(-1)
            good = (st.reshape(-1) == 1) & (st_back.reshape(-1) == 1) & (fb_err < LK_FB_ERROR)

            if good.sum() >= LK_MIN_POINTS:
                dx, dy = np.median(new[good] - pts[good], axis=0).reshape(2)
                b = t['box']
                t['box'] = [float(np.clip(b[0] + dx, 0, w - 1)), float(np.clip(b[1] + dy, 0, h - 1)),
                            float(np.clip(b[2] + dx, 0, w - 1)), float(np.clip(b[3] + dy, 0, h - 1))]
                t['pts'] = new[good].reshape(-1, 1, 2)
            else:
                # too few points survived: keep the box where it is and try to re-seed
                t['pts'] = self._seed_points(curr_gray, t['box'])

            b = t['box']
            t['trail'].append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))

    def update_detections(self, detections, gray):
        """Associate fresh YOLO detections with existing tracks (greedy IoU, same class)."""
        pairs = []
        for tid, t in self.tracks.items():
            for j, det in enumerate(detections):
                if det['name'] != t['name']:
                    continue
                iou = _iou(t['box'], det['box'])
                if iou >= IOU_MATCH_THRESHOLD:
                    pairs.append((iou, tid, j))
        pairs.sort(reverse=True)

        used_tracks, used_dets = set(), set()
        for iou, tid, j in pairs:
            if tid in used_tracks or j in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(j)
            t, det = self.tracks[tid], detections[j]
            t['box'] = list(det['box'])
            t['conf'] = det['conf']
            t['missed'] = 0
            t['detected'] = True
            t['pts'] = self._seed_points(gray, t['box'])

        for tid in list(self.tracks):
            if tid not in used_tracks:
                self.tracks[tid]['missed'] += 1
                if self.tracks[tid]['missed'] > MAX_MISSED_DETECTIONS:
                    del self.tracks[tid]

        for j, det in enumerate(detections):
            if j not in used_dets:
                self._new_track(det, gray)

    def draw(self, frame):
        """Draw boxes with IDs, trails and LK points. Bold box = confirmed by YOLO this frame."""
        for t in self.tracks.values():
            color = _id_color(t['id'])
            x1, y1, x2, y2 = (int(v) for v in t['box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if t['detected'] else 1)

            label = f"#{t['id']} {t['name']} {t['conf']:.2f}"
            if not t['detected']:
                label += " (LK)"
            cv2.putText(frame, label, (x1, y1 - 8 if y1 > 20 else y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if len(t['trail']) > 1:
                trail = np.array(t['trail'], np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [trail], False, color, 1)

            if SHOW_LK_POINTS:
                for p in t['pts'].reshape(-1, 2):
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, color, -1)
        return frame

class ToggleCheck(tk.Frame):
    """
    A checkbox drawn on a canvas. The ttk checkbox's tick colour is theme-internal and
    can end up invisible on a dark theme, so this one draws its own box and tick.
    """
    SIZE = 18

    def __init__(self, master, label_text, variable, command=None):
        super().__init__(master, bg=THEME["panel"], cursor="hand2")
        self.variable = variable
        self.command = command
        self._hover = False

        self.canvas = tk.Canvas(self, width=self.SIZE, height=self.SIZE, bg=THEME["panel"],
                                highlightthickness=0, bd=0)
        self.canvas.pack(side=tk.LEFT)
        self.label = tk.Label(self, text=label_text, bg=THEME["panel"], fg=THEME["text"], font=UI_FONT)
        self.label.pack(side=tk.LEFT, padx=(8, 0))

        for widget in (self, self.canvas, self.label):
            widget.bind("<Button-1>", self.toggle)
            widget.bind("<Enter>", lambda e: self._set_hover(True))
            widget.bind("<Leave>", lambda e: self._set_hover(False))

        variable.trace_add("write", lambda *args: self.redraw())   # stays in sync if set from code
        self.redraw()

    def toggle(self, event=None):
        self.variable.set(not self.variable.get())
        if self.command:
            self.command()

    def _set_hover(self, hover):
        self._hover = hover
        self.redraw()

    def redraw(self):
        t, s = THEME, self.SIZE
        on = self.variable.get()
        fill = t["accent"] if on else (t["hover"] if self._hover else t["field"])
        outline = t["accent_h"] if self._hover else (t["accent"] if on else t["border"])
        self.canvas.delete("all")
        self.canvas.create_rectangle(1, 1, s - 2, s - 2, fill=fill, outline=outline, width=1)
        if on:
            self.canvas.create_line(4, 9, 8, 13, 14, 5, fill=t["bg"], width=2,
                                    capstyle=tk.ROUND, joinstyle=tk.ROUND)

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Motion Detection & Object Tracking + Counting")
        self.pack(fill=tk.BOTH, expand=True)
        self.master.geometry(f"{DEFAULT_WINDOW_W}x{DEFAULT_WINDOW_H}")
        self.master.minsize(640, 480)

        self.video_path = None
        self.cap = None
        self.is_webcam = False
        self.after_id = None
        self.fps_label = None
        self.motion_label = None
        self.video_label = None
        self.diff_label = None
        self.frame_w = 640
        self.frame_h = 480
        self.show_diff = tk.BooleanVar(master=master, value=False)
        self.class_var = tk.StringVar(master=master, value=TARGET_CLASSES[0])

        self.prev_gray = None
        self.prev_full_gray = None
        self.hold_counter = 0
        self.tracker = LKTracker()
        self.frames_since_detect = DETECT_EVERY_N_FRAMES
        self.track_label = None

        self.model = load_yolo_model()
        if self.model is None:
            self.master.destroy()
            return

        self.create_widgets()

        # Esc stops the running stream (webcam or video); closing the window cleans up too
        self.master.bind("<Escape>", lambda event: self.stop_capture())
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_style(self):
        """Dark, flat look built on ttk's 'clam' theme (no extra packages needed)."""
        t = THEME
        style = ttk.Style(self.master)
        style.theme_use("clam")

        self.master.configure(bg=t["bg"])
        self.configure(bg=t["bg"])
        # Colours for the dropdown's popup list (a plain Tk listbox, so styled via the option database)
        self.master.option_add("*TCombobox*Listbox.background", t["field"])
        self.master.option_add("*TCombobox*Listbox.foreground", t["text"])
        self.master.option_add("*TCombobox*Listbox.selectBackground", t["accent"])
        self.master.option_add("*TCombobox*Listbox.selectForeground", t["bg"])
        self.master.option_add("*TCombobox*Listbox.font", UI_FONT)

        style.configure(".", background=t["panel"], foreground=t["text"], font=UI_FONT,
                        bordercolor=t["border"], focuscolor=t["panel"])
        style.configure("TFrame", background=t["panel"])
        style.configure("TLabel", background=t["panel"], foreground=t["text"])
        style.configure("Muted.TLabel", foreground=t["muted"])
        style.configure("Status.TLabel", foreground=t["text"], font=MONO_FONT)

        # Buttons: flat rectangles with a hover colour
        base = dict(padding=(14, 7), borderwidth=0, relief="flat", focusthickness=0)
        style.configure("TButton", background=t["field"], foreground=t["text"], **base)
        style.map("TButton", background=[("pressed", t["border"]), ("active", t["hover"])])
        style.configure("Accent.TButton", background=t["accent"], foreground=t["bg"],
                        font=(UI_FONT[0], UI_FONT[1], "bold"), **base)
        style.map("Accent.TButton", background=[("pressed", t["accent"]), ("active", t["accent_h"])])
        style.configure("Danger.TButton", background=t["danger"], foreground="#ffffff", **base)
        style.map("Danger.TButton", background=[("pressed", t["danger"]), ("active", t["danger_h"])])

        # Dropdown
        style.configure("TCombobox", fieldbackground=t["field"], background=t["field"],
                        foreground=t["text"], arrowcolor=t["text"], bordercolor=t["field"],
                        lightcolor=t["field"], darkcolor=t["field"], padding=5)
        style.map("TCombobox",
                  fieldbackground=[("readonly", t["field"])],
                  foreground=[("readonly", t["text"])],
                  selectbackground=[("readonly", t["field"])],
                  selectforeground=[("readonly", t["text"])],
                  bordercolor=[("focus", t["accent"])],
                  arrowcolor=[("active", t["accent"])])

    def create_widgets(self):
        t = THEME
        self.setup_style()

        # --- Toolbar (top) ---
        toolbar = ttk.Frame(self, padding=(14, 10))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Select Video", style="Accent.TButton",
                   command=self.select_video_file).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Use Webcam",
                   command=self.start_webcam).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Stop  (Esc)", style="Danger.TButton",
                   command=self.stop_capture).pack(side=tk.LEFT, padx=(0, 8))

        # Options sit on the right of the toolbar
        ToggleCheck(toolbar, "Show difference frames", self.show_diff,
                    command=self.on_diff_toggle).pack(side=tk.RIGHT)
        class_menu = ttk.Combobox(toolbar, textvariable=self.class_var, values=TARGET_CLASSES,
                                  state="readonly", width=8)
        class_menu.pack(side=tk.RIGHT, padx=(6, 22))
        ttk.Label(toolbar, text="Detect", style="Muted.TLabel").pack(side=tk.RIGHT)

        # --- Status bar (bottom). Fixed-width monospace labels so nothing shifts as numbers change ---
        status = ttk.Frame(self, padding=(14, 6))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self.fps_label = ttk.Label(status, text="FPS: 0.00", width=12, anchor="w", style="Status.TLabel")
        self.fps_label.pack(side=tk.LEFT, padx=(0, 18))
        self.track_label = ttk.Label(status, text="Tracks: 0", width=10, anchor="w", style="Status.TLabel")
        self.track_label.pack(side=tk.LEFT, padx=(0, 18))
        self.motion_label = ttk.Label(status, text="Motion: -", width=34, anchor="w", style="Status.TLabel")
        self.motion_label.pack(side=tk.LEFT)

        # --- Video area: grows with the window. The label is centred inside it and never
        # influences its size, so each frame can be scaled to fit the area. ---
        self.video_area = tk.Frame(self, bg=t["video_bg"], highlightthickness=1,
                                   highlightbackground=t["border"], highlightcolor=t["border"])
        self.video_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=12)
        self.video_area.pack_propagate(False)
        self.video_label = tk.Label(self.video_area, bg=t["video_bg"])
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # Error message, shown in the middle of the video area (see show_error)
        self.error_label = tk.Label(self.video_area, bg=t["video_bg"], fg=t["danger_h"],
                                    font=(UI_FONT[0], 12), justify="center")

        # Differential frames (raw difference | thresholded mask); packed below the
        # video only while the checkbox is on (see on_diff_toggle)
        self.diff_label = tk.Label(self, bg=t["bg"])

    def show_error(self, message):
        """Show an error message in the middle of the video area (and in the console)."""
        print(f"Error: {message}")
        wrap = max(200, self.video_area.winfo_width() - 80)
        self.error_label.config(text=message, wraplength=wrap)
        self.error_label.place(relx=0.5, rely=0.5, anchor="center")

    def clear_error(self):
        self.error_label.place_forget()

    def on_diff_toggle(self):
        if self.show_diff.get():
            self.diff_label.pack(side=tk.BOTTOM, padx=12, pady=(0, 12), before=self.video_area)
        else:
            self.diff_label.pack_forget()
            self.diff_label.config(image="")
            self.diff_label.imgtk = None

    def stop_capture(self):
        """Stop the current stream (video or webcam) and release the device."""
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
        if self.video_label is not None:
            self.video_label.config(image="")
            self.video_label.imgtk = None
        if self.diff_label is not None:
            self.diff_label.config(image="")
            self.diff_label.imgtk = None
        if self.fps_label is not None:
            self.fps_label.config(text="FPS: 0.00")
        if self.motion_label is not None:
            self.motion_label.config(text="Motion: -")
        if self.track_label is not None:
            self.track_label.config(text="Tracks: 0")
        self.prev_gray = None
        self.prev_full_gray = None
        self.hold_counter = 0
        self.tracker.reset()
        self.frames_since_detect = DETECT_EVERY_N_FRAMES

    def start_capture(self, source, is_webcam):
        """Open `source` (file path or camera index) and start the frame loop."""
        self.stop_capture()
        self.clear_error()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            if is_webcam:
                self.show_error(
                    f"Could not open the webcam (camera index {source}).\n\n"
                    "Check that a camera is connected, that no other application is using it, "
                    "and that this app is allowed to access it.\n"
                    "If you have several cameras, change WEBCAM_INDEX at the top of the script.")
            else:
                self.show_error(
                    f"Could not open the video file:\n{source}\n\n"
                    "The file may be missing, damaged, or in a format OpenCV cannot read.")
            return False

        self.cap = cap
        self.is_webcam = is_webcam

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self.frame_w, self.frame_h = width, height

        # The dropdown is read once here; changing it later applies on the next start
        self.show_frame(self.class_var.get())
        return True

    def select_video_file(self):
        file_path = filedialog.askopenfilename(
            initialdir=".",
            title="Select a video file",
            filetypes=(("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
        )

        if file_path:
            self.video_path = file_path
            print(f"Selected video: {self.video_path}")
            if self.start_capture(self.video_path, is_webcam=False):
                print("Video file opened successfully!")

    def start_webcam(self):
        if self.start_capture(WEBCAM_INDEX, is_webcam=True):
            print("Webcam opened. Press Esc to stop.")

    def on_close(self):
        self.stop_capture()
        self.master.destroy()

    def fit_to_video_area(self, frame):
        """Scale a frame, keeping its aspect ratio, to fit the video area (display only)."""
        avail_w = self.video_area.winfo_width()
        avail_h = self.video_area.winfo_height()
        h, w = frame.shape[:2]
        if avail_w <= 1 or avail_h <= 1:      # window not laid out yet
            return frame
        scale = min(avail_w / w, avail_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        if (new_w, new_h) == (w, h):
            return frame
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        return cv2.resize(frame, (new_w, new_h), interpolation=interp)

    def show_frame(self, target_class):
        if self.cap and self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()

            if ret:
                # --- Step 1: motion detection by frame differencing ---
                curr_gray, scale = preprocess_for_motion(frame)
                curr_full_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   # full-res gray for Lucas-Kanade
                motion = False
                moving_blocks = []
                if self.prev_gray is not None and self.prev_gray.shape == curr_gray.shape:
                    diff, mask = compute_difference(self.prev_gray, curr_gray)
                    motion, moving_blocks = detect_motion(mask)

                    # --- Differential frames (only drawn when enabled) ---
                    if self.show_diff.get():
                        panel = (DIFF_PANEL_WIDTH, max(1, int(DIFF_PANEL_WIDTH * self.frame_h / self.frame_w)))
                        diff_rgb = make_diff_display(diff, mask, panel)
                        diff_tk = ImageTk.PhotoImage(image=Image.fromarray(diff_rgb))
                        self.diff_label.imgtk = diff_tk
                        self.diff_label.config(image=diff_tk)
                self.prev_gray = curr_gray

                # Keep detecting briefly after motion stops so boxes don't flicker
                if motion:
                    self.hold_counter = MOTION_HOLD_FRAMES
                elif self.hold_counter > 0:
                    self.hold_counter -= 1

                run_detection = motion or self.hold_counter > 0

                # --- Step 2: Lucas-Kanade tracking (every frame, cheap) ---
                if self.prev_full_gray is not None and self.prev_full_gray.shape == curr_full_gray.shape:
                    self.tracker.update_flow(self.prev_full_gray, curr_full_gray)

                # --- Step 3: object detection only when needed ---
                self.frames_since_detect += 1
                if run_detection and self.frames_since_detect >= DETECT_EVERY_N_FRAMES:
                    detections = detect_objects(frame, self.model, target_class)
                    self.tracker.update_detections(detections, curr_full_gray)
                    self.frames_since_detect = 0
                self.prev_full_gray = curr_full_gray

                frame_processed = self.tracker.draw(frame)
                self.track_label.config(text=f"Tracks: {len(self.tracker.tracks)}")

                if moving_blocks:
                    frame_processed = draw_motion_blocks(frame_processed, moving_blocks, scale)

                frame_processed = self.fit_to_video_area(frame_processed)
                frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)

                end_time = time.time()
                fps = 1 / max(end_time - start_time, 1e-6)
                self.fps_label.config(text=f"FPS: {fps:.2f}")
                if motion:
                    self.motion_label.config(text=f"Motion: yes ({len(moving_blocks)} blocks)")
                elif run_detection:
                    self.motion_label.config(text="Motion: no (detecting, cooldown)")
                else:
                    self.motion_label.config(text="Motion: no (detection idle)")

                self.after_id = self.after(1 if self.is_webcam else 10,
                                           lambda: self.show_frame(target_class))
            else:
                was_webcam = self.is_webcam
                self.stop_capture()
                if was_webcam:
                    self.show_error("The webcam stopped delivering frames.\n\n"
                                    "It may have been disconnected or taken over by another application.")
                else:
                    print("Video stream ended.")

def main():
    root = tk.Tk()
    app = Application(master=root)
    root.mainloop()

if __name__ == "__main__":
    main()