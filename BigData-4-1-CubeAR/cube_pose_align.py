"""
Box Alignment (PySide6)

Manual click alignment + SIFT (RootSIFT) feature re-alignment.
1. Click the box's corners on a paused frame and give each the matching model corner index.
2. "Solve" computes the reference pose (solvePnP) and stores SIFT features back-projected onto the box.
3. On any other frame, "Re-align" matches SIFT features against that reference and re-solves the pose.
Optional auto re-align every N frames while playing.
"""

import sys

if (sys.platform == "linux"):
    import os
    os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import trimesh
from auto_align import BoxFitter, segment_box
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

VIDEO_PATH = "./cube.mp4"
OBJ_PATH = "./box.obj"

# Focal length (px) tuned for an 848 px wide frame; scaled to the real video width at load time.
FOCAL_AT_848 = 800.0
DIST_COEFFS = np.zeros(5)

DEFAULT_AUTO_EVERY = 3
SIFT_FEATURES = 1500
RATIO = 0.7

THEME = {
    "bg": "#15171d",
    "panel": "#1d2028",
    "field": "#262a35",
    "hover": "#313647",
    "border": "#323746",
    "text": "#e8eaf0",
    "muted": "#8d93a5",
    "accent": "#2ec4b6",
    "accent_h": "#4fd8cb",
    "danger": "#e5484d",
    "danger_h": "#f26a6e",
    "video_bg": "#0a0b0e",
}

STYLESHEET = """
* {{ font-family: "Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", sans-serif; font-size: 13px; }}
QMainWindow, QWidget#root {{ background: {bg}; }}
QLabel {{ color: {text}; background: transparent; }}
QLabel[role="muted"] {{ color: {muted}; }}
QLabel[role="section"] {{ color: {muted}; font-size: 11px; font-weight: 600; letter-spacing: 1.5px; }}
QLabel[role="mono"] {{ font-family: "JetBrains Mono", "Cascadia Mono", "Consolas", monospace; color: {muted}; }}
QLabel[role="value"] {{ color: {text}; font-weight: 600; }}

QFrame#card {{ background: {panel}; border: 1px solid {border}; border-radius: 14px; }}
QFrame#videoCard {{ background: {video_bg}; border: 1px solid {border}; border-radius: 14px; }}
QFrame#divider {{ background: {border}; max-height: 1px; min-height: 1px; border: none; }}

QPushButton {{
    background: {field}; color: {text}; border: 1px solid {border};
    border-radius: 9px; padding: 8px 14px; font-weight: 500;
}}
QPushButton:hover {{ background: {hover}; }}
QPushButton:pressed {{ background: {border}; }}
QPushButton:disabled {{ color: {muted}; background: {panel}; }}
QPushButton:checked {{ background: {hover}; border: 1px solid {accent}; color: {accent_h}; }}
QPushButton[variant="accent"] {{ background: {accent}; color: #0a0b0e; border: none; font-weight: 600; }}
QPushButton[variant="accent"]:hover {{ background: {accent_h}; }}
QPushButton[variant="danger"] {{ background: {danger}; color: white; border: none; font-weight: 600; }}
QPushButton[variant="danger"]:hover {{ background: {danger_h}; }}
QPushButton[variant="ghost"] {{ background: transparent; border: 1px solid {border}; color: {danger_h}; }}
QPushButton[variant="ghost"]:hover {{ background: rgba(229,72,77,0.12); border-color: {danger}; }}
QPushButton[variant="icon"] {{ padding: 8px 0; min-width: 38px; max-width: 38px; font-size: 15px; }}

QSlider::groove:horizontal {{ height: 6px; background: {field}; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: {text}; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px;
    border: 2px solid {accent};
}}
QSlider::handle:horizontal:hover {{ background: white; }}

QCheckBox {{ color: {text}; spacing: 10px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid {border}; background: {field}; }}
QCheckBox::indicator:hover {{ border-color: {accent}; }}
QCheckBox::indicator:checked {{ background: {accent}; border-color: {accent}; }}

QSpinBox {{
    background: {field}; color: {text}; border: 1px solid {border};
    border-radius: 8px; padding: 5px 8px; min-width: 52px;
}}
QSpinBox:focus {{ border-color: {accent}; }}
QSpinBox::up-button, QSpinBox::down-button {{ width: 0; border: none; }}

QLabel#status {{ background: {panel}; border: 1px solid {border}; border-radius: 10px; padding: 9px 14px; color: {muted}; }}
QLabel#status[level="ok"] {{ color: {accent_h}; }}
QLabel#status[level="warn"] {{ color: #f5b942; }}
QLabel#status[level="error"] {{ color: {danger_h}; }}

QToolTip {{ background: {panel}; color: {text}; border: 1px solid {border}; padding: 6px; }}
QMessageBox {{ background: {panel}; }}
""".format(**THEME)

def repolish(widget):
    """Re-apply the stylesheet after a dynamic property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)

def make_label(text="", role=None):
    lab = QLabel(text)
    if role:
        lab.setProperty("role", role)
    return lab

def make_button(text, variant=None, checkable=False, tooltip=None):
    b = QPushButton(text)
    if variant:
        b.setProperty("variant", variant)
    b.setCheckable(checkable)
    b.setCursor(Qt.PointingHandCursor)
    if tooltip:
        b.setToolTip(tooltip)
    return b

class VideoView(QWidget):
    """Shows a frame scaled to fit (aspect preserved) and reports clicks in image coordinates."""

    clicked = Signal(float, float)
    undo_requested = Signal()

    def __init__(self):
        super().__init__()
        self.image = None
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, qimg):
        self.image = qimg
        self.update()

    def set_pick_mode(self, on):
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def _target(self):
        iw, ih = self.image.width(), self.image.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h), s

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(THEME["video_bg"]))
        if self.image is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            rect, _s = self._target()
            p.drawImage(rect, self.image)

    def mousePressEvent(self, e):
        if self.image is None:
            return
        rect, s = self._target()
        pos = e.position()
        if e.button() == Qt.LeftButton and rect.contains(pos):
            self.clicked.emit((pos.x() - rect.x()) / s, (pos.y() - rect.y()) / s)
        elif e.button() == Qt.RightButton:
            self.undo_requested.emit()

class MainWindow(QMainWindow):
    FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

    def __init__(self, video_path, obj_path):
        super().__init__()
        self.setWindowTitle("Box Alignment")
        self.resize(1240, 760)

        #region Model
        mesh = trimesh.load(obj_path, force="mesh", process=False)
        b = mesh.bounds
        self.corners_3d = np.float32([
            [b[0, 0], b[0, 1], b[0, 2]], [b[1, 0], b[0, 1], b[0, 2]],
            [b[1, 0], b[1, 1], b[0, 2]], [b[0, 0], b[1, 1], b[0, 2]],
            [b[0, 0], b[0, 1], b[1, 2]], [b[1, 0], b[0, 1], b[1, 2]],
            [b[1, 0], b[1, 1], b[1, 2]], [b[0, 0], b[1, 1], b[1, 2]],
        ])
        self.box_min, self.box_max = b[0], b[1]

        #region Features
        self.sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        #region Video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        focal = FOCAL_AT_848 * img_w / 848.0
        self.K = np.array([[focal, 0, img_w / 2.0], [0, focal, img_h / 2.0], [0, 0, 1]], dtype=np.float32)
        self.fitter = BoxFitter(self.box_min, self.box_max, self.K)

        #region State
        self.idx = 0
        self.frame = None
        self.rvec = self.tvec = None
        self.show_overlay = False
        self.click_mode = False
        self.clicks = []  # (x, y, corner_idx) in image pixels
        self.ref_des = None
        self.ref_pts3d = None
        self.frame_counter = 0
        self.last_inliers = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_step)
        self.timer.setInterval(max(20, int(1000 / self.fps)))

        self._build_ui()
        self._bind_shortcuts()
        self.load_frame(0)
        self.notify("Ready. Use Manual Align to set a reference, then Re-align on other frames.")

    #region UI
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        #region Video + Status
        left = QVBoxLayout()
        left.setSpacing(12)

        video_card = QFrame()
        video_card.setObjectName("videoCard")
        vc = QVBoxLayout(video_card)
        vc.setContentsMargins(6, 6, 6, 6)
        self.view = VideoView()
        self.view.clicked.connect(self.on_click)
        self.view.undo_requested.connect(self.undo_click)
        vc.addWidget(self.view)
        left.addWidget(video_card, 1)

        transport = QFrame()
        transport.setObjectName("card")
        tl = QHBoxLayout(transport)
        tl.setContentsMargins(14, 10, 14, 10)
        tl.setSpacing(10)

        self.play_btn = make_button("▶  Play", "accent", tooltip="Play / pause (Space)")
        self.play_btn.setMinimumWidth(96)
        self.play_btn.clicked.connect(self.toggle_play)
        self.prev_btn = make_button("‹", "icon", tooltip="Previous frame (←)")
        self.next_btn = make_button("›", "icon", tooltip="Next frame (→)")
        self.prev_btn.clicked.connect(lambda: self.step(-1))
        self.next_btn.clicked.connect(lambda: self.step(1))
        restart_btn = make_button("↺", "icon", tooltip="Restart and clear")
        restart_btn.clicked.connect(self.restart)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(0, self.total - 1))
        self.slider.valueChanged.connect(self.on_slider)
        self.time_label = make_label("", "mono")
        self.time_label.setMinimumWidth(170)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        for w in (self.play_btn, self.prev_btn, self.next_btn, restart_btn):
            tl.addWidget(w)
        tl.addWidget(self.slider, 1)
        tl.addWidget(self.time_label)
        left.addWidget(transport)

        self.status = QLabel()
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        left.addWidget(self.status)

        outer.addLayout(left, 1)

        #region Sidebar
        side = QFrame()
        side.setObjectName("card")
        side.setFixedWidth(290)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(18, 18, 18, 18)
        sl.setSpacing(10)

        sl.addWidget(make_label("AUTO ALIGN", "section"))
        self.auto_btn = make_button("Auto align this frame", "accent",
                                    tooltip="Fit the model to the box automatically, no clicks needed")
        self.auto_btn.clicked.connect(self.auto_align)
        sl.addWidget(self.auto_btn)
        auto_hint = make_label("Fits the model to the box's outline and edges. "
                               "Takes about a second.", "muted")
        auto_hint.setWordWrap(True)
        sl.addWidget(auto_hint)

        sl.addSpacing(6)
        div0 = QFrame()
        div0.setObjectName("divider")
        sl.addWidget(div0)
        sl.addSpacing(6)

        sl.addWidget(make_label("MANUAL ALIGN", "section"))
        self.manual_btn = make_button("Pick corners", checkable=True,
                                      tooltip="Click the box corners on the paused frame")
        self.manual_btn.clicked.connect(self.toggle_click)
        sl.addWidget(self.manual_btn)

        row = QHBoxLayout()
        row.addWidget(make_label("Next corner index"))
        row.addStretch(1)
        self.corner_spin = QSpinBox()
        self.corner_spin.setRange(0, 7)
        self.corner_spin.setToolTip(
            "Model corner the next click maps to.\n"
            "0-3 = one face, 4-7 = the opposite face\n"
            "(4 sits over 0, 5 over 1, 6 over 2, 7 over 3)."
        )
        row.addWidget(self.corner_spin)
        sl.addLayout(row)

        hint = make_label("0–3 = one face, 4–7 = opposite face. Index auto-advances; "
                          "right-click undoes the last point.", "muted")
        hint.setWordWrap(True)
        sl.addWidget(hint)

        self.solve_btn = make_button("Solve clicks")
        self.solve_btn.clicked.connect(self.solve_clicks)
        sl.addWidget(self.solve_btn)

        sl.addSpacing(6)
        div = QFrame()
        div.setObjectName("divider")
        sl.addWidget(div)
        sl.addSpacing(6)

        sl.addWidget(make_label("RE-ALIGN", "section"))
        self.realign_btn = make_button("Re-align this frame")
        self.realign_btn.clicked.connect(lambda: self.realign())
        sl.addWidget(self.realign_btn)

        self.auto_chk = QCheckBox("Auto re-align while playing")
        sl.addWidget(self.auto_chk)
        row = QHBoxLayout()
        row.addWidget(make_label("Every N frames", "muted"))
        row.addStretch(1)
        self.auto_every = QSpinBox()
        self.auto_every.setRange(1, 60)
        self.auto_every.setValue(DEFAULT_AUTO_EVERY)
        row.addWidget(self.auto_every)
        sl.addLayout(row)

        sl.addSpacing(6)
        div2 = QFrame()
        div2.setObjectName("divider")
        sl.addWidget(div2)
        sl.addSpacing(6)

        sl.addWidget(make_label("OVERLAY", "section"))
        self.faces_chk = QCheckBox("Fill faces")
        self.faces_chk.setChecked(True)
        self.faces_chk.toggled.connect(self.update_display)
        sl.addWidget(self.faces_chk)

        sl.addSpacing(6)
        div3 = QFrame()
        div3.setObjectName("divider")
        sl.addWidget(div3)
        sl.addSpacing(6)

        sl.addWidget(make_label("STATE", "section"))
        self.stat_ref = self._stat_row(sl, "Reference")
        self.stat_clicks = self._stat_row(sl, "Clicked points")
        self.stat_inl = self._stat_row(sl, "Last inliers")

        sl.addStretch(1)
        clear_btn = make_button("Clear everything", "ghost")
        clear_btn.clicked.connect(self.clear)
        sl.addWidget(clear_btn)

        outer.addWidget(side)
        self._refresh_stats()

    def _stat_row(self, layout, name):
        row = QHBoxLayout()
        row.addWidget(make_label(name, "muted"))
        row.addStretch(1)
        val = make_label("—", "value")
        row.addWidget(val)
        layout.addLayout(row)
        return val

    def _bind_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.step(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.step(1))

    def notify(self, text, level=None):
        self.status.setText(text)
        self.status.setProperty("level", level or "")
        repolish(self.status)

    def _refresh_stats(self):
        self.stat_ref.setText("set" if self.ref_des is not None else "not set")
        self.stat_clicks.setText(str(len(self.clicks)))
        self.stat_inl.setText(str(self.last_inliers) if self.last_inliers is not None else "—")

    #region Playback
    def _fmt_time(self, frame):
        s = frame / self.fps
        return f"{int(s // 60):02d}:{s % 60:04.1f}"

    def _show_frame(self, fr, i):
        self.frame = fr
        self.idx = i
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self.time_label.setText(
            f"{self._fmt_time(i)} / {self._fmt_time(self.total)}  ·  {i + 1}/{self.total}")
        self.update_display()

    def load_frame(self, i):
        i = max(0, min(i, self.total - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = self.cap.read()
        if ok:
            self._show_frame(fr, i)
        return ok

    def on_slider(self, val):
        if self.timer.isActive():
            self.stop_play()
        if val != self.idx:
            self.load_frame(val)

    def step(self, delta):
        self.stop_play()
        self.load_frame(self.idx + delta)

    def stop_play(self):
        self.timer.stop()
        self.play_btn.setText("Play")
        self.play_btn.setProperty("variant", "accent")
        repolish(self.play_btn)

    def toggle_play(self):
        if self.timer.isActive():
            self.stop_play()
            return
        if self.idx >= self.total - 1:
            self.load_frame(0)
        self.frame_counter = 0
        self.play_btn.setText("Pause")
        self.play_btn.setProperty("variant", "danger")
        repolish(self.play_btn)
        self.timer.start()

    def play_step(self):
        ok, fr = self.cap.read() # sequential read
        if not ok:
            self.stop_play()
            return
        self._show_frame(fr, self.idx + 1)
        self.frame_counter += 1
        if (self.auto_chk.isChecked() and self.ref_des is not None
                and self.frame_counter % self.auto_every.value() == 0):
            self.realign(silent=True)

    def restart(self):
        self.stop_play()
        self.clear()
        self.load_frame(0)

    def clear(self):
        self.show_overlay = False
        self.rvec = self.tvec = None
        self.clicks.clear()
        self.click_mode = False
        self.manual_btn.setChecked(False)
        self.view.set_pick_mode(False)
        self.ref_des = None
        self.ref_pts3d = None
        self.last_inliers = None
        self.frame_counter = 0
        self.corner_spin.setValue(0)
        self.update_display()
        self.notify("Cleared.")

    #region Display Update
    def update_display(self):
        if self.frame is None:
            return
        disp = self.frame.copy()

        if self.show_overlay and self.rvec is not None:
            proj, _ = cv2.projectPoints(self.corners_3d, self.rvec, self.tvec, self.K, DIST_COEFFS)
            proj = proj.reshape(-1, 2).astype(int)
            if self.faces_chk.isChecked():
                layer = disp.copy()
                for f in self.FACES:
                    cv2.fillConvexPoly(layer, proj[list(f)], (182, 196, 46))
                disp = cv2.addWeighted(layer, 0.28, disp, 0.72, 0)
            for a, b in self.EDGES:
                cv2.line(disp, tuple(proj[a]), tuple(proj[b]), (182, 196, 46), 2, cv2.LINE_AA)
            for p in proj:
                cv2.circle(disp, tuple(p), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(disp, tuple(p), 5, (182, 196, 46), 1, cv2.LINE_AA)

        for (x, y, cidx) in self.clicks:
            cv2.circle(disp, (int(x), int(y)), 6, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(disp, str(cidx), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

        rgb = np.ascontiguousarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.view.set_image(qimg)
        self._refresh_stats()

    #region Manual Alignment
    def toggle_click(self):
        self.click_mode = self.manual_btn.isChecked()
        self.clicks.clear()
        self.corner_spin.setValue(0)
        self.view.set_pick_mode(self.click_mode)
        if self.click_mode:
            self.stop_play()
            self.notify("Manual Align on. Click a corner of the real box; the index on the right "
                        "is the model corner it maps to. Need at least 4 points.")
        else:
            self.notify("Manual Align off.")
        self.update_display()

    def on_click(self, x, y):
        if not self.click_mode or self.frame is None:
            return
        cidx = self.corner_spin.value()
        self.clicks.append((x, y, cidx))
        self.corner_spin.setValue((cidx + 1) % 8)
        self.notify(f"Added point → corner {cidx}. Total: {len(self.clicks)}. "
                    "Click Solve clicks when ready.")
        self.update_display()

    def undo_click(self):
        if self.click_mode and self.clicks:
            _, _, cidx = self.clicks.pop()
            self.corner_spin.setValue(cidx)
            self.notify(f"Removed last point. Total: {len(self.clicks)}.")
            self.update_display()

    def solve_clicks(self):
        if len(self.clicks) < 4:
            self.notify("Click at least 4 points first.", "warn")
            return
        pts2d = np.float32([(c[0], c[1]) for c in self.clicks])
        pts3d = self.corners_3d[[c[2] for c in self.clicks]]
        ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, self.K, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            self.notify("solvePnP failed. Check your corner indices.", "error")
            return
        rvec, tvec = cv2.solvePnPRefineLM(pts3d, pts2d, self.K, DIST_COEFFS, rvec, tvec)
        self.rvec, self.tvec = rvec, tvec
        self.show_overlay = True
        self.store_reference()
        self.click_mode = False
        self.manual_btn.setChecked(False)
        self.view.set_pick_mode(False)
        self.clicks.clear()
        self.update_display()
        if self.ref_des is None:
            self.notify("Pose set, but too few SIFT features inside the box to store a reference.", "warn")
        else:
            self.notify("Reference pose set. You can now Re-align on other frames.", "ok")

    #region Geometry
    def ray_box_intersect(self, origin, direction):
        inv = 1.0 / (direction + 1e-8)
        t1 = (self.box_min - origin) * inv
        t2 = (self.box_max - origin) * inv
        t_enter = np.max(np.minimum(t1, t2))
        t_exit = np.min(np.maximum(t1, t2))
        if t_enter > t_exit or t_exit < 0:
            return None
        t = t_enter if t_enter > 0 else t_exit
        if t < 0:
            return None
        return origin + t * direction

    def backproject_to_box(self, pts2d, rvec, tvec):
        R, _ = cv2.Rodrigues(rvec)
        cam_center = -R.T @ tvec.flatten()
        out = []
        for pt in pts2d:
            x = (pt[0] - self.K[0, 2]) / self.K[0, 0]
            y = (pt[1] - self.K[1, 2]) / self.K[1, 1]
            ray = np.array([x, y, 1.0])
            ray /= np.linalg.norm(ray)
            out.append(self.ray_box_intersect(cam_center, R.T @ ray))
        return out

    def get_box_mask(self, frame):
        """Box mask from colour seed + GrabCut (the old HSV range also caught the table)."""
        mask = segment_box(frame)
        if mask is None:
            return np.zeros(frame.shape[:2], np.uint8), None
        return mask, mask

    #region Automatic Alignment
    def auto_align(self):
        if self.frame is None:
            return
        self.stop_play()
        self.notify("Fitting the model to the box…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            mask, _ = self.get_box_mask(self.frame)
            result = self.fitter.fit(self.frame, mask) if mask.any() else None
        finally:
            QApplication.restoreOverrideCursor()
        if result is None:
            self.notify("Could not find the box in this frame.", "error")
            return
        self.rvec, self.tvec = result["rvec"], result["tvec"]
        self.show_overlay = True
        self.click_mode = False
        self.manual_btn.setChecked(False)
        self.view.set_pick_mode(False)
        self.clicks.clear()
        self.store_reference()
        self.update_display()
        quality = f"outline overlap {result['iou']:.0%}, edge match {result['edge']:.0%}"
        if result["iou"] < 0.85:
            self.notify(f"Weak fit ({quality}). Box may be partly out of frame; try Manual Align.", "warn")
        else:
            self.notify(f"Auto aligned: {quality}. Re-align now works on other frames.", "ok")

    #region SIFT Re-alignment
    def detect(self, frame):
        """SIFT keypoints + RootSIFT descriptors inside the box mask."""
        mask, _ = self.get_box_mask(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, mask)
        if des is not None:
            des = des / (des.sum(axis=1, keepdims=True) + 1e-7)  # L1 normalise
            des = np.sqrt(des).astype(np.float32)                # RootSIFT
        return kp, des

    def store_reference(self):
        if self.frame is None or self.rvec is None:
            return
        kp, des = self.detect(self.frame)
        if des is None or len(kp) < 8:
            return
        pts2d = np.float32([k.pt for k in kp])
        pts3d_list = self.backproject_to_box(pts2d, self.rvec, self.tvec)
        keep = [i for i, p in enumerate(pts3d_list) if p is not None]
        if len(keep) < 8:
            return
        self.ref_des = des[keep]
        self.ref_pts3d = np.float32([pts3d_list[i] for i in keep])

    def realign(self, silent=False):
        if self.ref_des is None or self.frame is None or self.ref_pts3d is None:
            if not silent:
                self.notify("No reference yet. Set one with Manual Align first.", "warn")
            return
        if self.timer.isActive() and not silent:
            self.stop_play()

        kp, des = self.detect(self.frame)
        if des is None or len(kp) < 6:
            if not silent:
                self.notify("Not enough SIFT features in this frame.", "warn")
            return

        good = []
        for pair in self.bf.knnMatch(des, self.ref_des, k=2):
            if len(pair) == 2 and pair[0].distance < RATIO * pair[1].distance:
                good.append(pair[0])
        if len(good) < 6:
            if not silent:
                self.notify(f"Only {len(good)} good matches. Try a frame closer to the reference.", "warn")
            return

        pts2d = np.float32([kp[m.queryIdx].pt for m in good])
        pts3d = self.ref_pts3d[[m.trainIdx for m in good]]
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, self.K, DIST_COEFFS,
            iterationsCount=100, reprojectionError=8.0, confidence=0.99,
        )
        if not success or inliers is None or len(inliers) < 4:
            if not silent:
                self.notify("solvePnPRansac could not find a consistent pose.", "error")
            return

        sel = inliers.flatten()
        rvec, tvec = cv2.solvePnPRefineLM(pts3d[sel], pts2d[sel], self.K, DIST_COEFFS, rvec, tvec)
        self.rvec, self.tvec = rvec, tvec
        self.show_overlay = True
        self.last_inliers = len(inliers)
        self.update_display()
        if silent:
            self.notify(f"Auto re-aligned (frame {self.idx}, {len(inliers)} inliers).", "ok")
        else:
            self.notify(f"Re-aligned using {len(inliers)} inlier matches.", "ok")

    def closeEvent(self, e):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(e)

def main():
    video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    obj = sys.argv[2] if len(sys.argv) > 2 else OBJ_PATH
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    try:
        win = MainWindow(video, obj)
    except Exception as exc:  # missing files, unreadable video, bad OBJ
        QMessageBox.critical(None, "Could not start", str(exc))
        return 1
    win.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())