import math
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import os

class Context:
    def __init__(self, name=""):
        self.name = name
        self.c0 = 1          
        self.c1 = 1          
        self.uses = 0
        self.bits = 0.0      

    def encode(self, bit: int):
        """Encode one bin, update model, accumulate ideal rate."""
        total = self.c0 + self.c1
        p = (self.c1 if bit == 1 else self.c0) / total
        self.bits += -math.log2(max(p, 1e-10))
        self.uses += 1
        if bit == 1:
            self.c1 += 1
        else:
            self.c0 += 1

        # skalerer ned counts regelmessig for å beholde adaptivitet 
        if total > 256:
            self.c0 = max(1, self.c0 // 2)
            self.c1 = max(1, self.c1 // 2)

    def prob1(self):
        return self.c1 / (self.c0 + self.c1)

    def __repr__(self):
        return (f"Context({self.name}): uses={self.uses}, "
                f"P(1)={self.prob1():.3f}, bits={self.bits:.1f}")

class SimpleCABAC:
    def __init__(self):
        self.contexts = [
            Context("sig_prev0"), # forrige residual var 0
            Context("sig_prev_small"), # forrige residual var 1-3
            Context("sig_prev_med"), # forrige residual var 4-10
            Context("sig_prev_large"), # forrige residual var over 10
            Context("unary_first"), # angir om størrelsen på residual var 1 eller mer
            Context("unary_rest"), # koder størrelsen på residual i unary (opptil 15)
        ]
        self.total_bits = 0.0
        self.total_residuals = 0

    def residual_class(self, level: int) -> int:
        """Map residual magnitude to a class for context selection."""
        if level == 0:
            return 0
        if level <= 3:
            return 1
        if level <= 10:
            return 2
        return 3

    def binarize_and_encode(self, level: int, prev_class: int):
        self.total_residuals += 1

        # 1. Significance flag
        sig = 1 if level > 0 else 0
        ctx_idx = min(prev_class, 3)
        self.contexts[ctx_idx].encode(sig)

        if sig == 0:
            return

        # 2. Unary for (level - 1), max 15
        mag = min(level - 1, 15)
        for i in range(mag):
            ctx = 4 if i == 0 else 5
            self.contexts[ctx].encode(1)
        if mag < 15:
            ctx = 4 if mag == 0 else 5
            self.contexts[ctx].encode(0)

    def get_report(self, residuals):
        lines = []
        lines.append("=== Encoding statistics ===")
        lines.append(f"Residuals processed : {self.total_residuals}")
        total = sum(c.bits for c in self.contexts)
        lines.append(f"Total ideal bits    : {total:.1f}")
        if self.total_residuals:
            lines.append(f"Bits / residual     : {total / self.total_residuals:.3f}")
        lines.append("\nContext usage:")
        for c in self.contexts:
            if c.uses > 0:
                lines.append(f"  {c}")
        # Most significant contexts
        used = [(c.uses, c.name, c.prob1(), c.bits) for c in self.contexts if c.uses > 0]
        used.sort(reverse=True)
        lines.append("\nMost used contexts:")
        for uses, name, p, bits in used[:4]:
            lines.append(f"  {name}: {uses} uses, final P(1)={p:.3f}, bits={bits:.1f}")
        # zero residual ratio
        if residuals:
            zeros = sum(1 for r in residuals if r == 0)
            lines.append(f"\nZero residual ratio: {zeros / len(residuals):.3f}")
        return "\n".join(lines)

def extract_residuals(video_path: str, max_frames: int = 30, quant: int = 8, step: int = 4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    residuals = []
    prev = None
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            # residual i tidsdomene
            diff = np.abs(gray.astype(np.int16) - prev.astype(np.int16))
            # kvantisering
            qdiff = (diff // quant).astype(np.int32)
            # subsampling
            levels = qdiff[::step, ::step].flatten()
            residuals.extend(levels.tolist())
        prev = gray
        frame_idx += 1

    cap.release()
    return residuals, frame_idx

class CABACGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Simplified CABAC Residual Encoder")
        self.root.geometry("900x700")
        self.root.minsize(700, 500)

        self.video_path = None
        self.processing = False

        # Controls
        ctrl = ttk.Frame(root, padding=10)
        ctrl.pack(fill=tk.X)

        ttk.Button(ctrl, text="Select Video", command=self.select_video).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(ctrl, text="No file selected", width=50, anchor="w")
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.run_btn = ttk.Button(ctrl, text="Run Encoding", command=self.run_encoding, state=tk.DISABLED)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        # Parameters 
        param = ttk.LabelFrame(root, text="Parameters", padding=5)
        param.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(param, text="Max frames:").grid(row=0, column=0, sticky="e", padx=5)
        self.max_frames_var = tk.IntVar(value=30)
        ttk.Spinbox(param, from_=1, to=1000, textvariable=self.max_frames_var, width=8).grid(row=0, column=1, padx=5)

        ttk.Label(param, text="Quant step:").grid(row=0, column=2, sticky="e", padx=5)
        self.quant_var = tk.IntVar(value=8)
        ttk.Spinbox(param, from_=1, to=64, textvariable=self.quant_var, width=8).grid(row=0, column=3, padx=5)

        ttk.Label(param, text="Spatial step:").grid(row=0, column=4, sticky="e", padx=5)
        self.step_var = tk.IntVar(value=4)
        ttk.Spinbox(param, from_=1, to=16, textvariable=self.step_var, width=8).grid(row=0, column=5, padx=5)

        # Output
        text_frame = ttk.LabelFrame(root, text="Report", padding=5)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.report_text = scrolledtext.ScrolledText(text_frame, height=12, wrap=tk.WORD, font=("Courier", 10))
        self.report_text.pack(fill=tk.BOTH, expand=True)

        # Plot
        plot_frame = ttk.LabelFrame(root, text="Context Usage Visualization", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(8, 3.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title("Run encoding to see context usage")
        self.fig.tight_layout()
        self.canvas.draw()

    def select_video(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.video_path = path
            self.file_label.config(text=os.path.basename(path))
            self.run_btn.config(state=tk.NORMAL)
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, f"Selected: {path}\nReady to encode.")

    def run_encoding(self):
        if not self.video_path or self.processing:
            return
        self.processing = True
        self.run_btn.config(state=tk.DISABLED)
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, "Processing video... please wait.\n")
        self.root.update_idletasks()

        # kjør i egen tråd
        thread = threading.Thread(target=self._process_thread, daemon=True)
        thread.start()

    def _process_thread(self):
        try:
            max_frames = self.max_frames_var.get()
            quant = self.quant_var.get()
            step = self.step_var.get()

            residuals, n_frames = extract_residuals(
                self.video_path, max_frames=max_frames, quant=quant, step=step
            )

            if not residuals:
                self.root.after(0, lambda: self._show_error("No residuals extracted (video too short or empty)."))
                return

            encoder = SimpleCABAC()
            prev_class = 0
            for level in residuals:
                encoder.binarize_and_encode(int(level), prev_class)
                prev_class = encoder.residual_class(int(level))

            report = encoder.get_report(residuals)
            header = (f"Extracted {len(residuals)} residual levels from {n_frames} frames "
                      f"(quant={quant}, step={step})\n\n")
            full_report = header + report

            # oppdater UI på main thread
            self.root.after(0, lambda: self._update_results(full_report, encoder))

        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
        finally:
            self.processing = False
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))

    def _show_error(self, msg):
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, f"Error: {msg}")
        messagebox.showerror("Error", msg)

    def _update_results(self, report, encoder):
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report)
        self._plot_contexts(encoder)

    def _plot_contexts(self, encoder):
        self.ax.clear()

        used = [c for c in encoder.contexts if c.uses > 0]
        if not used:
            self.ax.set_title("No contexts used")
            self.canvas.draw()
            return

        names = [c.name for c in used]
        bits = [c.bits for c in used]
        uses = [c.uses for c in used]
        probs = [c.prob1() for c in used]

        # fargekoding basert på symbol-probabilitet
        colors = plt.cm.coolwarm(probs)

        bars = self.ax.bar(names, bits, color=colors, edgecolor="black", linewidth=0.5)
        self.ax.set_ylabel("Ideal bits")
        self.ax.set_title("Context Usage — bits colored by final P(1)")
        self.ax.tick_params(axis="x", rotation=25, labelsize=9)

        for bar, u, p in zip(bars, uses, probs):
            h = bar.get_height()
            self.ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + max(bits) * 0.02,
                f"{u} uses\nP(1)={p:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                linespacing=1.1
            )

        self.ax.set_ylim(0, max(bits) * 1.25 if bits else 1)
        self.fig.tight_layout()
        self.canvas.draw()

def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = CABACGui(root)
    root.mainloop()

if __name__ == "__main__":
    main()
