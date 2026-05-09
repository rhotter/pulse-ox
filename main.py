# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "mss>=9.0.1",
#     "opencv-python>=4.10.0",
#     "numpy>=1.26",
#     "matplotlib>=3.8",
# ]
# ///
"""Capture a screen region containing a scrolling line plot, detect the
current value of the trace, and display it live."""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from dataclasses import dataclass

import cv2
import mss
import numpy as np


@dataclass
class Region:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


def select_region() -> Region | None:
    """Show a translucent borderless overlay and let the user drag a rect.

    Avoids tk's `-fullscreen` on macOS, which animates the window into a
    separate Space and renders a black backdrop behind the alpha overlay.
    """
    root = tk.Tk()
    root.withdraw()
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    root.overrideredirect(True)
    root.geometry(f"{sw}x{sh}+0+0")
    root.configure(bg="#000000")
    root.attributes("-alpha", 0.3)
    root.attributes("-topmost", True)
    root.deiconify()
    root.focus_force()

    canvas = tk.Canvas(
        root, cursor="cross", bg="#000000", highlightthickness=0
    )
    canvas.pack(fill=tk.BOTH, expand=True)
    canvas.create_text(
        sw // 2,
        24,
        text="Drag to select region — Esc to cancel",
        fill="#ffffff",
        font=("Helvetica", 14, "bold"),
    )

    state = {"x0": 0, "y0": 0, "rect": None, "result": None}

    def on_press(e):
        state["x0"], state["y0"] = e.x, e.y
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            e.x, e.y, e.x, e.y, outline="#33aaff", width=2
        )

    def on_drag(e):
        if state["rect"] is None:
            return
        canvas.coords(state["rect"], state["x0"], state["y0"], e.x, e.y)

    def on_release(e):
        x0, y0 = state["x0"], state["y0"]
        x1, y1 = e.x, e.y
        left, top = min(x0, x1), min(y0, y1)
        width, height = abs(x1 - x0), abs(y1 - y0)
        if width < 5 or height < 5:
            state["result"] = None
        else:
            state["result"] = Region(left, top, width, height)
        root.destroy()

    def on_escape(_):
        state["result"] = None
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_escape)

    root.mainloop()
    return state["result"]


def find_cursor_by_motion(
    curr_bgr: np.ndarray,
    prev_bgr: np.ndarray,
    motion_thresh: int = 5,
    min_total_motion: int = 20,
    prev_x: int | None = None,
    sigma: float = 18.0,
) -> int | None:
    """Return the column with peak motion energy (the sweep cursor's x).

    If prev_x is supplied, the per-column motion energy is multiplied by
    a circular Gaussian centered at prev_x. Distance wraps around the
    right edge of the frame back to the left edge, so a cursor that just
    wrapped is still considered "close" to one near the right edge.
    """
    diff = cv2.absdiff(curr_bgr, prev_bgr).max(axis=2)
    col_energy = (diff > motion_thresh).sum(axis=0).astype(np.float32)
    if int(col_energy.sum()) < min_total_motion:
        return None
    k = np.ones(7, dtype=np.float32) / 7.0
    smoothed = np.convolve(col_energy, k, mode="same")
    if smoothed.max() <= 0:
        return None
    if prev_x is not None:
        W = smoothed.size
        cols = np.arange(W, dtype=np.float32)
        d = np.abs(cols - float(prev_x))
        d = np.minimum(d, W - d)  # circular distance
        weight = np.exp(-(d * d) / (2.0 * sigma * sigma))
        smoothed = smoothed * weight
        if smoothed.max() <= 0:
            return None
    return int(np.argmax(smoothed))


def find_trace_gap(
    img_bgr: np.ndarray,
    gap_min: int = 6,
) -> tuple[int, int] | None:
    """Return (lo, hi) of the largest empty column-stretch inside the trace.

    Returns None when the trace has no significant interior gap (e.g. the
    sweep just wrapped and the whole width is filled).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([95, 80, 120]), np.array([130, 255, 255])
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    col_has = mask.any(axis=0)
    cols = np.where(col_has)[0]
    if cols.size == 0:
        return None
    first, last = int(cols[0]), int(cols[-1])
    interior = ~col_has[first : last + 1]
    if not interior.any():
        return None
    edges = np.diff(np.concatenate([[0], interior.view(np.uint8), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    lengths = ends - starts
    k = int(np.argmax(lengths))
    if int(lengths[k]) < gap_min:
        return None
    return first + int(starts[k]), first + int(ends[k]) - 1


def sample_blue_y(
    img_bgr: np.ndarray,
    cursor_x: int,
    search_radius: int = 8,
) -> tuple[int | None, int | None, np.ndarray]:
    """At cursor_x (with a small ± search), find the blue trace y."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.array([95, 80, 120]), np.array([130, 255, 255])
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    W = mask.shape[1]
    cursor_x = int(np.clip(cursor_x, 0, W - 1))
    for dx in range(0, search_radius + 1):
        for sgn in ((-1, +1) if dx else (0,)):
            x = cursor_x + sgn * dx
            if x < 0 or x >= W:
                continue
            rows = np.where(mask[:, x] > 0)[0]
            if rows.size:
                return x, int(rows.mean()), mask
    return cursor_x, None, mask


def detect_trace_value(
    img_bgr: np.ndarray,
    cursor_x: int | None = None,
    gap_min: int = 6,
) -> tuple[int | None, int | None, np.ndarray]:
    """Find the blue trace and return (cursor_x_px, trace_y_px, mask).

    For a sweep-style plot, the newest sample is the right edge of the
    newer segment, with a blank gap between it and the older segment to
    its right. We find the largest empty column-gap of width >= gap_min
    and use the column just before it. If no significant gap exists (the
    trace fills the width — sweep just wrapped), we fall back to the
    rightmost column. If cursor_x is provided, it overrides the search.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([95, 80, 120])
    upper = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    col_has = mask.any(axis=0)
    cols_with_trace = np.where(col_has)[0]
    if cols_with_trace.size == 0:
        return None, None, mask

    if cursor_x is None:
        first, last = int(cols_with_trace[0]), int(cols_with_trace[-1])
        # Find the longest run of empty columns strictly inside [first, last].
        interior = ~col_has[first : last + 1]
        cursor_x = last
        if interior.any():
            # Run-length encode the interior empty stretches.
            edges = np.diff(np.concatenate([[0], interior.view(np.uint8), [0]]))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]
            lengths = ends - starts
            if lengths.size:
                k = int(np.argmax(lengths))
                if int(lengths[k]) >= gap_min:
                    cursor_x = first + int(starts[k]) - 1
    else:
        cursor_x = int(np.clip(cursor_x, 0, mask.shape[1] - 1))

    rows = np.where(mask[:, cursor_x] > 0)[0]
    if rows.size == 0:
        nearest = cols_with_trace[np.argmin(np.abs(cols_with_trace - cursor_x))]
        cursor_x = int(nearest)
        rows = np.where(mask[:, cursor_x] > 0)[0]
        if rows.size == 0:
            return cursor_x, None, mask

    trace_y = int(rows.mean())
    return cursor_x, trace_y, mask


def y_to_value(y_px: int, height: int, ymin: float, ymax: float) -> float:
    """Convert pixel y (top-down) to a calibrated value (top = ymax)."""
    if height <= 1:
        return ymin
    frac = 1.0 - (y_px / (height - 1))
    return ymin + frac * (ymax - ymin)


def render_overlay(
    frame: np.ndarray,
    cursor_x: int | None,
    trace_y: int | None,
    value: float | None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if cursor_x is not None:
        cv2.line(out, (cursor_x, 0), (cursor_x, h - 1), (0, 255, 255), 1)
    if cursor_x is not None and trace_y is not None:
        cv2.circle(out, (cursor_x, trace_y), 6, (0, 0, 255), 2)

    label = f"value: {value:.2f}" if value is not None else "value: --"
    cv2.rectangle(out, (0, 0), (220, 32), (0, 0, 0), -1)
    cv2.putText(
        out, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    return out


def run(region: Region, ymin: float, ymax: float, fps_cap: float) -> None:
    print(
        f"Capturing region left={region.left} top={region.top} "
        f"w={region.width} h={region.height}",
        file=sys.stderr,
    )
    print("Press q in the preview window (or Ctrl-C in terminal) to quit.", file=sys.stderr)

    win = "pulse-ox"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, region.width, region.height)

    period = 1.0 / fps_cap if fps_cap > 0 else 0.0

    readings: list[tuple[float, float]] = []
    t0 = time.time()

    with mss.mss() as sct:
        monitor = region.as_mss()
        last = 0.0
        prev_frame: np.ndarray | None = None
        prev_cursor_x: int | None = None
        try:
            while True:
                now = time.time()
                if period and (now - last) < period:
                    time.sleep(period - (now - last))
                last = time.time()

                shot = np.array(sct.grab(monitor))  # BGRA
                frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

                cursor_x: int | None = None
                if prev_frame is not None and prev_frame.shape == frame.shape:
                    cursor_x = find_cursor_by_motion(
                        frame, prev_frame, prev_x=prev_cursor_x
                    )
                    if cursor_x is None and prev_cursor_x is not None:
                        # No motion this frame — keep the last known cursor.
                        cursor_x = prev_cursor_x

                if cursor_x is not None:
                    cursor_x, trace_y, _mask = sample_blue_y(frame, cursor_x)
                else:
                    cursor_x, trace_y, _mask = detect_trace_value(frame)

                prev_cursor_x = cursor_x

                value = (
                    y_to_value(trace_y, region.height, ymin, ymax)
                    if trace_y is not None
                    else None
                )
                if value is not None:
                    readings.append((time.time() - t0, value))
                    print(f"{value:.3f}", flush=True)

                overlay = render_overlay(frame, cursor_x, trace_y, value)
                cv2.imshow(win, overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                prev_frame = frame
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()

    plot_readings(readings)


def plot_readings(readings: list[tuple[float, float]]) -> None:
    if len(readings) < 2:
        print("No readings collected — skipping plot.", file=sys.stderr)
        return
    import matplotlib.pyplot as plt

    ts = np.array([r[0] for r in readings])
    vs = np.array([r[1] for r in readings])

    out_path = f"reading_{int(time.time())}.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, vs, color="#1f77ff", linewidth=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("value")
    ax.set_title(f"pulse-ox reading — {len(readings)} samples over {ts[-1]:.1f}s")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}", file=sys.stderr)
    plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ymin", type=float, default=0.0, help="value at bottom of region")
    p.add_argument("--ymax", type=float, default=1.0, help="value at top of region")
    p.add_argument("--fps", type=float, default=30.0, help="capture rate cap (0 = uncapped)")
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="skip the picker, pass left,top,width,height",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.region:
        parts = [int(v) for v in args.region.split(",")]
        if len(parts) != 4:
            print("--region must be left,top,width,height", file=sys.stderr)
            return 2
        region = Region(*parts)
    else:
        region = select_region()
        if region is None:
            print("No region selected.", file=sys.stderr)
            return 1
    run(region, args.ymin, args.ymax, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
