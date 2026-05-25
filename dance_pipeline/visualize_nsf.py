"""
NSF skeleton visualizer.

Two modes:
  matplotlib  — animated window, no browser needed
  plotly      — interactive HTML, open in browser, drag timeline slider

Usage:
    # After extracting NSF:
    from dance_pipeline.visualize_nsf import visualize
    visualize(seq, backend="matplotlib")   # or "plotly"

    # CLI:
    python -m dance_pipeline.visualize_nsf --file motions/my_dance.nsf.npz
    python -m dance_pipeline.visualize_nsf --file motions/my_dance.nsf.npz --backend plotly
"""

import sys
import argparse
from pathlib import Path
import numpy as np

from .nsf.format import NSFSequence, Joint, PARENT

# Bone connections: (parent_joint, child_joint)
BONES: list[tuple[Joint, Joint]] = [
    # Spine
    (Joint.ROOT,   Joint.SPINE),
    (Joint.SPINE,  Joint.CHEST),
    (Joint.CHEST,  Joint.NECK),
    (Joint.NECK,   Joint.HEAD),
    # Left leg
    (Joint.ROOT,    Joint.L_HIP),
    (Joint.L_HIP,   Joint.L_KNEE),
    (Joint.L_KNEE,  Joint.L_ANKLE),
    (Joint.L_ANKLE, Joint.L_FOOT),
    (Joint.L_FOOT,  Joint.L_TOE),
    # Right leg
    (Joint.ROOT,    Joint.R_HIP),
    (Joint.R_HIP,   Joint.R_KNEE),
    (Joint.R_KNEE,  Joint.R_ANKLE),
    (Joint.R_ANKLE, Joint.R_FOOT),
    (Joint.R_FOOT,  Joint.R_TOE),
    # Left arm
    (Joint.CHEST,      Joint.L_SHOULDER),
    (Joint.L_SHOULDER, Joint.L_ELBOW),
    (Joint.L_ELBOW,    Joint.L_WRIST),
    (Joint.L_WRIST,    Joint.L_HAND),
    # Right arm
    (Joint.CHEST,      Joint.R_SHOULDER),
    (Joint.R_SHOULDER, Joint.R_ELBOW),
    (Joint.R_ELBOW,    Joint.R_WRIST),
    (Joint.R_WRIST,    Joint.R_HAND),
]

# Color by body region
_BONE_COLORS = {
    "spine": "#FFD700",
    "left":  "#4FC3F7",
    "right": "#EF9A9A",
}

def _bone_color(parent: Joint, child: Joint) -> str:
    name = child.name.lower()
    if name.startswith("l_"):
        return _BONE_COLORS["left"]
    if name.startswith("r_"):
        return _BONE_COLORS["right"]
    return _BONE_COLORS["spine"]


# ── matplotlib backend ────────────────────────────────────────────────────────

def _visualize_matplotlib(seq: NSFSequence, step: int = 1) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pos = seq.positions  # (T, J, 3)
    # x=right, y=up, z=forward
    xs, ys, zs = pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False

    # Fixed axis limits based on full sequence
    pad = 0.2
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_zlim(zs.min() - pad, zs.max() + pad)
    ax.set_xlabel("X (right)", color="white")
    ax.set_ylabel("Y (up)",    color="white")
    ax.set_zlabel("Z (fwd)",   color="white")
    ax.tick_params(colors="white")

    joint_dots, = ax.plot([], [], [], "o", color="white", ms=4, zorder=5)
    bone_lines  = [
        ax.plot([], [], [], "-", color=_bone_color(p, c), lw=2.5)[0]
        for p, c in BONES
    ]
    frame_text = ax.text2D(
        0.02, 0.95, "", transform=ax.transAxes, color="white", fontsize=10
    )

    frames = range(0, seq.num_frames, step)

    def update(t):
        p = pos[t]
        joint_dots.set_data(p[:, 0], p[:, 1])
        joint_dots.set_3d_properties(p[:, 2])
        for line, (pj, cj) in zip(bone_lines, BONES):
            pp, pc = p[int(pj)], p[int(cj)]
            line.set_data([pp[0], pc[0]], [pp[1], pc[1]])
            line.set_3d_properties([pp[2], pc[2]])
        frame_text.set_text(
            f"frame {t}/{seq.num_frames}  t={t/seq.fps:.2f}s  src={seq.source}"
        )
        return [joint_dots, frame_text] + bone_lines

    ani = animation.FuncAnimation(
        fig, update, frames=frames,
        interval=1000 / seq.fps * step,
        blit=False,
    )
    plt.tight_layout()
    plt.show()
    return ani  # keep reference to avoid GC


# ── plotly backend ────────────────────────────────────────────────────────────

def _visualize_plotly(seq: NSFSequence, step: int = 1, output_html: str | None = None) -> None:
    import plotly.graph_objects as go

    pos = seq.positions  # (T, J, 3)
    frames_idx = range(0, seq.num_frames, step)

    def make_traces(t: int):
        p = pos[t]
        traces = []
        # Bones
        for pj, cj in BONES:
            pp, pc = p[int(pj)], p[int(cj)]
            color = _bone_color(pj, cj)
            traces.append(go.Scatter3d(
                x=[pp[0], pc[0]], y=[pp[1], pc[1]], z=[pp[2], pc[2]],
                mode="lines",
                line=dict(color=color, width=5),
                showlegend=False,
                hoverinfo="skip",
            ))
        # Joints
        traces.append(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2],
            mode="markers",
            marker=dict(size=4, color="white"),
            text=[j.name for j in Joint],
            hovertemplate="%{text}<br>x=%{x:.3f} y=%{y:.3f} z=%{z:.3f}<extra></extra>",
            showlegend=False,
        ))
        return traces

    # Initial frame
    init_traces = make_traces(0)

    # Slider frames
    plotly_frames = []
    for t in frames_idx:
        plotly_frames.append(go.Frame(
            data=make_traces(t),
            name=str(t),
        ))

    sliders = [{
        "steps": [
            {"args": [[str(t)], {"frame": {"duration": 0}, "mode": "immediate"}],
             "label": f"{t/seq.fps:.1f}s",
             "method": "animate"}
            for t in frames_idx
        ],
        "transition": {"duration": 0},
        "x": 0.0, "len": 1.0, "y": -0.02,
        "currentvalue": {"prefix": "time: ", "visible": True},
    }]

    pad = 0.3
    fig = go.Figure(
        data=init_traces,
        frames=plotly_frames,
        layout=go.Layout(
            title=f"NSF Skeleton: {seq.name}  ({seq.num_frames} frames @ {seq.fps:.0f}fps)",
            scene=dict(
                xaxis=dict(title="X (right)", range=[pos[:,:,0].min()-pad, pos[:,:,0].max()+pad]),
                yaxis=dict(title="Y (up)",    range=[pos[:,:,1].min()-pad, pos[:,:,1].max()+pad]),
                zaxis=dict(title="Z (fwd)",   range=[pos[:,:,2].min()-pad, pos[:,:,2].max()+pad]),
                aspectmode="cube",
                bgcolor="#1a1a2e",
            ),
            paper_bgcolor="#1a1a2e",
            font_color="white",
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {"label": "▶ Play",
                     "method": "animate",
                     "args": [None, {"frame": {"duration": int(1000/seq.fps*step)},
                                     "fromcurrent": True}]},
                    {"label": "⏸ Pause",
                     "method": "animate",
                     "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
                ],
                "x": 0.05, "y": 0.05,
            }],
            sliders=sliders,
            height=700,
        ),
    )

    if output_html:
        fig.write_html(output_html)
        print(f"Saved: {output_html}")
    else:
        fig.show()


# ── public API ────────────────────────────────────────────────────────────────

def visualize(
    seq: NSFSequence,
    backend: str = "matplotlib",
    step: int = 1,
    output_html: str | None = None,
) -> None:
    """
    Args:
        seq:         NSFSequence to visualize.
        backend:     "matplotlib" (animated window) or "plotly" (browser).
        step:        Render every Nth frame (use >1 to speed up large sequences).
        output_html: If set (plotly only), save to this HTML path instead of opening browser.
    """
    if backend == "matplotlib":
        _visualize_matplotlib(seq, step=step)
    elif backend == "plotly":
        _visualize_plotly(seq, step=step, output_html=output_html)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'matplotlib' or 'plotly'.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize an NSF sequence")
    parser.add_argument("--file",    required=True, help="Path to .nsf.npz file")
    parser.add_argument("--backend", default="matplotlib", choices=["matplotlib", "plotly"])
    parser.add_argument("--step",    type=int, default=1, help="Render every N frames")
    parser.add_argument("--html",    default=None, help="(plotly) save to HTML instead of browser")
    args = parser.parse_args()

    from .nsf.io import load_nsf
    seq = load_nsf(args.file)
    print(f"Loaded: {seq.num_frames} frames @ {seq.fps} fps  src={seq.source}  name={seq.name}")
    visualize(seq, backend=args.backend, step=args.step, output_html=args.html)


if __name__ == "__main__":
    main()
