"""
Thunder poster: Streamlit app.
Run with:  streamlit run streamlit_app.py
"""
import colorsys
import io

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------------------------------------------------------
# Tunable parameters (unchanged from the desktop version)
# --------------------------------------------------------------------------
CONFIG = dict(
    hue_range=(0.53, 0.68),      # HSV hue window: cyan-ish blue -> electric indigo
    layers=(5, 8),               # min / max number of layered bolts
    size_range=(0.3, 1.2),       # range for lightning size
    wobble_default=0.55,         # overall jaggedness
    subdivisions=7,              # midpoint-displacement depth (2**n segments)
    roughness_decay=0.55,        # how fast displacement shrinks per subdivision
    branch_count=(3, 6),         # side-branches per bolt
    branch_angle=(0.25, 0.9),    # radians away from the parent direction
    layer_spread=0.09,           # horizontal scatter of layers (poster widths)
    glow_passes=(9.0, 5.0, 2.6), # glow line widths (relative)
)

FIG_W, FIG_H = 6.0, 8.0
POSTER_W = FIG_W / FIG_H         # x-extent so aspect stays equal (no control strip)


# --------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------
def hsv(h, s, v):
    return colorsys.hsv_to_rgb(h % 1.0, np.clip(s, 0, 1), np.clip(v, 0, 1))


def make_palette(rng, mode, n):
    """Random electric-blue palette, ordered back (dark) -> front (bright)."""
    lo, hi = CONFIG["hue_range"]
    cols = []
    for _ in range(n):
        h = rng.uniform(lo, hi)
        if mode == "Pastel":
            s, v = rng.uniform(0.18, 0.42), rng.uniform(0.88, 1.0)
        else:
            s, v = rng.uniform(0.75, 1.0), rng.uniform(0.85, 1.0)
        cols.append((h, s, v))
    cols.sort(key=lambda c: c[2] - 0.2 * c[1])     # dimmer/more saturated first
    return [hsv(*c) for c in cols]


def background(rng, mode):
    h = rng.uniform(*CONFIG["hue_range"])
    if mode == "Pastel":
        return hsv(h, 0.30, 0.34), hsv(h, 0.22, 0.20)
    return hsv(h, 0.95, 0.24), hsv(h, 1.00, 0.04)


def lighten(rgb, amt):
    return tuple(np.array(rgb) * (1 - amt) + amt)


# --------------------------------------------------------------------------
# Lightning geometry
# --------------------------------------------------------------------------
def jagged_path(rng, p0, p1, wobble):
    """Midpoint displacement between two points -> (N, 2) array."""
    pts = np.array([p0, p1], dtype=float)
    length = np.linalg.norm(pts[1] - pts[0])
    disp = length * 0.30 * wobble
    for _ in range(CONFIG["subdivisions"]):
        a, b = pts[:-1], pts[1:]
        mid = (a + b) / 2
        d = b - a
        norm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        norm /= np.linalg.norm(norm, axis=1, keepdims=True) + 1e-12
        mid += norm * rng.uniform(-1, 1, (len(mid), 1)) * disp
        out = np.empty((2 * len(a) + 1, 2))
        out[0:-1:2], out[1::2], out[-1] = a, mid, pts[-1]
        pts = out
        disp *= CONFIG["roughness_decay"]
    return pts


def build_bolt(rng, x_top, reach, wobble):
    """Main trunk starting at the TOP edge (y = 1) plus side branches."""
    x_end = x_top + rng.uniform(-0.18, 0.18) * POSTER_W
    trunk = jagged_path(rng, (x_top, 1.02), (x_end, 1.0 - reach), wobble)
    paths = [(trunk, 1.0)]

    lo, hi = CONFIG["branch_count"]
    for _ in range(rng.integers(lo, hi + 1)):
        i = rng.integers(len(trunk) // 6, len(trunk) * 5 // 6)
        start = trunk[i]
        direction = trunk[min(i + 12, len(trunk) - 1)] - start
        base = np.arctan2(direction[1], direction[0])
        ang = base + rng.choice([-1, 1]) * rng.uniform(*CONFIG["branch_angle"])
        blen = reach * rng.uniform(0.12, 0.32)
        end = start + blen * np.array([np.cos(ang), np.sin(ang)])
        if end[1] > start[1]:                       # keep branches heading downward
            end[1] = start[1] - abs(end[1] - start[1])
        branch = jagged_path(rng, start, end, wobble * 1.1)
        paths.append((branch, 0.45))
    return paths


# --------------------------------------------------------------------------
# Poster: returns the figure instead of showing it
# --------------------------------------------------------------------------
def draw_poster(seed, mode, size, layers, wobble):
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, POSTER_W)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    rng_style = np.random.default_rng(seed)
    palette = make_palette(rng_style, mode, layers)
    bg_top, bg_bot = background(rng_style, mode)

    # gradient background, generated procedurally
    t = np.linspace(0, 1, 256)[:, None, None]
    grad = np.array(bg_top) * (1 - t) + np.array(bg_bot) * t   # row 0 = top
    ax.imshow(grad, extent=(0, POSTER_W, 0, 1), origin="upper",
              aspect="auto", zorder=0)

    lo, hi = CONFIG["size_range"]
    reach = np.interp(size, [lo, hi], [0.28, 1.0])
    width_scale = size
    n = layers
    cx = POSTER_W * rng_style.uniform(0.38, 0.62)

    for i in range(n):                       # 0 = back ... n-1 = front
        depth = i / max(n - 1, 1)
        rng = np.random.default_rng(seed * 1000 + i)   # stable per layer
        x_top = cx + rng.uniform(-1, 1) * CONFIG["layer_spread"] * POSTER_W \
            * (1.3 - depth)
        layer_reach = reach * rng.uniform(0.8, 1.0) * (0.9 + 0.1 * depth)
        paths = build_bolt(rng, x_top, layer_reach, wobble)
        color = palette[i]
        core = lighten(color, 0.55 + 0.4 * depth)
        base_w = (0.9 + 1.6 * depth) * width_scale * 1.6

        for pts, wf in paths:
            for k, gw in enumerate(CONFIG["glow_passes"]):
                ax.plot(pts[:, 0], pts[:, 1], color=color,
                        lw=gw * base_w * wf * (1.0 - 0.25 * depth),
                        alpha=(0.05 + 0.03 * depth) * (k + 1) * (1.4 - depth * 0.4),
                        solid_capstyle="round", solid_joinstyle="round",
                        zorder=2 + i * 2)
            ax.plot(pts[:, 0], pts[:, 1], color=core,
                    lw=max(0.5, base_w * wf * 0.55),
                    alpha=0.65 + 0.35 * depth,
                    solid_capstyle="round", solid_joinstyle="round",
                    zorder=3 + i * 2)

    ax.text(POSTER_W / 2, 0.055, "S T O R M   S I G N A L",
            ha="center", va="center", color="white", alpha=0.85,
            fontsize=14, fontweight="light", zorder=100)
    ax.text(POSTER_W / 2, 0.025, f"seed {seed}  ·  {mode.lower()}  ·  "
            f"{layers} layers", ha="center", va="center",
            color="white", alpha=0.5, fontsize=7, zorder=100)
    return fig


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Thunder Poster", page_icon="⚡", layout="centered")
st.title("⚡ Thunder Poster")
st.caption("A generative lightning poster in electric blues. "
           "Tweak the controls in the sidebar; the same seed always gives the same storm.")

if "seed" not in st.session_state:
    st.session_state.seed = int(np.random.default_rng().integers(0, 10_000))


def randomize_seed():
    st.session_state.seed = int(np.random.default_rng().integers(0, 10_000))


lo, hi = CONFIG["size_range"]
size = st.sidebar.slider("Lightning size", float(lo), float(hi), 0.8, 0.05)
layers = st.sidebar.slider("Layers", CONFIG["layers"][0], CONFIG["layers"][1], 6)
wobble = st.sidebar.slider("Wobble", 0.05, 1.2, CONFIG["wobble_default"], 0.01)
mode = st.sidebar.radio("Palette", ["Vivid", "Pastel"], horizontal=True)
st.sidebar.slider("Seed", 0, 9999, key="seed")
st.sidebar.button("🎲 New storm", on_click=randomize_seed)

fig = draw_poster(st.session_state.seed, mode, size, layers, wobble)
st.pyplot(fig)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=200, facecolor="black")
st.sidebar.download_button("Download PNG", buf.getvalue(),
                           file_name=f"thunder_poster_{st.session_state.seed}.png",
                           mime="image/png")
plt.close(fig)
