"""
Interactive 3D Generative Thunder Poster (matplotlib)
------------------------------------------------------
Pure matplotlib (mpl_toolkits.mplot3d), no external data. Same fractal
lightning-bolt generator as before, but every bolt is now extruded into a
thin 3D "glass shard" and the layers are stacked front-to-back in depth,
so nearer bolts read larger/brighter and further ones recede - with real
matplotlib shading (Poly3DCollection(shade=True)) giving each bolt a
beveled, glossy look instead of a flat fill.

Controls:
  - Size slider     : scales every bolt from its top anchor point
  - Layers slider    : shows/hides pre-generated bolts (3-10), stacked in depth
  - Tilt slider      : camera elevation
  - Rotate slider     : camera azimuth (spin the poster in 3D)
  - Vivid / Pastel   : switches the color treatment
  - Shuffle button   : regenerates the whole composition

Run:  python3 thunder_poster_3d.py

NOTE: needs an interactive matplotlib backend (TkAgg, Qt5Agg, MacOSX...) to
show and respond to the widgets - see the note at the bottom of the file if
nothing appears. 3D scenes with many wobbly, branching bolts also have a lot
of triangles/quads to shade, so on older machines dragging the sliders may
feel a little less snappy than the flat 2D version; lower WOBBLE_ITER_RANGE
below if you want more headroom.
"""

import math
import random
import colorsys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, RadioButtons, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------
CANVAS_W, CANVAS_H = 8.5, 11              # poster aspect ratio (portrait)
MAX_LAYERS = 10
ELECTRIC_BLUE_HUE_RANGE = (0.50, 0.70)    # cyan-blue -> blue-violet (colorsys hue, 0-1)

WOBBLE_AMP_RANGE   = (0.06, 0.22)
WOBBLE_ITER_RANGE  = (3, 5)               # kept a touch lower than the 2D version - each
                                           # extra level roughly doubles the 3D face count
ROUGHNESS_RANGE    = (0.45, 0.68)
WIDTH_RANGE        = (0.055, 0.115)
TAPER_POWER_RANGE  = (1.1, 2.4)
BRANCH_PROB        = 0.85
BRANCH_COUNT_RANGE = (1, 3)

BASE_LENGTH_FRAC = 0.95
TOP_JITTER_FRAC  = 0.02
CAMERA_ZOOM      = 5.5     # smaller = more zoomed in (compensates mplot3d's default padding)

DEPTH_STEP        = 0.55   # z-distance between consecutive layers
BOLT_THICKNESS    = 0.16   # extrusion depth of the main ribbon
BRANCH_THICKNESS  = 0.08
CORE_THICKNESS    = 0.06

POSTER_TITLE = "THUNDER"


# ----------------------------------------------------------------------
# FRACTAL LIGHTNING PATH (midpoint displacement) - unchanged, still 2D
# ----------------------------------------------------------------------
def displace(p0, p1, depth, amp, rng, roughness):
    if depth <= 0:
        return [p0, p1]
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    perp = (-dy / length, dx / length) if length else (0.0, 0.0)
    offset = rng.uniform(-1, 1) * amp
    mid = (mx + perp[0] * offset, my + perp[1] * offset)
    left = displace(p0, mid, depth - 1, amp * roughness, rng, roughness)
    right = displace(mid, p1, depth - 1, amp * roughness, rng, roughness)
    return left[:-1] + right


def path_to_polygon(pts, width0, tip_width, taper_power):
    pts = np.array(pts)
    n = len(pts)
    left, right = [], []
    for i in range(n):
        if i == 0:
            d = pts[1] - pts[0]
        elif i == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[i + 1] - pts[i - 1]
        dl = np.hypot(*d)
        normal = np.array([-d[1], d[0]]) / dl if dl else np.array([0.0, 0.0])
        t = i / (n - 1)
        w = tip_width + (width0 - tip_width) * (1 - t) ** taper_power
        left.append(pts[i] + normal * w / 2)
        right.append(pts[i] - normal * w / 2)
    return np.array(left + right[::-1])


def rotate_scale_translate(poly, angle_deg, scale, translate):
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (poly @ R.T) * scale + np.array(translate)


def extrude_polygon_3d(poly2d, depth_center, thickness):
    """Turn a closed 2D poster-space polygon (x, y_up) into a thin 3D solid:
    a front cap, a back cap, and a ring of side quads connecting them - a
    flat "glass shard". Uses the mplot3d convention x=horizontal,
    y=depth-into-the-screen, z=vertical, so the poster's own x/y map to the
    3D x/z and only the (new) depth axis is added as the 3D y."""
    n = len(poly2d)
    front = [(x, depth_center + thickness / 2, y) for x, y in poly2d]
    back = [(x, depth_center - thickness / 2, y) for x, y in poly2d]
    faces = [front, back[::-1]]
    for i in range(n):
        j = (i + 1) % n
        faces.append([front[i], front[j], back[j], back[i]])
    return faces


# ----------------------------------------------------------------------
# BOLT GENERATION (local/unit space; independent of size, depth, color mode)
# ----------------------------------------------------------------------
def make_bolt(rng):
    wobble_amp = rng.uniform(*WOBBLE_AMP_RANGE)
    iterations = rng.randint(*WOBBLE_ITER_RANGE)
    roughness = rng.uniform(*ROUGHNESS_RANGE)
    width0 = rng.uniform(*WIDTH_RANGE)
    taper_power = rng.uniform(*TAPER_POWER_RANGE)

    pts = displace((0.0, 0.0), (0.0, -1.0), iterations, wobble_amp, rng, roughness)
    main_poly = path_to_polygon(pts, width0, width0 * 0.06, taper_power)

    branches = []
    if rng.random() < BRANCH_PROB:
        pts_arr = np.array(pts)
        for _ in range(rng.randint(*BRANCH_COUNT_RANGE)):
            idx = rng.randint(int(len(pts_arr) * 0.2), int(len(pts_arr) * 0.75))
            origin = pts_arr[idx]
            b_length = rng.uniform(0.2, 0.45)
            b_angle = rng.uniform(25, 70) * rng.choice([-1, 1])
            b_pts = displace((0.0, 0.0), (0.0, -b_length),
                              max(1, iterations - 2), wobble_amp * 0.8, rng, roughness)
            b_poly = path_to_polygon(b_pts, width0 * 0.45, width0 * 0.03, taper_power)
            b_poly = rotate_scale_translate(b_poly, b_angle, 1.0, (0, 0)) + origin
            branches.append(b_poly)

    return {
        "main": main_poly,
        "branches": branches,
        "angle": rng.uniform(-13, 13),
        "hue_jitter": rng.uniform(-0.05, 0.05),
        "sat_rand": rng.random(),
        "light_rand": rng.random(),
        "alpha_rand": rng.random(),
        "has_core": rng.random() < 0.5,
    }


# ----------------------------------------------------------------------
# COMPOSITION STATE
# ----------------------------------------------------------------------
class Poster:
    def __init__(self):
        self.rng = random.Random()
        self.size_scale = 1.0
        self.num_layers = 6
        self.color_mode = "vivid"
        self.elev = 16
        self.azim = -80
        self.shuffle()

    def shuffle(self):
        self.rng.seed(random.randint(0, 10_000_000))
        self.hue_base = self.rng.uniform(*ELECTRIC_BLUE_HUE_RANGE)
        self.study_no = self.rng.randint(1000, 9999)
        self.bolts = [make_bolt(self.rng) for _ in range(MAX_LAYERS)]

        margin_x = CANVAS_W * 0.13
        for i, b in enumerate(self.bolts):
            frac = margin_x + (i + 0.5) / MAX_LAYERS * (CANVAS_W - 2 * margin_x)
            b["x"] = frac + self.rng.uniform(-CANVAS_W * 0.035, CANVAS_W * 0.035)
            b["top_jitter"] = self.rng.uniform(0, TOP_JITTER_FRAC) * CANVAS_H

    def bolt_color(self, b):
        hue = (self.hue_base + b["hue_jitter"]) % 1.0
        if self.color_mode == "vivid":
            s = 0.65 + b["sat_rand"] * 0.35
            v = 0.55 + b["light_rand"] * 0.35
        else:
            s = 0.28 + b["sat_rand"] * 0.30
            v = 0.85 + b["light_rand"] * 0.15
        return colorsys.hsv_to_rgb(hue, s, v)

    def bolt_alpha(self, b):
        if self.color_mode == "vivid":
            return 0.65 + b["alpha_rand"] * 0.3
        return 0.72 + b["alpha_rand"] * 0.25

    def palette(self):
        hue = self.hue_base
        if self.color_mode == "vivid":
            bg = colorsys.hsv_to_rgb(hue, 0.55, 0.045)
            text_color = colorsys.hsv_to_rgb(hue, 0.15, 0.98)
            border_color = colorsys.hsv_to_rgb(hue, 0.20, 0.90)
            core_color = colorsys.hsv_to_rgb(hue, 0.10, 0.99)
        else:
            bg = colorsys.hsv_to_rgb(hue, 0.30, 0.95)
            text_color = colorsys.hsv_to_rgb(hue, 0.45, 0.30)
            border_color = colorsys.hsv_to_rgb(hue, 0.30, 0.40)
            core_color = colorsys.hsv_to_rgb(hue, 0.08, 0.995)
        return dict(bg=bg, text=text_color, border=border_color, core=core_color)


# ----------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------
def draw(ax, hud_ax, poster: "Poster"):
    ax.cla()
    depth_range = DEPTH_STEP * (MAX_LAYERS + 2)
    # mplot3d convention: x=horizontal, y=depth-into-the-screen, z=vertical.
    # The poster's own width/height map straight onto x/z so "up" is really
    # up on screen; the new depth axis is what gives the layers parallax.
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(-depth_range / 2, depth_range / 2)
    ax.set_zlim(0, CANVAS_H)
    ax.margins(0)
    try:
        ax.set_box_aspect((CANVAS_W, depth_range, CANVAS_H))
    except AttributeError:
        pass  # older matplotlib without set_box_aspect
    # Orthographic (parallel) projection instead of the default perspective:
    # a poster is a flat design object, so bolts should keep a predictable
    # size/position as the scene rotates, rather than shrinking with camera
    # distance the way a photographed 3D object would.
    try:
        ax.set_proj_type("ortho")
    except AttributeError:
        pass
    ax.set_axis_off()
    ax.view_init(elev=poster.elev, azim=poster.azim)
    # mplot3d always leaves room to fit the *entire* 3D bounding box (as if
    # viewed along its longest diagonal) no matter the current camera angle,
    # which otherwise leaves a large, fixed empty margin around a poster-
    # shaped (tall, shallow) scene like this one. Pulling the camera in via
    # the private `_dist` knob compensates for that so the bolts actually
    # reach the poster's edges. This is undocumented/private API, so it's
    # wrapped defensively in case a future matplotlib version removes it.
    try:
        ax._dist = CAMERA_ZOOM
    except AttributeError:
        pass

    pal = poster.palette()
    fig = ax.figure
    fig.patch.set_facecolor(pal["bg"])
    # The background is set on the axes' own 2D patch rather than as a 3D
    # plane: mplot3d only sorts whole collections by an average depth (no
    # real z-buffer), so a single huge backdrop plane spanning the same
    # depth range as many smaller bolts gets mis-sorted and can occlude
    # them entirely. The axes patch sits behind the 3D scene unconditionally.
    ax.set_facecolor(pal["bg"])
    ax.patch.set_alpha(1.0)

    pixel_scale = CANVAS_H * BASE_LENGTH_FRAC * poster.size_scale
    visible = poster.bolts[:poster.num_layers]
    n_visible = len(visible)

    for i, b in enumerate(visible):
        cx = b["x"]
        cy = CANVAS_H - b["top_jitter"]
        # front-most bolt (i=0) closest to the viewer, receding by depth after
        depth_center = depth_range / 2 - DEPTH_STEP * 1.5 - i * DEPTH_STEP
        color = poster.bolt_color(b)
        alpha = poster.bolt_alpha(b)
        # bolts further back are slightly dimmer/less saturated, for aerial-perspective depth
        depth_fade = 1.0 - 0.35 * (i / max(1, n_visible - 1))
        alpha *= depth_fade

        for poly in b["branches"]:
            world = rotate_scale_translate(poly, b["angle"], pixel_scale, (cx, cy))
            faces = extrude_polygon_3d(world, depth_center + BRANCH_THICKNESS, BRANCH_THICKNESS)
            ax.add_collection3d(Poly3DCollection(
                faces, facecolors=color, edgecolors=color, linewidths=0,
                alpha=alpha * 0.7, shade=True))

        world_main = rotate_scale_translate(b["main"], b["angle"], pixel_scale, (cx, cy))
        faces_main = extrude_polygon_3d(world_main, depth_center, BOLT_THICKNESS)
        ax.add_collection3d(Poly3DCollection(
            faces_main, facecolors=color, edgecolors=color, linewidths=0,
            alpha=alpha, shade=True))

        if b["has_core"]:
            core_local = b["main"] * 0.55
            core_world = rotate_scale_translate(core_local, b["angle"], pixel_scale, (cx, cy))
            faces_core = extrude_polygon_3d(
                core_world, depth_center + BOLT_THICKNESS / 2 + CORE_THICKNESS, CORE_THICKNESS)
            ax.add_collection3d(Poly3DCollection(
                faces_core, facecolors=pal["core"], edgecolors=pal["core"],
                linewidths=0, alpha=0.55 * depth_fade, shade=False))

    # HUD overlay (title, subtitle, border): a separate transparent 2D axes
    # stacked on top of the 3D scene. Plain Artist/Patch objects can't be
    # added straight onto an Axes3D (it requires everything drawn there to
    # support do_3d_projection), so the poster frame/text lives here instead
    # - it also has the benefit of staying flat and legible regardless of
    # how the 3D scene underneath is rotated.
    hud_ax.clear()
    hud_ax.set_xlim(0, 1)
    hud_ax.set_ylim(0, 1)
    hud_ax.set_axis_off()
    hud_ax.patch.set_alpha(0)
    hud_ax.text(0.5, 0.085, POSTER_TITLE, ha="center", va="center",
                fontsize=40, fontweight="bold", color=pal["text"],
                family="monospace", zorder=100)
    hud_ax.text(0.5, 0.055, f"GENERATIVE STUDY NO. {poster.study_no:04d}",
                ha="center", va="center", fontsize=9, color=pal["text"],
                alpha=0.75, family="monospace", zorder=100)
    hud_ax.add_patch(Rectangle((0.025, 0.02), 0.95, 0.96,
                                fill=False, edgecolor=pal["border"], linewidth=1.1,
                                alpha=0.6, zorder=100))

    fig.canvas.draw_idle()


# ----------------------------------------------------------------------
# INTERACTIVE APP
# ----------------------------------------------------------------------
def main():
    poster = Poster()

    fig = plt.figure(figsize=(CANVAS_W, CANVAS_H + 2.3), dpi=150)
    poster_rect = [0.02, 0.24, 0.96, 0.74]
    ax = fig.add_axes(poster_rect, projection="3d")
    hud_ax = fig.add_axes(poster_rect)   # transparent overlay for title/border
    hud_ax.patch.set_alpha(0)
    draw(ax, hud_ax, poster)

    ax_size = fig.add_axes([0.15, 0.155, 0.55, 0.025])
    s_size = Slider(ax_size, "Size", 0.5, 1.6, valinit=poster.size_scale,
                     valstep=0.02, color="#6fa8ff")

    ax_layers = fig.add_axes([0.15, 0.115, 0.55, 0.025])
    s_layers = Slider(ax_layers, "Layers", 3, MAX_LAYERS, valinit=poster.num_layers,
                       valstep=1, color="#6fa8ff")

    ax_elev = fig.add_axes([0.15, 0.075, 0.55, 0.025])
    s_elev = Slider(ax_elev, "Tilt", -20, 60, valinit=poster.elev,
                     valstep=1, color="#6fa8ff")

    ax_azim = fig.add_axes([0.15, 0.035, 0.55, 0.025])
    s_azim = Slider(ax_azim, "Rotate", -180, 180, valinit=poster.azim,
                     valstep=1, color="#6fa8ff")

    ax_radio = fig.add_axes([0.76, 0.035, 0.16, 0.09])
    ax_radio.set_facecolor("#1b1e27")
    radio = RadioButtons(ax_radio, ("Vivid", "Pastel"), active=0)
    for label in radio.labels:
        label.set_color("#eef1f8")
        label.set_fontsize(9)

    ax_button = fig.add_axes([0.76, 0.135, 0.16, 0.045])
    b_shuffle = Button(ax_button, "Shuffle", color="#1b1e27", hovercolor="#2a2e3a")
    b_shuffle.label.set_color("#eef1f8")

    def on_size(val):
        poster.size_scale = val
        draw(ax, hud_ax, poster)

    def on_layers(val):
        poster.num_layers = int(val)
        draw(ax, hud_ax, poster)

    def on_elev(val):
        poster.elev = val
        draw(ax, hud_ax, poster)

    def on_azim(val):
        poster.azim = val
        draw(ax, hud_ax, poster)

    def on_palette(label):
        poster.color_mode = label.lower()
        draw(ax, hud_ax, poster)

    def on_shuffle(event):
        poster.shuffle()
        draw(ax, hud_ax, poster)

    s_size.on_changed(on_size)
    s_layers.on_changed(on_layers)
    s_elev.on_changed(on_elev)
    s_azim.on_changed(on_azim)
    radio.on_clicked(on_palette)
    b_shuffle.on_clicked(on_shuffle)

    plt.show()


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
# If nothing appears, or the sliders don't respond:
#   - You're likely on a headless/inline/Agg backend. Try:
#       pip install PyQt5        (or) sudo apt install python3-tk
#     and/or force a backend near the top of this file with:
#       import matplotlib; matplotlib.use("TkAgg")   # or "Qt5Agg"
#   - You can also grab-and-drag directly on the 3D scene with the mouse to
#     rotate it; the Tilt/Rotate sliders will snap back to their own values
#     next time you move any slider, since they're the single source of
#     truth for the camera angle.
# ----------------------------------------------------------------------
