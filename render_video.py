import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch

from traffic_model import predict_eta, solve_traffic_split, train_eta_model


FPS = 30
FRAME_COUNT = 450
OUTPUT_FILE = "traffic_simulation.mp4"
NODE_POSITIONS = {
    "A": np.array([1.2, 5.0]),
    "B": np.array([4.3, 8.0]),
    "C": np.array([4.3, 2.0]),
    "D": np.array([8.1, 5.0]),
}


def _matrix_text(matrix):
    return np.array2string(
        matrix, precision=2, suppress_small=True, max_line_width=52
    )


def _point_on_route(route1, progress, incident_slow):
    start = NODE_POSITIONS["A"]
    middle = NODE_POSITIONS["B"] if route1 else NODE_POSITIONS["C"]
    destination = NODE_POSITIONS["D"]

    if progress < 0.5:
        fraction = progress * 2.0
        point = start + fraction * (middle - start)
    else:
        fraction = (progress - 0.5) * 2.0
        if incident_slow and route1:
            fraction *= 0.2
        point = middle + fraction * (destination - middle)
    return point


def render_video():
    """Render the 15-second split-screen simulation to an MP4 file."""
    features, _, weights, snapshots = train_eta_model()
    del features

    plt.style.use("dark_background")
    figure = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#10151c")
    grid = GridSpec(
        1, 2, figure=figure, width_ratios=(3, 2), wspace=0.045,
        left=0.035, right=0.98, top=0.96, bottom=0.055,
    )
    road_ax = figure.add_subplot(grid[0, 0])
    hud_ax = figure.add_subplot(grid[0, 1])
    road_ax.set_facecolor("#10151c")
    road_ax.set_xlim(0, 9.3)
    road_ax.set_ylim(0, 10)
    road_ax.set_aspect("equal")
    road_ax.axis("off")
    hud_ax.set_facecolor("#10151c")
    hud_ax.axis("off")

    a, b, c, d = (NODE_POSITIONS[key] for key in ("A", "B", "C", "D"))
    road_color = "#40505e"
    for first, second in ((a, b), (b, d), (a, c), (c, d)):
        road_ax.plot(
            [first[0], second[0]], [first[1], second[1]],
            color="#1e2934", linewidth=18, solid_capstyle="round", zorder=1,
        )
        road_ax.plot(
            [first[0], second[0]], [first[1], second[1]],
            color=road_color, linewidth=8, solid_capstyle="round", zorder=2,
        )
        road_ax.plot(
            [first[0], second[0]], [first[1], second[1]],
            color="#82909b", linewidth=1.2, linestyle=(0, (5, 6)),
            alpha=0.65, zorder=3,
        )

    for label, point in NODE_POSITIONS.items():
        road_ax.add_patch(
            Circle(point, 0.34, facecolor="#16232d", edgecolor="#e8eef2",
                   linewidth=1.8, zorder=5)
        )
        road_ax.text(
            point[0], point[1], label, ha="center", va="center",
            color="white", fontsize=13, weight="bold", zorder=6,
        )
    road_ax.text(b[0], b[1] + 0.58, "TOP HIGHWAY", ha="center",
                 color="#b5c1ca", fontsize=9)
    road_ax.text(c[0], c[1] - 0.62, "SIDE STREET", ha="center",
                 color="#b5c1ca", fontsize=9)
    road_ax.text(a[0] - 0.1, a[1] + 0.65, "START", ha="center",
                 color="#b5c1ca", fontsize=9)
    road_ax.text(d[0] + 0.05, d[1] + 0.65, "DESTINATION", ha="center",
                 color="#b5c1ca", fontsize=9)
    road_ax.text(
        0.1, 9.55, "GPS TRAFFIC ETA  /  LIVE ROUTE ASSIGNMENT",
        color="#f0f4f7", fontsize=14, weight="bold",
    )

    car_count = 35
    offsets = np.arange(car_count, dtype=float) / car_count
    car_scatter = road_ax.scatter(
        [], [], s=38, edgecolors="#10151c", linewidths=0.7, zorder=8
    )
    accident_marker = road_ax.scatter(
        [5.9], [6.75], marker="*", s=360, c="#ff334d",
        edgecolors="#ffd4da", linewidths=1.0, zorder=9, visible=False,
    )
    accident_text = road_ax.text(
        5.95, 7.27, "", ha="center", color="#ff5268", fontsize=12,
        weight="bold", zorder=10,
    )
    time_text = road_ax.text(
        0.1, 0.35, "", color="#b5c1ca", fontsize=11, family="monospace",
    )
    eta_text = road_ax.text(
        0.1, 0.78, "", color="#7ef0ad", fontsize=14, weight="bold",
    )
    route_text = road_ax.text(
        0.1, 1.17, "", color="#dce5eb", fontsize=10, family="monospace",
    )

    cards = (
        (0.02, 0.675, 0.96, 0.305, "ML TRAINING  /  NORMAL EQUATIONS"),
        (0.02, 0.355, 0.96, 0.295, "LIVE ETA  /  DOT PRODUCT"),
        (0.02, 0.025, 0.96, 0.305, "FLOW CONSERVATION  /  DIAMOND NETWORK"),
    )
    for x, y, width, height, title in cards:
        hud_ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=hud_ax.transAxes, facecolor="#18212b",
                edgecolor="#334454", linewidth=1.0,
            )
        )
        hud_ax.text(
            x + 0.025, y + height - 0.045, title,
            transform=hud_ax.transAxes, color="#8fe3ff",
            fontsize=10, weight="bold", family="monospace",
        )

    training_text = hud_ax.text(
        0.06, 0.72, "", transform=hud_ax.transAxes,
        color="#e2e8ed", fontsize=8.5, family="monospace",
        va="top", linespacing=1.25,
    )
    eta_hud_text = hud_ax.text(
        0.06, 0.59, "", transform=hud_ax.transAxes,
        color="#e2e8ed", fontsize=10, family="monospace",
        va="top", linespacing=1.5,
    )
    flow_text = hud_ax.text(
        0.06, 0.27, "", transform=hud_ax.transAxes,
        color="#e2e8ed", fontsize=8.5, family="monospace",
        va="top", linespacing=1.25,
    )

    def initialize():
        car_scatter.set_offsets(np.empty((0, 2)))
        return (
            car_scatter, accident_marker, accident_text, time_text, eta_text,
            route_text, training_text, eta_hud_text, flow_text,
        )

    def update(frame):
        time_seconds = frame / FPS
        incident = 180 <= frame < 210
        accident_marker.set_visible(incident)
        accident_text.set_text("ACCIDENT ON B→D" if incident else "")

        raw_progress = frame / 240.0 + offsets
        trip_number = np.floor(raw_progress)
        progress = raw_progress - trip_number
        trip_start = (trip_number - offsets) * 240.0
        initial_route1 = np.arange(car_count) < int(car_count * 0.75)
        route1 = np.where(trip_start >= 210.0, False, initial_route1)
        on_accident_segment = incident & route1 & (progress >= 0.5)
        positions = np.array(
            [
                _point_on_route(
                    bool(route1[index]), progress[index],
                    bool(on_accident_segment[index]),
                )
                for index in range(car_count)
            ]
        )
        colors = np.where(route1, "#39e879", "#ffd166").astype(object)
        colors[on_accident_segment] = "#ff3b55"
        car_scatter.set_offsets(positions)
        car_scatter.set_color(colors.tolist())

        volume = 55.0 if frame < 210 else 100.0
        weather = 0.2
        accident_flag = 1.0 if incident else 0.0
        eta = predict_eta(weights, volume, weather, accident_flag)
        route1_flow, route2_flow, flow_matrix = solve_traffic_split(
            100.0, frame >= 180
        )
        eta_text.set_text(f"Predicted ETA: ~{eta:.1f} min")
        route_text.set_text(
            f"Route 1  {route1_flow:.0f}%   /   "
            f"Route 2  {route2_flow:.0f}%"
        )
        time_text.set_text(
            f"t = {time_seconds:04.1f}s    FRAME {frame:03d} / {FRAME_COUNT - 1}"
        )

        if frame < 150:
            snapshot_index = (frame // 9) % len(snapshots)
            matrix_display = snapshots[snapshot_index]
            training_text.set_text(
                "(XᵀX)w = Xᵀy   |   REF snapshot "
                f"{snapshot_index + 1}/{len(snapshots)}\n"
                + _matrix_text(matrix_display)
            )
        else:
            weight_lines = np.array2string(
                weights, precision=3, suppress_small=True, max_line_width=52
            )
            training_text.set_text(
                "(XᵀX)w = Xᵀy   |   SOLVED\n"
                f"w = {weight_lines}\n"
                "features: [bias, volume, weather, accident]"
            )

        eta_hud_text.set_text(
            "ETA = x_live · w\n"
            f"x_live = [1, {volume:.0f}, {weather:.1f}, {accident_flag:.0f}]\n"
            f"ETA = {eta:.2f} min"
        )
        flow_text.set_text(
            "[M | b]  columns: AB, BD, AC, CD | demand=100\n"
            + _matrix_text(flow_matrix)
            + f"\nRoute 1: {route1_flow:.0f}%   Route 2: {route2_flow:.0f}%"
        )

        if frame % 50 == 0 or frame == FRAME_COUNT - 1:
            print(f"Rendering frame {frame}/{FRAME_COUNT - 1}", flush=True)
        return (
            car_scatter, accident_marker, accident_text, time_text, eta_text,
            route_text, training_text, eta_hud_text, flow_text,
        )

    animation = FuncAnimation(
        figure, update, frames=FRAME_COUNT, init_func=initialize,
        interval=1000.0 / FPS, blit=False, repeat=False,
    )
    writer = FFMpegWriter(fps=FPS, metadata={"title": "GPS Traffic ETA"})
    animation.save(OUTPUT_FILE, writer=writer, dpi=100)
    plt.close(figure)
    print(f"Saved {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    render_video()
