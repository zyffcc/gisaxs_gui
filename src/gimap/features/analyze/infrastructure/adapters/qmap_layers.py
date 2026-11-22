"""Draw q-map annotations in reciprocal coordinates, including in vector files."""

from matplotlib.lines import Line2D


def draw_layers(axes, layers, font_pt):
    axes.set_xlim(layers["limits"][0])
    axes.set_ylim(layers["limits"][1])
    axes.set_aspect("equal", adjustable="box")
    for xs, ys, color in layers["region_lines"]:
        axes.plot(xs, ys, color=color, lw=0.8)
    for color, xs, ys in layers["arcs"]:
        axes.plot(xs, ys, color=color, lw=0.7, linestyle="--")
    for color, xs, ys, sizes, hollow in layers["spots"]:
        axes.scatter(xs, ys, s=[(size * 0.42)**2 for size in sizes], edgecolors=color,
                     facecolors=["none" if open_circle else color for open_circle in hollow], linewidths=0.6)
    for x, y, text, color in layers["labels"]:
        axes.annotate(text, (x, y), xytext=(2, 2), textcoords="offset points",
                      color=color, fontsize=max(4, font_pt - 2), clip_on=True)
    if layers["beam"] is not None:
        axes.plot(*layers["beam"], marker="+", color="#22d3ee", ms=7)
    if layers["horizon"] is not None:
        axes.axhline(layers["horizon"], color="#a1a1aa", lw=0.65, ls="--")
    entries = [Line2D([], [], color=color, label=name, lw=0.8) for name, color in layers["regions"]]
    spot_colors = {color for color, *_ in layers["spots"]}
    for name, color in layers["phases"]:
        style = {"marker": "o", "markersize": 4, "linestyle": "none"} if color in spot_colors else {"linestyle": "--"}
        entries.append(Line2D([], [], color=color, label=name, lw=0.8, **style))
    if entries:
        axes.figure.legend(handles=entries, loc="outside lower center", ncols=2,
                           fontsize=max(4, font_pt - 1), frameon=False)
