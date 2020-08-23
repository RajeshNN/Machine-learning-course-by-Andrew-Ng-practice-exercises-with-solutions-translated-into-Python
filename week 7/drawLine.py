def func(p1, p2, ax,):
#DRAWLINE Draws a line from point p1 to point p2
#   DRAWLINE(p1, p2) Draws a line from point p1 to point p2 and holds the
#   current figure

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]])
    return ax;
