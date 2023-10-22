from __future__ import annotations
import datetime

import pathlib
import random
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils

POINT_COLORS = {
    "cr": "green",
    "eye": "blue",
    "pupil": "red",
}
ELLIPSE_COLORS = {
    "cr": "lime",
    "eye": "cyan",
    "pupil": "magenta",
}

def plot_video_frame_with_ellipses(video_path: str | pathlib.Path | cv2.VideoCapture, all_ellipses: dict[utils.BodyPart, Sequence[utils.Ellipse] | pd.DataFrame], frame_index: int | None = None) -> plt.Figure:
    v = utils.get_video_data(video_path)
    if frame_index is None:
        frame_index = random.randint(0, utils.get_video_frame_count(v))
    v.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, frame = v.read()
    fig = plt.figure(facecolor="0.5")
    ax = fig.add_subplot(111)
    im = ax.imshow(frame, aspect="equal")
    im.set_clim([0, 175])
    plt.colorbar(im, ax=ax)
    for body_part, ellipses in all_ellipses.items():
        if isinstance(ellipses, pd.DataFrame):
            ellipse = utils.Ellipse(ellipses.iloc[frame_index].to_dict())
        else:
            ellipse = ellipses[frame_index]
        if np.isnan(ellipse.center_x):
            continue
        ax.add_patch(
            plt.Ellipse(
                (ellipse.center_x, ellipse.center_y),
                ellipse.width,
                ellipse.height,
                ellipse.phi,
                fill=False,
                color=ELLIPSE_COLORS[body_part],
                linewidth=.5,
            )
        )
        ax.plot(ellipse.center_x, ellipse.center_y, "+", color=POINT_COLORS[body_part], markersize=1)
    ax.axis("off")
    # ax.tick_params(
    #     top=False,
    #     bottom=False,
    #     left=False,
    #     right=False,
    #     labelleft=False,
    #     labelbottom=False,
    # )
    ax.set_title(
        datetime.timedelta(seconds=frame_index / v.get(cv2.CAP_PROP_FPS)), fontsize=8,
    )
    return fig
