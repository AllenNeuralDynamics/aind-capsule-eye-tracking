from __future__ import annotations
import datetime

import pathlib
import random
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd
import utils

ELLIPSE_COLORS = POINT_COLORS = {
    "cr": "lime",
    "eye": "cyan",
    "pupil": "magenta",
}


def plot_video_frame_with_ellipses(
    video_path: str | pathlib.Path | cv2.VideoCapture, 
    all_ellipses: dict[utils.BodyPart, Sequence[utils.Ellipse] | pd.DataFrame], 
    frame_index: int | None = None,
    dlc_output_h5_path: str | pathlib.Path | None = None,
    ) -> plt.Figure:
    v = utils.get_video_data(video_path)
    if frame_index is None:
        frame_index = random.randint(0, utils.get_video_frame_count(v))
    dlc_df = None if dlc_output_h5_path is None else utils.get_dlc_df(dlc_output_h5_path)
    v.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, frame = v.read()
    fig = plt.figure(facecolor="0.5")
    ax = fig.add_subplot()
    im = ax.imshow(frame, aspect="equal", cmap="Greys", clim=[0, 150])
    for body_part, ellipses in all_ellipses.items():
        if isinstance(ellipses, pd.DataFrame):
            ellipse = utils.Ellipse(**ellipses.iloc[frame_index].to_dict())
        else:
            ellipse = ellipses[frame_index]
        assert isinstance(ellipse, utils.Ellipse), f"Expected Ellipse, got {type(ellipse)=}"
        # ax.plot(ellipse.center_x, ellipse.center_y, "o", color=ELLIPSE_COLORS[body_part], markersize=1)
        if not np.isnan(ellipse.center_x):
            ax.add_patch(
                matplotlib.patches.Ellipse(
                    xy=(ellipse.center_x, ellipse.center_y),
                    width=2*ellipse.width,
                    height=2*ellipse.height,
                    angle=ellipse.phi*180/np.pi,
                    fill=False,
                    color=ELLIPSE_COLORS[body_part],
                    linewidth=.1,
                    alpha=.8,
                )
            )
        if dlc_df is not None:
            xy = [utils.get_values_from_row(dlc_df.iloc[frame_index], annotation, body_part) for annotation in ('x', 'y')]
            plt.plot(*xy, "+", color=POINT_COLORS[body_part], markersize=1, alpha=1)
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labelbottom=False,
    )
    ax.set_title(
        (
            f"frame {frame_index}/{utils.get_video_frame_count(v)}"
            " | "
            f"time {datetime.timedelta(seconds=frame_index / v.get(cv2.CAP_PROP_FPS))}"
                " / "
                f"{datetime.timedelta(seconds=utils.get_video_frame_count(v) / v.get(cv2.CAP_PROP_FPS))}"
            f" | {im.get_clim() = }"
        ), 
        fontsize=8,
    )
    return fig
