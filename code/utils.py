from __future__ import annotations

import copy
import contextlib
import pathlib
from typing import Dict, Tuple, Iterator, Iterable, NamedTuple, Literal
import concurrent.futures
import functools

from typing_extensions import TypeAlias
import numpy as np
import pandas as pd
import cv2
import tqdm

DATA_PATH = pathlib.Path('/root/capsule/data/')
RESULTS_PATH = pathlib.Path('/root/capsule/results/')
QC_PATH = RESULTS_PATH / "qc" 

DLC_PROJECT_PATH = DATA_PATH / 'universal_eye_tracking-peterl-2019-07-10'
DLC_SCORER_NAME = 'DLC_resnet50_universal_eye_trackingJul10shuffle1_1030000'

DLC_LABELS = ('cr', 'eye', 'pupil')
VIDEO_SUFFIXES = ('.mp4', '.avi', '.wmv', '.mov')

BodyPart: TypeAlias = Literal['cr', 'eye', 'pupil']
Annotation: TypeAlias = Literal['x', 'y', 'likelihood']
AnnotationData: TypeAlias = Dict[Tuple[BodyPart, Annotation], float]

MIN_LIKELIHOOD_THRESHOLD = 0.01
MIN_NUM_POINTS_FOR_ELLIPSE_FITTING = 6 # at least 6 tracked points for annotation quality data

def get_eye_video_paths() -> Iterator[pathlib.Path]:
    yield from (
        p for p in DATA_PATH.rglob('*[eE]ye*') 
        if (
            DLC_PROJECT_PATH not in p.parents
            and p.suffix in VIDEO_SUFFIXES
        )
    )

def get_dlc_df(dlc_output_h5_path: str | pathlib.Path) -> pd.DataFrame:
    # df has MultiIndex 
    # TODO extract label from df
    return getattr(pd.read_hdf(dlc_output_h5_path), DLC_SCORER_NAME) 

def get_dlc_output_h5_path(
    input_video_file_path: str | pathlib.Path, 
    output_dir_path: str | pathlib.Path = RESULTS_PATH,
) -> pathlib.Path:
    output = next(
        pathlib.Path(output_dir_path)
        .rglob(
            glob := f"{pathlib.Path(input_video_file_path).stem}*.h5"
        ),
        None
    )
    if output is None:
        raise FileNotFoundError(f"No file matching {glob} in {output_dir_path}")
    return output

class Ellipse(NamedTuple):
    center_x: np.floating = np.nan
    center_y: np.floating = np.nan
    width: np.floating = np.nan
    """semi-major axis"""
    height: np.floating = np.nan
    """semi-minor axis"""
    phi: np.floating = np.nan
    """angle of counterclockwise rotation of major-axis in radians from x-axis"""

def is_in_ellipse(x: float, y: float, ellipse: Ellipse) -> bool:
    """check whether `(x, y)` is within the perimeter of `ellipse`.
    
    - used for validating pupil center and cr are within eye ellipse
    
    >>> is_in_ellipse(0, 3.1, Ellipse(0, 0, 3, 5, 0))
    False
    >>> is_in_ellipse(0, 3, Ellipse(0, 0, 3, 5, 0))
    True
    >>> is_in_ellipse(5.1, 0, Ellipse(0, 0, 3, 5, 0))
    False
    >>> is_in_ellipse(5, 0, Ellipse(0, 0, 3, 5, 0))
    True
    """
    n1 = np.cos(ellipse.phi) * (x - ellipse.center_x) + np.sin(ellipse.phi) * (y - ellipse.center_y)
    n2 = np.sin(ellipse.phi) * (x - ellipse.center_x) - np.cos(ellipse.phi) * (y - ellipse.center_y)
    if (n1 * n1)/(ellipse.height * ellipse.height) + (n2 * n2)/(ellipse.width * ellipse.width) <= 1:
        return True
    return False

def get_values_from_row(row: AnnotationData, annotation: Annotation, body_part: BodyPart) -> np.array:
    return np.array([v for k, v in row.items() if k[1] == annotation and body_part in k[0]])

class InvalidEigenVectors(ValueError):
    pass

def get_ellipses_from_row(row: AnnotationData) -> dict[BodyPart, Ellipse]:
    out = dict()
    for body_part in DLC_LABELS:
        arrays = {annotation: get_values_from_row(row, annotation, body_part) for annotation in ('x', 'y', 'likelihood')}
        ellipse = Ellipse() # default nan values
        likely = arrays["likelihood"] > MIN_LIKELIHOOD_THRESHOLD
        if len(arrays["likelihood"][likely]) >= MIN_NUM_POINTS_FOR_ELLIPSE_FITTING: 
            with contextlib.suppress(InvalidEigenVectors):
                ellipse = fit_ellipse([arrays["x"][likely], arrays["y"][likely]])
        out[body_part] = ellipse
    return out


def process_ellipses(dlc_output_h5_path: pathlib.Path, output_file_path: pathlib.Path) -> dict[BodyPart, pd.DataFrame]:
    output_file_path = pathlib.Path(output_file_path).with_suffix('.h5')
    dlc_df = get_dlc_df(dlc_output_h5_path)
    future_to_index = {}
    results = {}
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for idx, row in dlc_df.iterrows():
            future_to_index[
                executor.submit(get_ellipses_from_row, row.to_dict())
            ] = idx 
        for future in tqdm.tqdm(
                concurrent.futures.as_completed(future_to_index.keys()), 
                desc='fitting',
                unit='frames',
                total=len(dlc_df),
                ncols=79,
                ascii=True, 
            ):
            for body_part in DLC_LABELS:
                results.setdefault(
                    body_part, 
                    [None] * len(dlc_df),
                )[future_to_index[future]] = future.result()[body_part]

    output_file_path.touch()
    body_part_to_df = {}
    for body_part in DLC_LABELS:
        df = pd.DataFrame.from_records(results[body_part], columns=Ellipse._fields)
        body_part_to_df[body_part] = df
        df.to_hdf(output_file_path, key=body_part, mode='a')       
      
    return body_part_to_df


def fit_ellipse(data) -> Ellipse:
    """Least Squares fitting algorithm 
    
    Theory taken from (*)
    Solving equation Sa=lCa. with a = |a b c d f g> and a1 = |a b c> 
        a2 = |d f g>
    Args
    ----
    data (list:list:float): list of two lists containing the x and y data of the
        ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
    Returns
    ------
    coef (list): list of the coefficients describing an ellipse
        [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g

    uses https://github.com/bdhammel/least-squares-ellipse-fitting
    * based on the publication Halir, R., Flusser, J.: 'Numerically Stable Direct Least Squares Fitting of Ellipses'
    """

    x, y = np.asarray(data, dtype=float)
    #PL introduced weights!

    #Quadratic part of design matrix [eqn. 15] from (*)
    D1 = np.mat(np.vstack([x**2, x*y, y**2])).T
    
    #Linear part of design matrix [eqn. 16] from (*)
    D2 = np.mat(np.vstack([x, y, np.ones(len(x))])).T
    
    #forming scatter matrix [eqn. 17] from (*)
    S1 = D1.T*D1
    S2 = D1.T*D2
    S3 = D2.T*D2  
    
    #Constraint matrix [eqn. 18]
    C1 = np.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

    #Reduced scatter matrix [eqn. 29]
    M=C1.I*(S1-S2*S3.I*S2.T)

    #M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this equation [eqn. 28]
    eval, evec = np.linalg.eig(M) 

    # eigenvector must meet constraint 4ac - b^2 to be valid.
    cond = 4*np.multiply(evec[0, :], evec[2, :]) - np.power(evec[1, :], 2)
    a1 = evec[:, np.nonzero(cond.A > 0)[1]]
    
    #|d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
    a2 = -S3.I*S2.T*a1
    
    if not (a1.any() and a2.any()):
        raise InvalidEigenVectors()

    # eigenvectors |a b c d f g> 
    coef = np.vstack([a1, a2])     

    #eigenvectors are the coefficients of an ellipse in general form
    #a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
    a = coef[0,0]
    b = coef[1,0]/2.
    c = coef[2,0]
    d = coef[3,0]/2.
    f = coef[4,0]/2.
    g = coef[5,0]
    
    #finding center of ellipse [eqn.19 and 20] from (**)
    x0 = (c*d-b*f)/(b**2.-a*c)
    y0 = (a*f-b*d)/(b**2.-a*c)
    
    #Find the semi-axes lengths [eqn. 21 and 22] from (**)
    numerator = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    denominator1 = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    denominator2 = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    width = np.sqrt(numerator/denominator1)
    height = np.sqrt(numerator/denominator2)

    # angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
    # or [eqn. 26] from (***).
    phi = .5*np.arctan((2.*b)/(a-c))

    return Ellipse(
        center_x=x0.real,
        center_y=y0.real,
        width=width.real,
        height=height.real,
        phi=phi.real,
    )


def get_video_data(video_path: str | pathlib.Path | cv2.VideoCapture) -> cv2.VideoCapture:
    """Open video file as cv2.VideoCapture object."""
    if isinstance(video_path, cv2.VideoCapture):
        return video_path
    return cv2.VideoCapture(str(video_path))

def get_video_frame_count(video_path_or_data: str | pathlib.Path | cv2.VideoCapture) -> int:
    return int(get_video_data(video_path_or_data).get(cv2.CAP_PROP_FRAME_COUNT))

@functools.cache
def get_video_frame_size_xy(video_path_or_data: str | pathlib.Path | cv2.VideoCapture) -> Tuple[int, int]:
    return get_video_data(video_path_or_data).read()[1][:,:,0].shape.T

def is_in_frame(x: float, y: float, video_path_or_data: str | pathlib.Path | cv2.VideoCapture) -> bool:
    """Check if point is inside video frame."""
    w, h = get_video_frame_size_xy(video_path_or_data)
    return (0 <= x < w) and (0 <= y < h)


def get_pupil_area_pixels(
    pupil_ellipses: Iterable[Ellipse] | pd.DataFrame, 
    ) -> pd.Series:
    if not isinstance(pupil_ellipses, pd.DataFrame):
        pupil_ellipses = pd.DataFrame.from_records(pupil_ellipses, columns=Ellipse._fields)
    return compute_circular_areas(pupil_ellipses)

def compute_circular_areas(ellipse_params: pd.DataFrame) -> pd.Series:
    """Compute circular area of a pupil using half-major axis.
    
    Copied verbatim from allensdk.brain_observatory.gaze_mapping._gaze_mapper
    - temp use for getting pupil area: will add whole sdk soon
    - output is in pixel^2 and assumes square pixels

    Assume the pupil is a circle, and that as it moves off-axis
    with the camera, the observed ellipse semi-major axis remains the
    radius of the circle.

    Parameters
    ----------
    ellipse_params (pandas.DataFrame): A table of pupil parameters consisting
        of 5 columns: ("center_x", "center_y", "height", "phi", "width")
        and n-row timepoints.

        NOTE: For ellipse_params produced by the Deep Lab Cut pipeline,
        "width" and "height" columns, in fact, refer to the
        "half-width" and "half-height".

    Returns
    -------
        pandas.Series: A series of pupil areas for n-timepoints.
    """
    # Take the biggest value between height and width columns and
    # assume that it is the pupil circle radius.
    radii = ellipse_params[["height", "width"]].max(axis=1)
    return np.pi * radii * radii