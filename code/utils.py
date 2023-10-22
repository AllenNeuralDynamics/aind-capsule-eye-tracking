from __future__ import annotations

import pathlib
from typing import Dict, Tuple, Iterator, NamedTuple, Literal
import concurrent.futures
import functools

from typing_extensions import TypeAlias
import numpy as np
import pandas as pd
import tqdm

DATA_PATH = pathlib.Path('/root/capsule/data/')
RESULTS_PATH = pathlib.Path('/root/capsule/results/')

DLC_PROJECT_PATH = DATA_PATH / 'universal_eye_tracking-peterl-2019-07-10'
DLC_SCORER_NAME = 'DLC_resnet50_universal_eye_trackingJul10shuffle1_1030000'

DLC_LABELS = ('cr', 'eye', 'pupil')

VIDEO_SUFFIXES = ('.mp4', '.avi', '.wmv', '.mov')

BodyPart: TypeAlias = Literal['cr', 'eye', 'pupil']

def get_eye_video_paths() -> Iterator[pathlib.Path]:
    yield from (
        p for p in DATA_PATH.rglob('*[eE]ye*') 
        if (
            DLC_PROJECT_PATH not in p.parents
            and p.suffix in VIDEO_SUFFIXES
        )
    )


def get_dlc_output_h5_path(
    input_video_file_path: str | pathlib.Path, 
    output_dir_path: str | pathlib.Path = RESULTS_PATH,
):
    return next(
        pathlib.Path(output_dir_path)
        .rglob(
            f"{pathlib.Path(input_video_file_path).stem}*.h5"
        )
    )


class Ellipse(NamedTuple):
    center_x: np.floating = np.nan
    center_y: np.floating = np.nan
    width: np.floating = np.nan
    height: np.floating = np.nan
    phi: np.floating = np.nan
    """angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)"""


def fit_ellipse(data) -> Ellipse:
    """Lest Squares fitting algorithm 
    
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
    
    # eigenvectors |a b c d f g> 
    coef = np.vstack([a1, a2])
     
    """finds the important parameters of the fitted ellipse
    
    Theory taken form http://mathworld.wolfram
    Args
    -----
    coef (list): list of the coefficients describing an ellipse
        [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
    Returns
    _______
    center (List): of the form [x0, y0]
    width (float): major axis 
    height (float): minor axis
    phi (float): rotation of major axis form the x-axis in radians 
    """

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
        center_x=x0,
        center_y=y0,
        width=width,
        height=height,
        phi=phi,
    )


def make_test_ellipse(center=[1,1], width=1, height=.6, phi=3.14/5):
    """Generate Elliptical data with noise
    
    Args
    ----
    center (list:float): (<x_location>, <y_location>)
    width (float): semimajor axis. Horizontal dimension of the ellipse (**)
    height (float): semiminor axis. Vertical dimension of the ellipse (**)
    phi (float:radians): tilt of the ellipse, the angle the semimajor axis
        makes with the x-axis 
    Returns
    -------
    data (list:list:float): list of two lists containing the x and y data of the
        ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
    """
    t = np.linspace(0, 2*np.pi, 1000)
    x_noise, y_noise = np.random.rand(2, len(t))
    
    ellipse_x = center[0] + width*np.cos(t)*np.cos(phi)-height*np.sin(t)*np.sin(phi) + x_noise/2.
    ellipse_y = center[1] + width*np.cos(t)*np.sin(phi)+height*np.sin(t)*np.cos(phi) + y_noise/2.

    return [ellipse_x, ellipse_y]

#TODO make plot fn (provide video frame, draw ellipse, annotations)

Annotation: TypeAlias = Literal['x', 'y', 'likelihood']

AnnotationData: TypeAlias = Dict[Tuple[BodyPart, Annotation], float]

def get_values_from_row(row: AnnotationData, annotation: Annotation, body_part: BodyPart) -> np.array:
    return np.array([v for k, v in row.items() if k[1] == annotation and body_part in k[0]])

def get_ellipses_from_row(row: AnnotationData) -> dict[BodyPart, Ellipse]:
    likelihood_threshold = 0.2
    min_num_points = 6                  # at least 6 tracked points for annotation quality data

    out = dict()
    for body_part in DLC_LABELS:
        arrays = {annotation: get_values_from_row(row, annotation, body_part) for annotation in ('x', 'y', 'likelihood')}
        ellipse = Ellipse() # default nan values
        likely = arrays["likelihood"] > likelihood_threshold
        if len(arrays["likelihood"][likely]) >= min_num_points: 
            try:
                ellipse = fit_ellipse([arrays["x"][likely], arrays["y"][likely]])
            except Exception as e:
                print(e) 
        out[body_part] = ellipse
    return out


def process_ellipses(dlc_output_h5_path: pathlib.Path, output_file_path: pathlib.Path) -> dict[BodyPart, pd.DataFrame]:
    output_file_path = pathlib.Path(output_file_path).with_suffix('.h5')
    # df has MultiIndex 
    # TODO extract label from df
    df = getattr(pd.read_hdf(dlc_output_h5_path), DLC_SCORER_NAME) 

    future_to_index = {}
    results = dict.fromkeys(DLC_LABELS, [None] * len(df))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for idx, row in df.iterrows():
            future_to_index[
                executor.submit(get_ellipses_from_row, row.to_dict())
            ] = idx 
        for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_index.keys())):
            for body_part in DLC_LABELS:
                results[body_part][future_to_index[future]] = future.result()[body_part]

    output_file_path.touch()
    body_part_to_df = {}
    for body_part in DLC_LABELS:
        df = pd.DataFrame.from_records(results[body_part], columns=Ellipse._fields)
        body_part_to_df[body_part] = df
        df.to_hdf(output_file_path, key=body_part, mode='a')       
      
    return body_part_to_df



