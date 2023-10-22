from __future__ import annotations

import pathlib
from typing import Iterator, NamedTuple
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
import functools

import numpy as np
import pandas as pd

DATA_PATH = pathlib.Path('/root/capsule/data/')
RESULTS_PATH = pathlib.Path('/root/capsule/results/')

DLC_PROJECT_PATH = DATA_PATH / 'universal_eye_tracking-peterl-2019-07-10'
DLC_SCORER_NAME = 'DLC_resnet50_universal_eye_trackingJul10shuffle1_1030000'

DLC_LABELS = ('cr', 'eye', 'pupil')

VIDEO_SUFFIXES = ('.mp4', '.avi', '.wmv', '.mov')


def get_eye_video_paths() -> Iterator[pathlib.Path]:
    yield from (
        p for p in DATA_PATH.rglob('*[eE]ye*') 
        if (
            DLC_PROJECT_PATH not in p.parents
            and p.suffix in VIDEO_SUFFIXES
        )
    )
  
            
def get_dlc_output_path(
    input_video_file_path: str | pathlib.Path, 
    dlc_destfolder: str | pathlib.Path = RESULTS_PATH,
):
    return next(
        pathlib.Path(dlc_destfolder)
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

#ellipse fitting method
    
def get_ellipses_from_row(row) -> dict[Literal['cr', 'eye', 'pupil'], Ellipse]:

    l_threshold = 0.2
    min_num_points = 6

    # uses https://github.com/bdhammel/least-squares-ellipse-fitting
    # based on the publication Halir, R., Flusser, J.: 'Numerically Stable Direct Least Squares Fitting of Ellipses'

    out = dict()
    for label in DLC_LABELS:

        x_data = row.filter(regex=(f"{label}*")).values[0::3]
        y_data = row.filter(regex=(f"{label}*")).values[1::3]
        l = row.filter(regex=(f"{label}*")).values[2::3]
        
        ellipse = Ellipse() # default nan values 
        if len(l[l>l_threshold]) >= min_num_points: #at least 6 tracked points for annotation quality data
            try:
                ellipse = fit_ellipse([x_data[l>l_threshold], y_data[l>l_threshold]])
            except Exception as e:
                print(e) 
        out[label] = ellipse
    return out


def process_ellipses(h5name, out_path):

    # df has MultiIndex 
    # TODO extract label from df
    df = getattr(pd.read_hdf(h5name), DLC_SCORER_NAME) 

    results = dict.fromkeys(DLC_LABELS, value=list())
    with ProcessPoolExecutor() as executor:
        for idx, ellipses in enumerate(executor.map(get_ellipses_from_row, functools.partial(df.itertuples, index=False))):
            for label in DLC_LABELS:
                results[label][idx] = ellipses[label]

    ellipse_file_path = out_path 
    for label in DLC_LABELS:
        pd.DataFrame(results[label]).to_hdf(ellipse_file_path, key=label, mode='w')       
      
    return results



