from __future__ import annotations

import npc_session
import json
import datetime

import concurrent.futures
import contextlib
import copy
import functools
import pathlib
import pickle
from typing import (Dict, Iterable, Iterator, Literal, Mapping, NamedTuple,
                    Sequence, Tuple)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tqdm
from typing_extensions import TypeAlias

from aind_data_schema.core.data_description import (
    Organization,
    Modality,
    Modality,
    Platform,
    Funding,
    DataLevel,
)
from aind_data_schema_models.pid_names import PIDName

from aind_data_schema.core.quality_control import QualityControl, QCEvaluation, QCMetric, QCStatus, Status, Stage
from aind_data_schema_models.modalities import Modality
from aind_qcportal_schema.metric_value import CheckboxMetric

DATA_PATH = pathlib.Path('/data/')
RESULTS_PATH = pathlib.Path('/results/')
QC_PATH = RESULTS_PATH / "qc" 

DLC_PROJECT_PATH = DATA_PATH / 'universal_eye_tracking-peterl-2019-07-10'
DLC_SCORER_NAME = 'DLC_resnet50_universal_eye_trackingJul10shuffle1_1030000'

VIDEO_SUFFIXES = ('.mp4', '.avi', '.wmv', '.mov')

BodyPart: TypeAlias = Literal['cr', 'eye', 'pupil']
Annotation: TypeAlias = Literal['x', 'y', 'likelihood']
AnnotationData: TypeAlias = Dict[Tuple[BodyPart, Annotation], float]

DLC_LABELS: tuple[BodyPart, ...] = ('eye', 'pupil', 'cr') # order matters for ellipse fitting
ANNOTATION_PROPERTIES: tuple[Annotation, ...] = ('x', 'y', 'likelihood')

MIN_LIKELIHOOD_THRESHOLD = 0.2
MIN_NUM_POINTS_FOR_ELLIPSE_FITTING = 6
# at least 6 tracked points at 0.2 min likelihood for annotation quality data - from original peterl/waynew
NUM_NAN_FRAMES_EITHER_SIDE_OF_INVALID_EYE_FRAME = 2
"""number of frames to set to nan for pupil & cr on either side of an invalid eye ellipse 
- discards fits when the eyelid is closing or opening, which gives bad results
- see visual behavior whitepaper
https://portal.brain-map.org/explore/circuits/visual-behavior-2p
"""

VIDEO_FILE_GLOB_PATTERN = '*[eE]ye*'

def read_and_make_qc_figure(scale:int=10) -> None:
    image_paths = sorted(tuple(RESULTS_PATH.glob('qc/*.png')))
    if not image_paths:
        raise FileNotFoundError(f'No images found in {RESULTS_PATH}')
    
    fig, ax = plt.subplots(1, len(image_paths))

    index = 0
    for path in image_paths:
        image = np.array(Image.open(path))
        ax[index].figure.set_size_inches(scale, len(image_paths) * scale)
        ax[index].imshow(image)
        index += 1
    
    fig.savefig(RESULTS_PATH / 'dlc_eye_qc.png')

def get_number_of_frames_from_pickle(dlc_output_h5_path: str | pathlib.Path) -> int:
    meta_pickle = get_dlc_pickle_metadata(dlc_output_h5_path)
    return meta_pickle['nframes']

def is_failed_fits(dlc_output_h5_path: str | pathlib.Path) -> tuple[bool, list]:
    number_of_frames = get_number_of_frames_from_pickle(dlc_output_h5_path)
    percentage_of_failed_fits = []

    for feature in DLC_LABELS:
        feature_path = tuple(RESULTS_PATH.glob(f'qc/*/{feature}.json'))
        if not feature_path:
            raise FileNotFoundError(f'No ellipse data json found for {feature}')

        with open(feature_path[0]) as f:
            feature_json = json.load(f)
            percentage_of_failed_fits.append(len(feature_json['frames_without_ellipses']) / number_of_frames)

    return len([percentage for percentage in percentage_of_failed_fits if percentage > 0.1]) > 0, percentage_of_failed_fits

def write_qc_json(dlc_output_h5_path: str | pathlib.Path) -> None:
    qc_metric_video = QCMetric(name='Eye Tracking qc images', description='Evaluation of dlc eye tracking model', reference=f'/dlc_eye_qc.png',
                         value=CheckboxMetric(
                            value="Placeholder CheckboxMetric Value",
                            # Possible options for the metric
                            options=[
                                'No problems detected',
                                'Video too dim',
                                'Other Issues'
                            ],
                            status=[
                                Status.PASS,
                                Status.FAIL,
                                Status.FAIL,
                            ]
                         ),
                        status_history=[                                
                            QCStatus(
                                evaluator='',
                                timestamp=datetime.datetime.now(),
                                # Requires manual annotation
                                status=Status.PENDING
                            )
                        ]
                )
    
    is_failed, percentages = is_failed_fits(dlc_output_h5_path)
    qc_metric_frames = QCMetric(
        name='proportion of failed frames',
        description='Proportion of frames that failed to fit ellipse',
        value=percentages,
        reference=None,
        status_history=[
            QCStatus(
                evaluator='',
                timestamp=datetime.datetime.now(),
                status=Status.PASS if not is_failed else Status.FAIL
            )]
    )

    qc = QualityControl(
        notes='This is a dataset level quality control object for dlc eye tracking videos & their metadata',
        evaluations=[
            QCEvaluation(
                name="DLC Eye Tracking Video Quality",
                description="This evaluation ensures the quality of the eye tracking video",
                stage=Stage.PROCESSING,
                modality=Modality.from_abbreviation('behavior-videos'),
                notes="",
                allow_failed_metrics=False,
                metrics=[qc_metric_frames, qc_metric_video]
            )
        ],
    )
    qc.write_standard_file(RESULTS_PATH)

def get_data_description_dict() -> dict:
    session_id = parse_session_id()

    data_description_dict = {}
    data_description_dict["creation_time"] = datetime.datetime.now()
    data_description_dict["name"] = session_id
    data_description_dict["institution"] = Organization.AIND
    data_description_dict["data_level"] = DataLevel.DERIVED
    data_description_dict["investigators"] = [PIDName(name="Unknown")]
    data_description_dict["funding_source"] = [Funding(funder=Organization.AI)]
    data_description_dict["modality"] = [Modality.ECEPHYS]
    data_description_dict["platform"] = Platform.ECEPHYS
    data_description_dict["subject_id"] = str(npc_session.SessionRecord(session_id).subject)
    
    return data_description_dict

def get_processing_dict(start_date_time: datetime.datetime, end_date_time: datetime.datetime) -> dict:
    data_processing_dict = {}
    data_processing_dict["name"] = "Other"
    data_processing_dict["software_version"] = "0.1.0"
    data_processing_dict["start_date_time"] = str(start_date_time)
    data_processing_dict["end_date_time"] = str(end_date_time)
    data_processing_dict["input_location"] = DATA_PATH.as_posix()
    data_processing_dict["output_location"] = RESULTS_PATH.as_posix()
    data_processing_dict["code_url"] = "https://github.com/AllenNeuralDynamics/aind-capsule-eye-tracking"
    data_processing_dict["parameters"] = {
        "min_liklelihood_threshold": MIN_LIKELIHOOD_THRESHOLD,
        "min_num_points_for_ellipse_fitting": MIN_NUM_POINTS_FOR_ELLIPSE_FITTING,
        "num_nan_frames_either_side_of_invalid_eye_frame": NUM_NAN_FRAMES_EITHER_SIDE_OF_INVALID_EYE_FRAME
    }
    data_processing_dict["notes"] = "DeepLabCut Eye Tracking"
    data_processing_dict["outputs"] = {}

    return data_processing_dict

def parse_session_id() -> str:
    """
    parses the session id following the aind format, test assumes that raw data asset 686740_2023-10-26 is attached
    >>> parse_session_id()
    'ecephys_686740_2023-10-26_12-29-08'
    """
    session_paths = tuple(DATA_PATH.glob('*'))
    print(session_paths)
    if not session_paths or len(session_paths) == 1:
        raise FileNotFoundError('No session data assets attached')
    
    session_id = None
    for session_path in session_paths:
        try: # avoid parsing model folder, better way to do this?
            session_id = npc_session.parsing.extract_aind_session_id(session_path.stem)
        except ValueError:
            print('searching session_path for data description:', session_path)
            data_description_jsons = list(session_path.glob('data_description.json'))
            if len(data_description_jsons) == 0:
                raise Exception('no data description jsons found', data_description_jsons)
            elif len(data_description_jsons) == 1:
                data_description = open(data_description_jsons[0])
                session_id = json.load(data_description)['name']
                break
            else:
                raise Exception('multiple data description jsons found', data_description_jsons)

    if session_id is None:
        raise FileNotFoundError('No data asset attached that follows aind session format')
    
    return session_id

def write_json_with_session_id() -> None:
    """
    Test assumes raw data asset 686740_2023-10-26 is attached
    >>> write_json_with_session_id()
    >>> path = tuple(RESULTS_PATH.glob('*.json'))[0]
    >>> path.as_posix()
    '/results/ecephys_686740_2023-10-26_12-29-08.json'
    """
    session_id = parse_session_id()
    session_dict = {'session_id': session_id}
    with open(RESULTS_PATH / f'{session_id}.json', 'w') as f:
        json.dump(session_dict, f)

def get_video_paths() -> Iterator[pathlib.Path]:
    yield from (
        p for p in DATA_PATH.rglob(VIDEO_FILE_GLOB_PATTERN) 
        if (
            DLC_PROJECT_PATH not in p.parents
            and p.suffix in VIDEO_SUFFIXES
        )
    )

def get_dlc_pickle_metadata(dlc_output_h5_path: str | pathlib.Path) -> dict:
    h5 = pathlib.Path(dlc_output_h5_path)
    pkl = (
        h5
        .with_stem(f'{h5.stem}_meta')
        .with_suffix('.pickle')
    )
    return pickle.loads(pkl.read_bytes())['data']

def get_dlc_df(dlc_output_h5_path: str | pathlib.Path) -> pd.DataFrame:
    # df has MultiIndex 
    return getattr(pd.read_hdf(dlc_output_h5_path), get_dlc_pickle_metadata(dlc_output_h5_path)['Scorer']) 

def write_area_and_average_confidence(dlc_output_h5_path: pathlib.Path, body_part: BodyPart, body_part_to_df: pd.DataFrame) -> None:
    df_dlc = get_dlc_df(dlc_output_h5_path)
    get_keys = [[body_part + str(i), 'likelihood'] for i in range(1, 13)]
    avg_confidence = np.nanmean(df_dlc[list(get_keys)].values, axis = 1)
    body_part_to_df[f'{body_part}_average_confidence'] = avg_confidence
    body_part_to_df[f'{body_part}_area'] = get_pupil_area_pixels(body_part_to_df)
    body_part_to_df[f'{body_part}_is_bad_frame'] = pd.isna(body_part_to_df['center_x'])

    body_part_to_df.to_hdf(RESULTS_PATH / 'ellipses_processed.h5', key=body_part, mode='a')

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

def get_ellipses_from_row(row: AnnotationData, bounds: MinMax) -> dict[BodyPart, Ellipse]:
    ellipses = dict.fromkeys(DLC_LABELS, Ellipse())
    assert DLC_LABELS[0] == 'eye', f"expected `eye` to be first in {DLC_LABELS=}"
    for body_part in DLC_LABELS:
        arrays = {annotation: get_values_from_row(row, annotation, body_part) for annotation in ANNOTATION_PROPERTIES}
        in_bounds = np.array([
            is_in_min_max_xy(x, y, bounds)
            for x, y in zip(arrays["x"], arrays["y"])
        ])
        likely = arrays["likelihood"] > MIN_LIKELIHOOD_THRESHOLD
        if len(arrays["likelihood"][likely]) < MIN_NUM_POINTS_FOR_ELLIPSE_FITTING: 
            continue
        try:
            ellipse = fit_ellipse([arrays["x"][likely & in_bounds], arrays["y"][likely & in_bounds]])
        except InvalidEigenVectors:
            continue
        if (
            (body_part == 'eye' and is_ellipse_invalid(ellipse))
            or 
            (body_part != 'eye' and is_ellipse_invalid(ellipses['eye']))
        ):
            # if eye ellipse is invalid, all other ellipses are invalid
            break
        if body_part == 'eye' and not is_ellipse_in_min_max_xy(ellipse, bounds):
            # eye ellipse should not extend beyond bounds 
            ellipses['eye'] = Ellipse()
            break
        if body_part != 'eye':
            assert not is_ellipse_invalid(ellipses['eye']), f"expected valid eye ellipse: {ellipses['eye']=}"
            if not is_in_ellipse(ellipse.center_x, ellipse.center_y, ellipses['eye']):
                # cr or pupil centers outside eye ellipse are invalid
                continue            
        ellipses[body_part] = ellipse
    return ellipses

def get_ellipse_vertices(ellipse: Ellipse) -> tuple[tuple[float, float], ...]:
    """
    >>> get_ellipse_vertices(Ellipse(0, 0, 3, 5, 0))
    ((3.0, 0.0), (3.061616997868383e-16, 5.0), (-3.0, 0.0), (-3.061616997868383e-16, -5.0))
    """
    e = ellipse
    x1, y1 = e.width * np.cos(e.phi), e.width * np.sin(e.phi)
    x2, y2 = e.height * np.cos(e.phi + np.pi/2), e.height * np.sin(e.phi + np.pi/2)
    return tuple(
        (e.center_x + a, e.center_y + b)
        for a, b in (
            (x1, y1),
            (x2, y2),
            (-x1, -y1),
            (-x2, -y2),
        )
    )

def is_ellipse_in_min_max_xy(ellipse: Ellipse, min_max: MinMax) -> bool:
    for x, y in get_ellipse_vertices(ellipse):
        if not is_in_min_max_xy(x, y, min_max):
            return False
    return True

def run_ellipse_processing(
    dlc_output_h5_path: pathlib.Path, 
    output_file_path: pathlib.Path,
) -> dict[BodyPart, pd.DataFrame]:
    output_file_path = pathlib.Path(output_file_path).with_suffix('.h5')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        body_part_to_ellipses = get_ellipses_from_dlc_in_parallel(dlc_output_h5_path, executor)
    body_part_to_ellipses = get_filtered_ellipses(body_part_to_ellipses, dlc_output_h5_path)
    output_file_path.touch()
    body_part_to_df: dict[BodyPart, pd.DataFrame] = {}
    for body_part in DLC_LABELS:
        df = pd.DataFrame.from_records(body_part_to_ellipses[body_part], columns=Ellipse._fields)
        body_part_to_df[body_part] = df
        df.to_hdf(output_file_path, key=body_part, mode='a')       
      
    return body_part_to_df
                    
def get_ellipses_from_dlc_in_parallel(
    dlc_output_h5_path: pathlib.Path, 
    executor: concurrent.futures.Executor | None = None,
) -> dict[BodyPart, tuple[Ellipse, ...]]:

    dlc_df = get_dlc_df(dlc_output_h5_path)
    dlc_bounds = get_dlc_min_max_xy(dlc_output_h5_path)
    results: dict[BodyPart, list[Ellipse | None]] = {}
    future_to_index: dict[concurrent.futures.Future, int] = {}
    if executor is None:
        c = executor = concurrent.futures.ProcessPoolExecutor()
    else:
        c = contextlib.nullcontext()
    with c:
        for idx, row in tqdm.tqdm(
                dlc_df.iterrows(),             
                desc='submitting jobs',
                unit='frames',
                total=len(dlc_df),
                ncols=79,
                ascii=True, 
            ):
            future_to_index[
                executor.submit(get_ellipses_from_row, row.to_dict(), dlc_bounds)
            ] = idx 
        for future in tqdm.tqdm(
                concurrent.futures.as_completed(future_to_index.keys()), 
                desc='fitting ellipses',
                unit='frames',
                total=len(dlc_df),
                ncols=79,
                ascii=True, 
            ):
            for body_part in DLC_LABELS:
                result = future.result()[body_part] # any exceptions raised here
                results.setdefault(
                    body_part, 
                    [None] * len(dlc_df),
                )[future_to_index[future]] = result
                
    assert all(results[body_part].count(None) == 0 for body_part in DLC_LABELS)
    output: dict[BodyPart, tuple[Ellipse, ...]] = {
        body_part: tuple(e for e in ellipses if e is not None) # there are no None values - this is for mypy 
        for body_part, ellipses in results.items() 
    }
    assert all(len(output[body_part]) == len(dlc_df) for body_part in DLC_LABELS)
    return output

def is_ellipse_invalid(ellipse: Ellipse) -> bool:
    """
    >>> is_ellipse_invalid(Ellipse()) # default values represent invalid
    True
    """
    return all(np.isnan(ellipse))

def get_filtered_ellipses(
    body_part_to_ellipses: Mapping[BodyPart, Sequence[Ellipse]], 
    dlc_output_h5_path: pathlib.Path,
    ) -> dict[BodyPart, tuple[Ellipse, ...]]:
    """Apply physical constraints (e.g. pupil must be within eye perimeter) and
    good-practice filtering learned from experinece (e.g. bad pupil fits before/after blinks)"""
    num_frames = len(body_part_to_ellipses['eye'])
    dlc_min_max_xy = get_dlc_min_max_xy(dlc_output_h5_path)
    n = NUM_NAN_FRAMES_EITHER_SIDE_OF_INVALID_EYE_FRAME
    invalid = Ellipse()
    _body_part_to_ellipses: dict[BodyPart, list[Ellipse]] = {k: list(v) for k, v in body_part_to_ellipses.items()}
    for idx, eye in tqdm.tqdm(
            enumerate(_body_part_to_ellipses['eye']), 
            desc='filtering ellipses',
            unit='frames',
            total=num_frames,
            ncols=79,
            ascii=True, 
        ):
        
        # no ellipses should have centers outside of min/max range of dlc-analyzed
        # area of video
        for body_part in DLC_LABELS:
            if is_ellipse_invalid(e := _body_part_to_ellipses[body_part][idx]):
                continue
            if not is_in_min_max_xy(
                    e.center_x, 
                    e.center_y, 
                    dlc_min_max_xy,
            ):
                _body_part_to_ellipses[body_part][idx] = invalid
        
        # if eye ellipse is invalid, all other ellipses are invalid in adjacent frames
        if is_ellipse_invalid(eye):
            for invalid_idx in range(max(idx - n, 0), min(idx + n + 1, num_frames)):
                _body_part_to_ellipses['cr'][invalid_idx] = invalid
                _body_part_to_ellipses['pupil'][invalid_idx] = invalid
            continue

        
        if is_ellipse_invalid(eye):
            for body_part in DLC_LABELS[1:]:
                if is_ellipse_invalid(_body_part_to_ellipses[body_part][idx]):
                    continue
                if not is_in_ellipse(
                        _body_part_to_ellipses[body_part][idx].center_x, 
                        _body_part_to_ellipses[body_part][idx].center_y, 
                        eye,
                    ):
                    _body_part_to_ellipses[body_part][idx] = invalid
    return {k: tuple(v) for k, v in _body_part_to_ellipses.items()}


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

class MinMax(NamedTuple):
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    
def get_dlc_min_max_xy(dlc_output_h5_path: str | pathlib.Path) -> MinMax:
    with contextlib.suppress(FileNotFoundError):
        MinMax(*get_dlc_pickle_metadata(dlc_output_h5_path)['cropping_parameters'])
    df = get_dlc_df(dlc_output_h5_path)
    annotations = {i[0] for i in df}
    minmax = dict(MinMax(np.inf, -np.inf, np.inf, -np.inf)._asdict())
    for annotation in annotations:
        minmax['min_x'] = min(minmax['min_x'], df[annotation, 'x'].min())
        minmax['max_x'] = max(minmax['max_x'], df[annotation, 'x'].max())
        minmax['min_y'] = min(minmax['min_y'], df[annotation, 'y'].min())
        minmax['max_y'] = max(minmax['max_y'], df[annotation, 'y'].max())
    return MinMax(**minmax)

def is_in_min_max_xy(x: float, y: float, min_max: MinMax) -> bool:
    """Check if point is within range."""
    return (
        min_max.min_x <= x <= min_max.max_x
        and min_max.min_y <= y <= min_max.max_y
    )

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
