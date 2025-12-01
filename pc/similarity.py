import numpy as np
from similaritymeasures import frechet_dist


def similarity_measurements(*arrays, **kwargs) -> float:
    measurments = kwargs.pop("measurments", "procrustes")
    measurments_coefficients = kwargs.pop(
        "measurments_coefficients",
        [
            1.0,
        ],
    )

    if isinstance(measurments, str):
        if measurments == "procrustes":
            measurments = procrustes
        elif measurments == "frechet":
            measurments = frechet_dist

    elif isinstance(measurments, list):
        if "procrustes" in measurments:
            measurments.remove("procrustes")
            measurments.append(procrustes)
        if "frechet" in measurments:
            measurments.remove("frechet")
            measurments.append(frechet_dist)

    for func in measurments:
        if not callable(func):
            raise NotImplementedError(f"{type(func)} with {func} is not Implemented.")

    mtx1 = np.array(arrays[0], dtype=np.double, copy=True)
    mtx2 = np.array(arrays[1], dtype=np.double, copy=True)

    mtx1 -= np.mean(mtx1, 0)
    mtx1 /= np.linalg.norm(mtx1)

    mtx2 -= np.mean(mtx2, 0)
    mtx2 /= np.linalg.norm(mtx2)

    scores = []
    for func, coeff in zip(measurments, measurments_coefficients):
        if coeff != 0:
            _score_1 = func(mtx1, mtx2)
            _score_2 = func(mtx2, mtx1)
            scores.append(np.max([_score_1, _score_2]) * coeff)
        else:
            scores.append(0.0)

    return np.sum(scores)


def procrustes(mtx1: np.ndarray, mtx2: np.ndarray) -> np.ndarray:
    from scipy.linalg import orthogonal_procrustes

    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s
    disparity = np.sum(np.square(mtx1 - mtx2))
    return disparity
