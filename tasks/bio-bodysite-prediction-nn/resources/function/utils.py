from typing import Optional, Union

import numpy as np


def oneHotEncoding(vector: Union[np.ndarray, int], numClasses: Optional[int] = None) -> np.ndarray:

    """
        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Parameters
        ----------
        vector : ArrayLike
            A vector of integers
        numClasses : int
            Optionally declare the number of classes (can not exceed the maximum value of the vector)

        Returns
        -------
        result : np.ndarray
            The one hot encoded vecotr or matrix

        Example
        -------
        >>> v = np.array([1, 0, 4])
        >>> one_hot_v = oneHotEncoding(v)
        >>> print one_hot_v
        [[0 1 0 0 0]
        [1 0 0 0 0]
        [0 0 0 0 1]]
    """

    if isinstance(vector, int):
        vector = np.array([vector])

    vecLen = vector.shape[0]

    if numClasses is None:
        numClasses = vector.max() + 1

    result = np.zeros(shape = (vecLen, numClasses))
    result[np.arange(vecLen), vector] = 1
    return result.astype(int)


def convertFromOneHot(matrix: np.ndarray) -> np.ndarray:
    numOfRows = matrix.shape[0]
    if not numOfRows > 0:
        raise RuntimeError(f">> [MicrobiomeForensics] Encountered array with {numOfRows} rows when decoding one hot vector")

    result = np.empty(shape = (numOfRows, ), dtype = np.int32)
    for i, sample in enumerate(matrix):
        result[i] = sample.argmax()

    return result.astype(np.int32)
