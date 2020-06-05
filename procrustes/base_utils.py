import numpy as np


def hide_zero_padding_array(array: np.array, tol=1.e-8):
    """
    Removes any zero-padding that may be on the array.

    Parameters
    -------------------
    array: An array that may or may not contain zero-padding, where all important
       information is contained in upper-left block of array.

    tol: Tolerance for which is sum(row/column) < tol, then the row/col will be removed.

    Returns
    --------------------
    Returns the input array with any zero-padding removed.
    All zero padding is assumed to be such that all relevant information is contained
    within upper-left array block
    """

    m, n = array.shape
    # Start from the last row, and work up...
    for i in range(m)[::-1]:
        # If the sum of the row is very small... remove the row
        if sum(np.absolute(array[i, :])) < tol:
            # Update the array
            array = np.delete(array, i, 0)
        else:
            # Assume zero padding is only on exterior
            break
    # Start from the last column, and work up...
    for j in range(n)[::-1]:
        # If the sum of the column is very small... remove the column
        if sum(np.absolute(array[:, j])) < tol:
            # Update the array
            array = np.delete(array, j, 1)
        else:
            # Assume zero padding is only on exterior
            break
    return array


def is_diagonalizable(array: np.array):
    """
    Parameters
    -----------------------------
    array: A square 2d array

    Returns
    ------------------------------
    Returns a boolean value dictating whether or not the input array is diagonalizable
    """

    m, n = array.shape
    if m != n:
        raise ValueError('The input array must be square to be diagonalizable.')
    # Compute the singular value decomposition of the array
    u, s, vt = np.linalg.svd(array)
    # Get the rank of the eigenspace
    rank_u = np.linalg.matrix_rank(u)
    # Get the rank of the input array
    rank_array = np.linalg.matrix_rank(array)

    # The rank of the eigenspace must equal the rank of the array for diagonalizability
    if rank_u != rank_array:
        # The eigenvectors cannot span the dimension of the vector space
        # The array cannot be diagonalizable
        return False
    else:
        # The eigenvectors span the dimension of the vector space and therefore the array is diagonalizable
        return True


def centroid(array: np.array):
    """
    Parameters
    -----------------------------
    array: 2d array

    Returns
    ------------------------------
    Returns the coordinates of the centroid of the array, where columns are taken to be the dimensions
    and rows the points of the object.
    """

    centroid = array.mean(0)
    return centroid


def frobenius_norm(array: np.array):
    """
    Parameters
    -----------------------------
    array: 2d array

    Returns
    ------------------------------
    Returns the Frobenius norm of the array
    """
    return np.sqrt((array ** 2.).sum())


def pad_rows(x1, x2=None, set_rows=None):
    """
    Parameters
    ----------
    x1 : ndarray
        A 2D array
    x2 : ndarray
        A 2D array
    set_rows: int
        An integer value

    Returns
    -------
    If x2 is not None and set_rows is None:
        Returns the input arrays, x1 and x2, where the array with the fewer number of rows
        is row-padded such that the output arrays have the same number of rows.

    If x2 is not None and set_rows is not None:
        Returns the input arrays, x1 and x2, where both arrays are row_padded such that the
        output arrays both have set_rows number of rows. The value of set_rows must be greater
        than or equal to the number of rows of the input array with the most rows.

    If x2 is None and set_rows is not None:

         Returns the row-padded input array, x1, where the number of rows of the output array
         is equal to set_rows. The value of set_rows must be greater than or equal to the number
         of rows of the input x1.
    """

    # x2 and set_rows cannot both be none. Raise an error.
    if x2 is None and set_rows is None:
        raise ValueError('Both x2 and set_rows cannot be None. Please adjust the parameters.')

    # Row-padding the input array with fewer rows
    if set_rows is None:
        m1, n1 = x1.shape
        m2, n2 = x2.shape
        # If the row numbers are already equal...
        if m1 == m2:
            return x1, x2
        # If x1 has fewer rows than x2...
        elif m1 < m2:
            # padding x1 with zero rows
            pad = np.zeros((m2 - m1, n1))
            x1 = np.concatenate((x1, pad), axis=0)
            return x1, x2
        # If x2 has fewer rows than x1...
        elif m2 < m1:
            # padding x2 with zero rows
            pad = np.zeros((m1 - m2, n2))
            x2 = np.concatenate((x2, pad), axis=0)
            return x1, x2
    # Row-padding one (or both) arrays so final number of rows equal set_rows
    else:
        m1, n1 = x1.shape
        # If x1 has more rows than set_rows... Raise an error
        if m1 > set_rows:
            raise ValueError(' x1 has more rows than set_rows. Padding cannot continue.')
        # Row-pad if x1 has fewer rows than set_rows
        elif m1 < set_rows:
                # padding x1 with zero rows
                pad = np.zeros((set_rows - m1, n1))
                x1 = np.concatenate((x1, pad), axis=0)
        # Pad x2 if x2 is supplied
        if x2 is not None:
            m2, n2 = x2.shape
            # If x2 has more rows than set_rows... Raise an error
            if m2 > set_rows:
                raise ValueError(' x2 has more rows than set_rows. Padding cannot continue.')
            # Row-pad if x2 has fewer rows than set_rows
            elif m2 < set_rows:
                # padding x2 with zero rows
                pad = np.zeros((set_rows - m2, n2))
                x2 = np.concatenate((x2, pad), axis=0)
                return x1, x2
            else:
                return x1, x2
        else:
            return x1


def pad_columns(x1, x2=None, set_columns=None):
    """
    Parameters
    ----------
    x1 : ndarray
        A 2D array
    x2 : ndarray
        A 2D array
    set_columns: int
        An integer value

    Returns
    -------
    If x2 is not None and set_columns is None:
        Returns the input arrays, x1 and x2, where the array with the fewer number of columns
        is column-padded such that the output arrays have the same number of columns.

    If x2 is not None and set_columns is not None:
        Returns the input arrays, x1 and x2, where both arrays are column_padded such that the
        output arrays both have set_columns number of columns. The value of set_columns must be
        greater than or equal to the number of columns of the input array with the most columns.

    If x2 is None and set_columns is not None:

         Returns the column-padded input array, x1, where the number of columns of the output array
         is equal to set_columns. The value of set_columns must be greater than or equal to the number
         of columns of the input x1.
    """

    # x2 and set_columns cannot both be none. Raise an error.
    if x2 is None and set_columns is None:
        raise ValueError('Both x2 and set_columns cannot be None. Please adjust the parameters.')

    # Column-padding the input array with fewer rows
    if set_columns is None:
        m1, n1 = x1.shape
        m2, n2 = x2.shape
        # If the column numbers are already equal...
        if n1 == n2:
            return x1, x2
        # If x1 has fewer columns than x2...
        elif n1 < n2:
            # padding x1 with zero columns
            pad = np.zeros((m1, n2 - n1))
            x1 = np.concatenate((x1, pad), axis=1)
            return x1, x2
        # If x2 has fewer columns than x1...
        elif n2 < n1:
            # padding x2 with zero columns
            pad = np.zeros((m2, n1 - n2))
            x2 = np.concatenate((x2, pad), axis=1)
            return x1, x2
    # Column-padding one (or both) arrays so final number of columns equal set_columns
    else:
        m1, n1 = x1.shape
        # If x1 has more columns than set_columns... Raise an error
        if n1 > set_columns:
                raise ValueError(' x1 has more rows than set_rows. Padding cannot continue.')
        # Row-pad if x1 has fewer columns than set_columns
        elif n1 < set_columns:
            # padding x1 with zero rows
            pad = np.zeros((m1, set_columns - n1))
            x1 = np.concatenate((x1, pad), axis=1)
        # Pad x2 if x2 is supplied
        if x2 is not None:
            m2, n2 = x2.shape
            # If x2 has more columns than set_columns... Raise an error
            if n2 > set_columns:
                raise ValueError('x2 has equal (or more) rows than set_rows. Padding cannot continue.')
            # column-pad if x2 has fewer columns than set_columns
            elif n2 < set_columns:
                # padding x2 with zero columns
                pad = np.zeros((m2, set_columns - n2))
                x2 = np.concatenate((x2, pad), axis=1)
                return x1, x2
            else:
                return x1, x2
        else:
            return x1


def translate_array_to_origin(array_a: np.array, preserve_symmetry: bool = False):

    """
    Parameters
    ----------
    array_a : ndarray
        A 2D array
    preserve_symmetry : bool

    Returns
    ----------
    When preserve_symmetry is False:
        Returns the origin-centred array_a and the translation vector which brings array_a's centroid
        to the origin.

    When preserve_symmetry is True:
        Returns a heuristic for origin-centering a symmetric array, as well as the translation vector bringing
        the centroid of array_a to the origin.
    #  """

    # If array_b is not supplied...
    if preserve_symmetry:
        # translates a symmetric matrix to the origin, preserving symmetry
        # Get the diagonal components of array a
        a = np.diag(array_a)
        # Create a diagonal matrix of the diagonal elements of a
        b = np.diag(a)
        # Remove the diagonal components of array b, making all diagonals of b equal to zero
        c = array_a - b
        m, n = c.shape
        # translate vector maps the symmetric array to the origin
        # Initiate the translate vector, as -b
        translation_vector = -b
        # Find the diagonal value (of each column) which will bring the centroid of the array to the origin
        for i in range(n):
            temp_sum = c[:, i].sum()
            insert_val = -temp_sum
            c[i, i] = insert_val
            translation_vector[i, i] += insert_val
        # Here, c is the origin-centred array. The translation-vector is also saved.
        origin_centred_array = c
    else:
        # Calculate array_a's centroid, i.e. the translation vector from the origin to array_a's centroid
        centroid_ = array_a.mean(0)
        # Translate array_a's centroid to the origin. '-centroid' is the translation vector bringing
        # array_a's centroid to the origin
        origin_centred_array = array_a - centroid_
        translation_vector = -centroid_
    return origin_centred_array, translation_vector


def singular_value_decomposition(array: np.array):

    """
    Singular Value Decomposition of an array
    Decomposes an mxn array A such that A = U*S*V.T

    Parameters
    -----------

    array: ndarray
    A 2D array who's singular value decomposition is to be calculated.

    Returns
    --------------
    u = a unitary matrix.
    s = diagonal matrix of singular values, sorting from greatest to least.
    v = a unitary matrix.
    """
    u, s, vt = np.linalg.svd(array)
    return u, s, vt


def eigenvalue_decomposition(array: np.array):

    """
    Computes the eigenvalue decomposition of array
    Decomposes array A such that A = U*S*U.T

    Parameters
    ------------
    array: ndarray
       A 2D array who's eigenvalue decomposition is to be calculated.

    Returns
    ------------
    s = 1D array of the eigenvalues of array, sorted from greatest to least.
    v = 2D array of eigenvectors of array, sorted according to S.
    """
    # Test whether eigenvalue decomposition is possible on the input array
    if is_diagonalizable(array) is False:
        raise ValueError('The input array is not diagonalizable. The analysis cannot continue.')

    # Find eigenvalues and eigenvectors
    s, v = np.linalg.eigh(array)
    # Sort the eigenvalues from greatest to least
    idx = s.argsort()[::-1]
    s = s[idx]
    v = v[:, idx]
    return s, v


def zero_padding(x1: np.array, x2: np.array, row: bool = False, column: bool = False, square: bool = False):
    """
    Match the number of rows and/or columns of arrays x1 and x2 by
    padding zero rows and/or columns to the array with the smaller dimensions.

    Parameters
    ----------
    x1 : ndarray
        A 2D array
    x2 : ndarray
        A 2D array
    row : bool
        Set to True to match the number of rows by zero-padding; default=False.
    column : bool
        Set to True to match the number of columns by zero-padding; default=False.
    square: bool
        Set to True to zero pad the input arrays such that the inputs become square
        arrays of the same size; default=False
    Returns
    -------
    If row = True and Column = False:

         Returns the input arrays, x1 and x2, where the array with the fewer number
         of rows has been padded with zeros to match the number of rows of the other array

    if row = False and column = True

         Returns the input arrays, x1 and x2, where the array with the fewer number
         of columns has been padded with zeros to match the number of columns of the other array

    if row = True and column = True

         Returns the input arrays, x1 and x2, where the array with the fewer rows has been row-padded
         with zeros, and the array with the fewer number of columns has been column-padded with zeros
         in order to match the row/column number of the array with the greatest number of rows/columns.
         The outputs have the same size, and need not be square.

    if square = True
         Returns the input arrays x1 and x2 zero padded such that both arrays are square and of the same size.
    """

    # Confirm the input arrays are 2d numpy arrays
    assert isinstance(x1, np.ndarray) and isinstance(x2, np.ndarray)
    assert x1.ndim == 2 and x2.ndim == 2

    # Row padding
    if row:
        x1, x2 = pad_rows(x1, x2)
    # Column padding
    if column:
        x1, x2 = pad_columns(x1, x2)
    # Square padding
    if square:
        dimension = max(max(x1.shape),  max(x2.shape))
        x1, x2 = pad_rows(x1, x2, set_rows=dimension)
        x1, x2 = pad_columns(x1, x2, set_columns=dimension)
    # Select at least one dimension for zero-padding
    if not (row or column or square):
        print('You\'ve not selected any dimension(s) for zero-padding.')

    return x1, x2


def scale_array(array_a, array_b=None, preserve_symmetry=False):
    """
    Note: scaling is always preceded by translation to the origin.

    Parameters
    ----------
    array_a : ndarray
        A 2D array

    array_b : ndarray
        A 2D array

    Returns
    ----------
    if array_b is None:
        Returns the Frobeniusly normalized array_a and the corresponding scaling factor, s = 1 / Frobenius_norm_a.

    if array_b is not None :
        Returns the optimal scaling factor which brings array_a to the same Frobenius norm of array_b, aswell as
        the rescaled array_a.
     """
    if array_b is not None:
        if preserve_symmetry:
            # When scaling symmetric arrays, we cannot translate them first
            at = array_a
            bt = array_b
        else:
            at, ta = translate_array_to_origin(array_a)
            bt, tb = translate_array_to_origin(array_b)
        # Calculate Frobenius norm of array a
        fna = frobenius_norm(at)
        # Calculate Frobenius norm of array b
        fnb = frobenius_norm(bt)
        # Bring the scale of array a to the scale of array b's
        scale_a_to_b = fnb / fna
        # Compute the rescaled array a
        array_a_rescaled = scale_a_to_b * array_a
        return array_a_rescaled, scale_a_to_b

    # If array_b is not supplied...
    else:
        # Calculate Frobenius norm
        at, ta = translate_array_to_origin(array_a)
        fn = frobenius_norm(at)
        # Scale array to lie on unit sphere
        array = array_a / fn
        scale_a = 1./fn
        return array, scale_a


def single_sided_procrustes_error(array_a, array_b, transform=None):

    """
    Returns the error for all single-sided procrustes problems.

    min { (([array_a]*[t_array] - [array_b]).T)*([array_a]*[t_array] - [array_b]) }

        : transform1 satisfies constraints specified by the procrustes analysis.

    Parameters
    ----------

    array_a : ndarray
        A 2D array representing the array to be transformed (as close as possible to array_b).

    array_b : ndarray
        A 2D array representing the reference array.

    transform: ndarray
       A 2D array representing the 'optimum' transformation.

    Returns
    ------------
    If transform1 is None:
        Returns the error of matching array_a to array_b.

    If transform1 is not None:
        Returns the error of matching array_a transformed to array_b, where array_a transformed is
        the array resulting from right-multiplication of tranform1 to array_a.

    """
    # zero-pad self.a_transformed and array_b
    array_a, array_b = zero_padding(array_a, array_b, row=True, column=True)

    # If no transformation is supplied...
    if transform is None:
        at = array_a
    # If a transformation is supplied...
    else:
        # Minimum dimensions of array_a, apply transformation...
        at = np.dot(array_a, transform)

    # Error as specified in the description, with minimal dimensions of array_b
    error = np.trace(np.dot((at-array_b).T, at-array_b))
    return error


def double_sided_procrustes_error(array_a, array_b, transform1=None, transform2=None):

    """
    Returns the error for all double-sided procrustes problems

    min { ([transform1].T*[array_a]*[transform2] - [array_b]).T) * ([transform1].T*[array_a]*[transform2] - [array_b])) }
        : transform1, transform2 satisfies some condition.

    Parameters
    ----------

    array_a : ndarray
        A 2D array representing the array to be transformed (as close as possible to array_b).

    array_b : ndarray
        A 2D array representing the reference array.

    transform1: ndarray
       A 2D array representing the 1st 'optimum' transformation
       in the two-sided procrustes problems.
    transform2: ndarray
       A 2D array representing the 2nd 'optimum' transformation
       in the two-sided procrustes problems.

    Returns
    ------------
    If transform1 and transform2 are both None:
        Returns the error of matching array_a to array_b.

    If transform1 and transform2 are both not None:
        Returns the error of matching array_a transformed to array_b, where array_a transformed is
        the array resulting from left and right-multiplication of tranform1 and transform2, respectively
        to array_a.
    """
    # If one of transform1/transform2 is None and the other is not None, the analysis cannot proceed...
    if (transform1 is None and transform2 is not None) or (transform2 is None and transform1 is not None):
        raise ValueError('Either both transformations are None, or both must be supplied. Please '
                         'adjust the parameters.')

    # zero-pad self.a_transformed and array_b
    array_a, array_b = zero_padding(array_a, array_b, row=True, column=True)

    # If the transformations are not supplied...
    if transform1 is None and transform2 is None:
        t1_t_at2 = array_a

    # If the transformations are supplied...
    else:
        # Minimum dimensions of array_a, apply transformations...
        t1_t_at2 = np.dot(np.dot(transform1.T, array_a), transform2)
    # Error as specified in the description, with minimal dimensions of array_b
    error = np.trace(np.dot((t1_t_at2 - array_b).T,
                            (t1_t_at2-array_b)))
    return error

