__author__ = 'Jonny'
# -*- coding: utf-8 -*-
"""Base class for Procrustes Analysis"""

import numpy as np


class Procrustes(object):
    """
    This class provides a base class for all types of Procrustes analysis.
    """
    def __init__(self, array_a, array_b, translate=False, scale=False, preserve_symmetry=False, hide_padding=True,
                 translate_symmetrically=False):
        """
        Parameters
        ----------
        array_a : ndarray
            A 2D array
        array_b : ndarray
            A 2D array
        translate : bool
            Set to True to allow for translation of the input arrays; default=False
        scale : bool
            Set to True to allow for scaling of the input arrays; default=False

            Note: Scaling is meaningful only if preceded with translation. So, the code
            sets translate=True if scale=True and translate=False.
        """

        self.hide_padding = hide_padding
        self.translate = translate
        self.scale = scale
        self.preserve_symmetry = preserve_symmetry
        self.translate_symmetrically = translate_symmetrically

        # Initialize the transformation dictionary, which specifies the mapping from array a to the reference matrix,
        # array b. Set translation and scaling to False by default.
        self.transformation = None
        self.translate_scale = False

        # These arrays will ve altered dyring the course of the analysis
        self.array_a = array_a
        self.array_b = array_b

        # Remove any zero-padding which may be on the input arrays.
        if hide_padding is True:
            array_a = self.hide_zero_padding_array(array_a)
            array_b = self.hide_zero_padding_array(array_b)

        # Find the minimum row and column dimensions of thee (removed of any zero padding) input arrays.
        # This is important for operations on arrays of different dimensions, and will be used throughout
        # the following analysis.
        self.min_row = min(array_a.shape[0], array_b.shape[0])
        self.min_column = min(array_a.shape[1], array_b.shape[1])

        # Initialize the transformed array_a, and the original input arrays.
        self.a_transformed = array_a
        self.a_in = array_a
        self.b_in = array_b

        # Check type and dimension of arrays
        if not isinstance(array_a, np.ndarray) or not isinstance(array_b, np.ndarray):
            raise ValueError('The array_a and array_b should be of numpy.ndarray type.')

        if array_a.ndim != 2 or array_b.ndim != 2:
            raise ValueError('The array_a and array_b should be 2D arrays.')

        # Preserve symmetry is used for the procrustes problems in which translation is not well defined,
        # i.e. the double sided procrustes problems.
        # This is the case for the two-sided orthogonal and permutation single transformation problems.
        if preserve_symmetry:
            if (translate_symmetrically) is False:
                pass
            if translate:
                # Allow for initial translation
                # In particular, this is for the two-sided orthogonal single transformation problem
                array_a = self.translate_array(array_a, preserve_symmetry=preserve_symmetry)[0]
                array_a = self.scale_array(array_a)[0]
                array_b = self.translate_array(array_b, preserve_symmetry=preserve_symmetry)[0]
                array_b = self.scale_array(array_b)[0]
                self.translate_scale = True
            else:
                # Don't allow for translation
                # In particular, this is for the two-sided permutation single transformation problem
                array_a, scale_a_to_b = self.scale_array(array_a, array_b, preserve_symmetry=preserve_symmetry)
                array_b = array_b
        elif scale:
                # For scaling in non symmetry-preserving problems to be meaningful, we require translate to be true
                # Scale the input arrays
                array_a = self.translate_array(array_a)[0]
                array_a = self.scale_array(array_a)[0]
                array_b = self.translate_array(array_b)[0]
                array_b = self.scale_array(array_b)[0]
                self.translate_scale = True
        elif translate:
                # Translate the initial arrays to the origin
                array_a = self.translate_array(array_a)[0]
                array_b = self.translate_array(array_b)[0]
        # At the end of the analysis, re-pad the arrays to allow for further operations.
        if preserve_symmetry:
            array_a, array_b = self.zero_padding(array_a, array_b, square=True)
        elif array_a.shape[0] != array_b.shape[0]:
            # print 'The general Procrustes analysis requires two 2D arrays with the same number of rows,',
            # print 'so the array with the smaller number of rows will automatically be padded with zero rows.'
            array_a, array_b = self.zero_padding(array_a, array_b, row=True, column=False)

        # Update the final arrays with all of the above modifications
        self.array_a = array_a
        self.array_b = array_b

    def zero_padding(self, x1, x2, row=False, column=False, square=False):

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
            Set to True to match the number of rows by zero-padding; default=True.
        column : bool
            Set to True to match the number of columns by zero-padding; default=False.
        square: bool
            Set to True to zero pad the input arrays such that the inputs become square
            arrays of the same size
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
            x1, x2 = self.pad_rows(x1, x2)
        # Column padding
        if column:
            x1, x2 = self.pad_columns(x1, x2)
        # Square padding
        if square:
            dimension = max(max(x1.shape),  max(x2.shape))
            x1, x2 = self.pad_rows(x1, x2, set_rows=dimension)
            x1, x2 = self.pad_columns(x1, x2, set_columns=dimension)
        # Select at least one dimension for zero-padding
        if not (row or column or square):
            print 'You\'ve not selected any dimension(s) for zero-padding.'

        return x1, x2

    def pad_rows(self, x1, x2=None, set_rows=None):
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

    def pad_columns(self, x1, x2=None, set_columns=None):
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

    def translate_array(self, array_a, preserve_symmetry=False):

        """
        Parameters
        ----------
        array_a : ndarray
            A 2D array

        array_b : ndarray
            A 2D array

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
                translate_vector = -b
                # Find the diagonal value (of each column) which will bring the centroid of the array to the origin
                for i in range(n):
                    temp_sum = c[:, i].sum()
                    insert_val = -temp_sum
                    c[i, i] = insert_val
                    translate_vector[i, i] += insert_val
                # Here, c is the origin-centred array. The translation-vector is also saved.
                return c, translate_vector

        else:
                # Calculate array_a's centroid, i.e. the translation vector from the origin to array_a's centroid
                centroid = array_a.mean(0)
                # Translate array_a's centroid to the origin. '-centroid' is the translation vector bringing
                # array_a's centroid to the origin
                origin_centred_array = array_a - centroid
                return origin_centred_array, -centroid

    def scale_array(self, array_a, array_b=None, preserve_symmetry=False):
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
                at, ta = self.translate_array(array_a)
                bt, tb = self.translate_array(array_b)
            # Calculate Frobenius norm of array a
            fna = self.frobenius_norm(at)
            # Calculate Frobenius norm of array b
            fnb = self.frobenius_norm(bt)
            # Bring the scale of array a to the scale of array b's
            scale_a_to_b = fnb / fna
            # Compute the rescaled array a
            array_a_rescaled = scale_a_to_b * array_a
            return array_a_rescaled, scale_a_to_b

        # If array_b is not supplied...
        else:
            # Calculate Frobenius norm
            at, ta = self.translate_array(array_a)
            fn = self.frobenius_norm(at)
            # Scale array to lie on unit sphere
            array = array_a / fn
            scale_a = 1./fn
            return array, scale_a

    def map_a_b(self, transform1, transform2=None, preserve_symmetry=False, translate_symmetrically=True):

        """
        This function is called by each procrustes subclass. It is only meaningful after
        the optimum transformation(s) have been solved for (i.e. by an individual
        procrustes subclass).

        Parameters
        ----------
        transform1 : ndarray
            A 2D array which corresponds the (right-multiplied) optimum
            transformation bringing array_a to array_b in any of the procrustes problems.

        transform2 : ndarray
            A 2D array which corresponds the (left-multiplied) optimum
            transformation bringing array_a to array_b in any of the procrustes problems.

        Returns
        ----------
        if transform2 is None:
            Updates self.a_transformed, the coordinates of array_a after the optimum
            transformations have been applied.

            Updates self.transformation, a dictionary specifying (one possible) set of
            transformations which optimally maps array_a onto array_b.

            This corresponds case to single-sided procrustes problems, where the optimum
            transformation is always right-multiplied to array_a.

        if array_b is not None :
            Updates self.a_transformed, the coordinates of array_a after the optimum
            transformations have been applied.

            Updates self.transformation, a dictionary specifying (one possible) set of
            transformations which optimally maps array_a onto array_b.

            This corresponds case to double-sided procrustes problems, where the optimum
            transformations transform1 and transform2 are left and right-multiplied to
            array_a, respectively.
            """
        t1 = None
        t2 = None
        s = None


        # If transform2 is not supplied...
        #  This is used for computing what's identified as the 'real' error.
        if transform2 is None:  # For all cases where we are computing the real errors
            # The (original) input array_a, before any analysis or preparation.
            a = self.a_in
            # The original array_a, with centroid translated the to origin via t1
            at, t1 = self.translate_array(a)
            # Resize at to the minimal dimensions between original array_a and original array_b
            at = at[:self.min_row, :self.min_column]
            # Transform at by right-multiplication of transform1, the optimum transformation obtained
            # from one of the procrustes analyses
            at_t = np.dot(at, transform1)
            # Scale at_t to the same scaling as array_b (with minimal dimensions)
            at_ts, s = self.scale_array(at_t[:self.min_row, :self.min_column], self.b_in[:self.min_row, :self.min_column])
            # Obtain the last translation vector, which is array_b (with minimal dimensions) - at_ts
            t2 = self.centroid(self.b_in[:self.min_row, :self.min_column]) - self.centroid(at_ts)
            # Update self.a_transformed, the new coordinates of the optimally transformed array_a
            # Update self.transformation, a dictionary specifying the optimal transformations to array_a (in order).
            self.a_transformed = at_ts + t2
            self.transformation = {'first translate': t1,
                                   'right-sided transformation': transform1,
                                   'scaling factor': s,
                                   'second translate': t2,
                                   'predicted coordinates': self.a_transformed,
                                   'expected coordinates': self.b_in}

        # In all other cases, transform2 is non-None
        else:
            # Transform2 is necessarily True for all preserve_symmetry problems
            if preserve_symmetry:
                # The two sided permutation single transformation problem doesn't currently allow for translation
                if translate_symmetrically is False:
                    #a, s = self.scale_array(self.a_in, self.b_in)
                    self.a_transformed = np.dot(np.dot(transform1, self.array_a), transform2)
                    self.transformation = {'first translation': t1,
                                           'left-sided transform': transform1,
                                           'right-sided transform': transform2,
                                           'scaling factor': s,
                                           'second translation': t2,
                                           'predicted coordinates': self.a_transformed,
                                           'expected coordinates': self.b_in}
                # The two sided orthogonal single transformation problem does allow for translation
                else:
                    a1, t1 = self.translate_array(self.a_in, preserve_symmetry=True)
                    assert(t1.shape == a1.shape)
                    a2, scale1 = self.scale_array(a1)
                    # print 't1, a2, t2 are {0}, {1} and {2}'.format(transform1.shape, a2.shape, transform2.shape)
                    a3 = np.dot(np.dot(transform1, a2), transform2)
                    b1, t = self.translate_array(self.b_in, preserve_symmetry=True)
                    a4, s = self.scale_array(a3, b1)
                    assert transform1.shape[1] == self.a_in.shape[0]
                    assert transform2.shape[0] == self.a_in.shape[1]
                    assert t.shape == np.dot(np.dot(transform1, (self.a_in + t1) * scale1), transform2).shape
                    self.a_transformed = np.dot(np.dot(transform1, (self.a_in + t1) * scale1), transform2) * s - t
                    self.transformation = {'first translation': t1,
                                           'left-sided transform': transform1,
                                           'right-sided transform': transform2,
                                           'scaling factor': scale1*s,
                                           'second translation': -t,
                                           'predicted coordinates': np.dot(np.dot(transform1,
                                                                    (self.a_in + t1) * scale1), transform2) * s - t,
                                           'expected coordinates': self.b_in}

            # When preserve symmetry is false (for all single-sided problems)...
            elif self.translate_scale is False:
                a = self.array_a
                a1 = np.dot(np.dot(transform1, a), transform2)
                self.a_transformed = a1
                self.transformation = {'left-sided transform': transform1,
                                       'right-sided transform': transform2,
                                       'predicted coordinates': self.a_transformed,
                                       'expected coordinates': self.b_in}

            else:   # Scaling and translation
                a1 = self.translate_array(self.a_in)[0]
                a1 = self.scale_array(a1)[0]
                t1 = -self.centroid(self.a_in)
                t2 = self.centroid(self.b_in)
                a2 = np.dot(np.dot(transform1, a1), transform2)
                a3, s = self.scale_array(a2, self.b_in)
                a4 = a3 + t2
                self.a_transformed = a4
                self.transformation = {'first translation': t1,
                                       'left-sided transform': transform1,
                                       'right-sided transform': transform2,
                                       'scaling factor': s,
                                       'second translation': t2,
                                       'predicted coordinates': self.a_transformed,
                                       'expected coordinates': self.b_in}

    def single_sided_procrustes_error(self, array_a, array_b, transform1=None):

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

        t_array: ndarray
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
        array_a, array_b = self.zero_padding(array_a, array_b, row=True, column=True)

        # If no transformation is supplied...
        if transform1 is None:
            at = array_a
        # If a transformation is supplied...
        else:
            # Minimum dimensions of array_a, apply transformation...
            at = np.dot(array_a, transform1)

        # Error as specified in the description, with minimal dimensions of array_b
        error = np.trace(np.dot((at-array_b).T, at-array_b))
        return error

    def double_sided_procrustes_error(self, array_a, array_b, transform1=None, transform2=None):

        """
        Returns the error for all double-sided procrustes problems

        min { ([t_array1].T*[array_a]*[t_array2] - [array_b]).T) * ([t_array1].T*[array_a]*[t_array2] - [array_b])) }
            : t_array1, t_array2 satisfies some condition.

        Parameters
        ----------

        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b).

        array_b : ndarray
            A 2D array representing the reference array.

        t_array: ndarray
           A 2D array representing the 1st 'optimum' transformation
           in the two-sided procrustes problems.
        t_array2: ndarray
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
        array_a, array_b = self.zero_padding(array_a, array_b, row=True, column=True)

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

    def singular_value_decomposition(self, array):

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

    def eigenvalue_decomposition(self, array):

        """
        Computes the eigenvalue decomposition of array
        Decomposes array A such that A = U*S*U.T

        Parameters
        ------------
        array: ndarray
           A 2D array who's eigenvalue decomposition is to be calculated.

        two_sided_single : bool
            Set to True when dealing with two-sided single transformation procrustes problems,
            such as two_sided single transformation orthogonal / permutation. When true, array of
            eigenvectors is rearranged according to rows rather than columns, allowing the analysis
            to proceed.

        Returns
        ------------
        s = 1D array of the eigenvalues of array, sorted from greatest to least.
        v = 2D array of eigenvectors of array, sorted according to S.
        """
        # Test whether eigenvalue decomposition is possible on the input array
        if self.is_diagonalizable(array) is False:
            raise ValueError('The input array is not diagonalizable. The analysis cannot continue.')

        # Find eigenvalues and eigenvectors
        s, v = np.linalg.eigh(array)
        # Sort the eigenvalues from greatest to least
        idx = s.argsort()[::-1]
        s = s[idx]
        v = v[:, idx]
        return s, v

    def hide_zero_padding_array(self, array, tol=1.e-8):

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

    def is_diagonalizable(self, array):

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

    def centroid(self, array):
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

    def frobenius_norm(self, array):
        """
        Parameters
        -----------------------------
        array: 2d array

        Returns
        ------------------------------
        Returns the Frobenius norm of the array
        """
        return np.sqrt((array ** 2.).sum())

    def array_normalization(self, array):
        """
        Parameters
        -----------------------------
        array: a numpy array

        Returns
        ------------------------------
        Returns (column) normalized original array. If the array is just a vector, returns the normalized vector.
        """














