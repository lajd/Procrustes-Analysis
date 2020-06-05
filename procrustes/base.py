__author__ = 'Jonny'
# -*- coding: utf-8 -*-
"""Base class for Procrustes Analysis"""

import numpy as np
from procrustes.base_utils import centroid, translate_array_to_origin, hide_zero_padding_array, scale_array, zero_padding


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
            array_a = hide_zero_padding_array(array_a)
            array_b = hide_zero_padding_array(array_b)

        # Find the minimum row and column dimensions (removed of any zero padding) of the input arrays.
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
            if translate_symmetrically is False:
                pass
            if translate:
                # Allow for initial translation
                # In particular, this is for the two-sided orthogonal single transformation problem
                array_a = translate_array_to_origin(array_a, preserve_symmetry=preserve_symmetry)[0]
                array_a = scale_array(array_a)[0]
                array_b = translate_array_to_origin(array_b, preserve_symmetry=preserve_symmetry)[0]
                array_b = scale_array(array_b)[0]
                self.translate_scale = True
            else:
                # Don't allow for translation
                # In particular, this is for the two-sided permutation single transformation problem
                array_a, scale_a_to_b = scale_array(array_a, array_b, preserve_symmetry=preserve_symmetry)
                array_b = array_b
        elif scale:
            # For scaling in non symmetry-preserving problems to be meaningful,
            # we require translate to be true Scale the input arrays
            array_a = translate_array_to_origin(array_a)[0]
            array_a = scale_array(array_a)[0]
            array_b = translate_array_to_origin(array_b)[0]
            array_b = scale_array(array_b)[0]
            self.translate_scale = True
        elif translate:
            # Translate the initial arrays to the origin
            array_a = translate_array_to_origin(array_a)[0]
            array_b = translate_array_to_origin(array_b)[0]
        # At the end of the analysis, re-pad the arrays to allow for further operations.
        if preserve_symmetry:
            array_a, array_b = zero_padding(array_a, array_b, square=True)
        elif array_a.shape[0] != array_b.shape[0]:
            # print 'The general Procrustes analysis requires two 2D arrays with the same number of rows,',
            # print 'so the array with the smaller number of rows will automatically be padded with zero rows.'
            array_a, array_b = zero_padding(array_a, array_b, row=True, column=False)

        # Update the final arrays with all of the above modifications
        self.array_a = array_a
        self.array_b = array_b

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
            at, t1 = translate_array_to_origin(a)
            # Resize at to the minimal dimensions between original array_a and original array_b
            at = at[:self.min_row, :self.min_column]
            # Transform at by right-multiplication of transform1, the optimum transformation obtained
            # from one of the procrustes analyses
            at_t = np.dot(at, transform1)
            # Scale at_t to the same scaling as array_b (with minimal dimensions)
            at_ts, s = scale_array(at_t[:self.min_row, :self.min_column], self.b_in[:self.min_row, :self.min_column])
            # Obtain the last translation vector, which is array_b (with minimal dimensions) - at_ts
            t2 = centroid(self.b_in[:self.min_row, :self.min_column]) - centroid(at_ts)
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
                    #a, s = scale_array(self.a_in, self.b_in)
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
                    a1, t1 = translate_array_to_origin(self.a_in, preserve_symmetry=True)
                    assert(t1.shape == a1.shape)
                    a2, scale1 = scale_array(a1)
                    # print 't1, a2, t2 are {0}, {1} and {2}'.format(transform1.shape, a2.shape, transform2.shape)
                    a3 = np.dot(np.dot(transform1, a2), transform2)
                    b1, t = translate_array_to_origin(self.b_in, preserve_symmetry=True)
                    a4, s = scale_array(a3, b1)
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
                a1 = translate_array_to_origin(self.a_in)[0]
                a1 = scale_array(a1)[0]
                t1 = -centroid(self.a_in)
                t2 = centroid(self.b_in)
                a2 = np.dot(np.dot(transform1, a1), transform2)
                a3, s = scale_array(a2, self.b_in)
                a4 = a3 + t2
                self.a_transformed = a4
                self.transformation = {'first translation': t1,
                                       'left-sided transform': transform1,
                                       'right-sided transform': transform2,
                                       'scaling factor': s,
                                       'second translation': t2,
                                       'predicted coordinates': self.a_transformed,
                                       'expected coordinates': self.b_in}
