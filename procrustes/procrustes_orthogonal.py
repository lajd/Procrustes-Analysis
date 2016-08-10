__author__ = 'Jonny'

from procrustes.base import Procrustes
import numpy as np


class OrthogonalProcrustes(Procrustes):

    """
    This method deals with the orthogonal Procrustes problem
    """

    def __init__(self, array_a, array_b, translate=False, scale=False, hide_padding=True):
        self.hide_padding = hide_padding
        self.translate = translate
        self.scale = scale
        Procrustes.__init__(self, array_a, array_b, translate=self.translate, scale=self.scale,
                            hide_padding=self.hide_padding)

    def calculate(self):
        """
        Calculates the optimum orthogonal transformation array in the
        single-sided procrustes problem

        Parameters
        ----------
        array_a : ndarray
            A 2D array representing the array to be transformed (as close as possible to array_b)

        array_b : ndarray
            A 2D array representing the reference array

        Returns
        ----------
        u_optimum, array_transformed, error
        u_optimum = the optimum orthogonal transformation array satisfying the single
             sided procrustes problem
        array_ transformed = the transformed input array after transformation by u_optimum
        error = the error as described by the single-sided procrustes problem
        """
        array_a = self.array_a
        array_b = self.array_b

        # Calculate SVD
        product_matrix = np.dot(array_a.T, array_b)
        u, s, v_trans = self.singular_value_decomposition(product_matrix)

        # Define the optimum orthogonal transformation
        u_optimum = np.dot(u, v_trans)

        # Map array_a to array_b with the optimum orthogonal transformation
        self.map_a_b(u_optimum)

        # Compute the real error
        real_error = self.single_sided_procrustes_error(self.a_transformed, self.b_in)

        return u_optimum, self.a_transformed, real_error, self.transformation
