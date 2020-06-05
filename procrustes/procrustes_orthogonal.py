__author__ = 'Jonny'

from procrustes.base import Procrustes
import numpy as np
from procrustes.base_utils import singular_value_decomposition, single_sided_procrustes_error


class OrthogonalProcrustes(Procrustes):

    """
    This method deals with the orthogonal Procrustes problem

    Given an m x n matrix A and a reference m x n matrix B, find the orthogonal
    transformation of A that brings it as close as possible to B
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
        u, s, v_trans = singular_value_decomposition(product_matrix)

        # Define the optimum orthogonal transformation
        u_optimum = np.dot(u, v_trans)

        # Map array_a to array_b with the optimum orthogonal transformation
        self.map_a_b(u_optimum)

        # Compute the real error
        real_error = single_sided_procrustes_error(self.a_transformed, self.b_in)

        return u_optimum, self.a_transformed, real_error, self.transformation


if __name__ == "__main__":

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    rot_matrix = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    a_prime = np.matmul(a, rot_matrix)

    # Find the transformation u which takes a_prime to a

    procrustes = OrthogonalProcrustes(a, a_prime, translate=False, scale=False)
    u_optimum, a_transformed, real_error, transformation, = procrustes.calculate()

    frobenius_error = single_sided_procrustes_error(a_transformed, a_prime)
