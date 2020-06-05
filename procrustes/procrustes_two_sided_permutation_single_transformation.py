__author__ = 'Jonny'

from procrustes.base import Procrustes
from procrustes.procrustes_two_sided_orthogonal_single_transformation import \
    TwoSidedOrthogonalSingleTransformationProcrustes
from procrustes.procrustes_permutation import PermutationProcrustes
import numpy as np
import time
from procrustes.base_utils import hide_zero_padding_array, double_sided_procrustes_error


class TwoSidedPermutationSingleTransformationProcrustes(Procrustes):

    """
    This method deals with the two-sided orthogonal Procrustes problem
    limited to a single transformation

    We require symmetric input arrays to perform this analysis
    """

    """
    translate and scale for this analysis is False by default. The reason is that the inputs are really the outputs of two-
    sided single orthogonal procrustes, where translate_scale is True.
    """

    def __init__(
            self,
            array_a,
            array_b,
            translate=False,
            scale=False,
            preserve_symmetry=True,
            hide_padding=True,
            translate_symmetrically=False
    ):
        self.hide_padding = hide_padding
        self.translate = translate
        self.scale = scale
        self.preserve_symmetry = preserve_symmetry
        self.translate_symmetrically = translate_symmetrically

        Procrustes.__init__(
            self,
            array_a,
            array_b,
            translate=self.translate,
            scale=self.scale,
            preserve_symmetry=self.preserve_symmetry,
            hide_padding=self.hide_padding,
            translate_symmetrically=self.translate_symmetrically
        )
        if (abs(self.array_a - self.array_a.T) > 1.e-10).all() or (abs(self.array_b - self.array_b.T) > 1.e-10).all():
            raise ValueError('Arrays array_a and array_b must both be symmetric for this analysis.')

    def calculate(self, tol=1e-5):
        """
        Calculates the single optimum two-sided permuation transformation matrix in the
        double-sided procrustes problem

        Returns
        ----------
        perm_optimum, array_transformed, error
        perm_optimum= the optimum permutation transformation array satisfying the double
             sided procrustes problem. Array represents the closest permutation array to
             u_umeyama given by the permutation procrustes problem
        array_transformed = the transformed input array after transformation by perm_optimum
        error = the error as described by the double-sided procrustes problem
        """
        # Timing information
        time_start = time.time()

        # Arrays already translated_scaled
        array_a = self.array_a
        array_b = self.array_b

        # Initialize
        perm_optimum1 = None
        perm_optimum2 = None
        perm_optimum3 = None

        """Finding initial guess"""
        # Method 1

        # Solve for the optimum initial permutation transformation array by finding the closest permutation
        # array to u_umeyama_approx given by the two-sided orthogonal single transformation problem
        twosided_ortho_single_trans = TwoSidedOrthogonalSingleTransformationProcrustes(array_a, array_b,
                               hide_padding=self.hide_padding, translate_symmetrically=self.translate_symmetrically)
        u_approx, u_best, array_transformed_approx, array_transformed_best, error_approx, error_best,\
        unused_translate_and_or_scale = twosided_ortho_single_trans.calculate(return_u_approx=True, return_u_best=True)
        m, n = u_best.shape
        # Find the closest permutation matrix to the u_optima obtained from TSSTO with permutation procrustes analysis
        if u_approx is not None:
            perm1 = PermutationProcrustes(np.eye(m), u_approx, scale=self.scale, translate=False, hide_padding=self.hide_padding,
                                          translate_symmetrically=self.translate_symmetrically)
            perm_optimum_trans1, array_transformed1, total_potential1, error1, unused_translate_and_or_scale =\
                perm1.calculate()
            perm_optimum1 = perm_optimum_trans1  # Initial guess due to u_approx
            least_error_array_transformed = array_transformed1  # Arbitrarily initiate the least error transformed array

        if u_best is not None:
            perm2 = PermutationProcrustes(np.eye(m), u_best, scale=self.scale, translate=False,
                                hide_padding=self.hide_padding, translate_symmetrically=self.translate_symmetrically)
            perm_optimum_trans2, array_transformed2, total_potential2, error2, unused_translate_and_or_scale =\
                perm2.calculate()
            perm_optimum2 = perm_optimum_trans2  # Initial guess due to u_exact
            least_error_array_transformed = array_transformed2  # Arbitrarily initiate the least error transformed array

        # Method 2
        if self.hide_padding:
            array_a = hide_zero_padding_array(array_a)
            array_b = hide_zero_padding_array(array_b)
        n_a, m_a = array_a.shape
        diagonals_a = np.diagonal(array_a)
        b = np.zeros((2 * n_a - 1, m_a))
        # First row is the diagonal elements
        b[0, :] = diagonals_a
        # Populate remaining rows with columns of array_a sorted from greatest to least (excluding diagonals)
        for j in range(m_a):
            col_j = array_a[:, j]
            col_j = np.delete(col_j, j)
            col_j_abs_idx_sorted = abs(col_j).argsort()[::-1]
            col_j_sorted = col_j[col_j_abs_idx_sorted]
            col_j_sorted = np.insert(col_j_sorted, 0, array_a[col_j_abs_idx_sorted[0] + 1, col_j_abs_idx_sorted[0] + 1])

            idx_counter = 0
            for i in range(2, 2 * (m_a - 1), 2):
                idx_counter += 1
                idx_aj = col_j_abs_idx_sorted[idx_counter]
                col_j_sorted = np.insert(col_j_sorted, i, array_a[idx_aj + 1, idx_aj + 1])

            b[1:, j] = col_j_sorted

        # Array B
        n_a0, m_a0 = array_b.shape
        diagonals_a0 = np.diagonal(array_b)
        b0 = np.zeros((2 * n_a0 - 1, m_a0))
        # First row is the diagonal elements
        b0[0, :] = diagonals_a0
        # Populate remaining rows with columns of array_a sorted from greatest to least (excluding diagonals)
        for j in range(m_a0):
            col_j = array_b[:, j]
            col_j = np.delete(col_j, j)
            col_j_abs_idx_sorted = abs(col_j).argsort()[::-1]
            col_j_sorted = col_j[col_j_abs_idx_sorted]
            col_j_sorted = np.insert(col_j_sorted, 0, array_b[col_j_abs_idx_sorted[0] + 1, col_j_abs_idx_sorted[0] + 1])

            idx_counter = 0
            for i in range(2, 2 * (m_a - 1), 2):
                idx_counter += 1
                # idx = np.insert(idx, i, 0)
                idx_aj = col_j_abs_idx_sorted[idx_counter]
                col_j_sorted = np.insert(col_j_sorted, i, array_b[idx_aj + 1, idx_aj + 1])

            b0[1:, j] = col_j_sorted

        # Match the matrices b and b0 via the permutation procrustes problem
        perm = PermutationProcrustes(b, b0, scale=self.scale, translate=self.translate, hide_padding=self.hide_padding)
        perm_optimum3, array_transformed, total_potential, error, translate_and_or_scale = perm.calculate()

        """
        Done finding initial guesses.
        """

        least_error_perm = perm_optimum1  # Arbitrarily initiate the least error perm. Will be adjusted in
        #  following procedure
        initial_perm_list = [perm_optimum1, perm_optimum2, perm_optimum3]

        min_error = 1.e8  # Arbitrarily initialize error ; will be adjusted in following procedure
        # initial_perm_list[0] = None
        # initial_perm_list[1] = None

        for k in range(3):
            perm_optimum = initial_perm_list[k]
            if perm_optimum is not None:
                """Beginning Iterative Procedure"""
                n, m = perm_optimum.shape

                # Initializing updated arrays. See literature for a full description of algorithm
                t_array = np.dot(np.dot(array_a, perm_optimum), array_b)
                p_new = perm_optimum
                p_old = perm_optimum
                # For simplicity, shorten t_array to t.
                t = t_array
                # Arbitrarily initialize error
                # Define breakouter, a boolean value which will skip the current method if NaN values occur

                iteration = 0
                end_while = False
                max_iterations = 10000
                while (error > tol and iteration < max_iterations) and end_while is False:
                    for i in range(n):
                        for j in range(m):
                            # compute sqrt factor in (28)
                            num = 2 * t[i, j]
                            denom = np.dot(p_old, (np.dot(p_old.T, t)) + (np.dot(p_old.T, t)).T)[i, j]
                            factor = np.sqrt(abs(num / denom))
                            if type(factor) is not float:
                                end_while = True
                                break
                            iteration += 1
                            print(iteration)
                            p_new[i, j] = p_old[i, j] * factor
                        if end_while is True:
                            break
                    if end_while is True:
                        break
                    error = np.trace(np.dot((p_new - p_old).T, (p_new - p_old)))

                """Converting optimal permutation (step 2) into permutation matrix """
                # Convert the array found above into a permutation matrix with permutation procrustes analysis
                # perm = PermutationProcrustes(np.eye(n), p_new,translate = self.translate, scale = self.scale,
                #                              preserve_symmetry=self.preserve_symmetry, hide_padding=self.hide_padding)
                perm = PermutationProcrustes(np.eye(n), p_new, scale=True, translate=self.translate, hide_padding=self.hide_padding)
                perm_optimum, array_transformed, total_potential, error, unused_translate_and_or_scale = perm.calculate()

                # Calculate the error
                error_perm_optimum = double_sided_procrustes_error(array_a, array_b, perm_optimum, perm_optimum)

                if error_perm_optimum < min_error:
                    least_error_perm = perm_optimum
                    min_error = error_perm_optimum
        # Timing
        time_elapsed = time.time() - time_start
        print('Analysis Completed with time {0}'.format(time_elapsed))
        # Map array_a to array_b
        self.map_a_b(least_error_perm.T, least_error_perm, preserve_symmetry=True, translate_symmetrically=self.translate_symmetrically)

        # Real Error
        least_error_array_transformed = np.dot(np.dot(least_error_perm.T, self.array_a), least_error_perm)
        real_error = double_sided_procrustes_error(self.a_transformed, self.array_b)

        return least_error_perm, least_error_array_transformed, real_error, self.transformation
