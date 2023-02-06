import pytest
import numpy as np
from scipy.stats import chi2

from repliclust import Archetype
from repliclust.utils import assemble_covariance_matrix
from repliclust.overlap._gradients import matvecprod_vectorized
from repliclust.overlap._gradients import cov_transform
from repliclust.overlap._gradients import marginal_std
from repliclust.overlap._gradients import get_1d_idx
from repliclust.overlap._gradients import compute_quantiles
from repliclust.overlap._gradients import mask_repulsive
from repliclust.overlap._gradients import check_for_attraction
from repliclust.overlap._gradients import assess_obs_separation
from repliclust.overlap._gradients import ReLU_vec, poly_vec, H_vec
from repliclust.overlap._gradients import single_cluster_loss
from repliclust.overlap._gradients import overlap_loss
from repliclust.overlap._gradients import quantile_gradients
from repliclust.overlap._gradients import loss_gradient_vec
from repliclust.overlap._gradients import apply_gradient_update
from repliclust.overlap._gradients import update_centers
from repliclust.overlap.centers import overlap2quantile_vec


def make_random_cluster_quantities():
    if np.random.uniform() > 0.5:
        # produce a case where clusters are too far apart
        loss_mode = 'attract'
        o_bounds = {'min': 0.1, 'max': 0.15}
    else:
        # produce a case where clusters are too close together
        loss_mode = 'repel'
        o_bounds = {'min': 0.001, 'max':0.01}
    mm = (Archetype(n_clusters=np.random.choice(np.arange(2,10)),
                    min_overlap=0.01,
                    max_overlap=0.05)
            .sample_mixture_model())
    centers = mm.centers 
    n_clusters = centers.shape[0]
    cov_list = [assemble_covariance_matrix(
                    axes=mm.axes_list[idx],
                    axis_lengths=mm.axis_lengths_list[idx],
                    inverse=False) for idx in range(n_clusters)]
    ave_cov_inv_list = [ np.linalg.inv((cov_list[i] + cov_list[j])/2)
                            for i in range(n_clusters)
                                for j in range(i+1,n_clusters) ]
    
    ref_cluster_idx = np.random.choice(np.arange(n_clusters))
    # Take subset of other clusters to simulate only looking at the
    # other clusters that violate overlap constraints
    n_other_clusters = np.random.choice(np.arange(1,n_clusters))
    all_other_cluster_idx = np.array(
        [j for j in range(n_clusters) if j != ref_cluster_idx], 
        dtype='int')
    other_cluster_idx = np.random.choice(
                            all_other_cluster_idx, size=n_other_clusters, 
                            replace=False).astype('int')

    # Make axes for different methods of calculating overlap 
    c2c_axis_mat = np.transpose(centers[[ref_cluster_idx],:]
                                - centers[other_cluster_idx,:])
    lda_axis_mat = matvecprod_vectorized(
                        ave_cov_inv_list, 
                        get_1d_idx(ref_cluster_idx, other_cluster_idx,
                                   n_clusters),
                        c2c_axis_mat)
    return {'centers': centers, 'cov_list': cov_list,
            'ave_cov_inv_list': ave_cov_inv_list,
            'ref_cluster_idx': ref_cluster_idx,
            'other_cluster_idx': other_cluster_idx,
            'c2c_axis_mat': c2c_axis_mat, 'lda_axis_mat': lda_axis_mat,
            'loss_mode': loss_mode,
            'q_bounds': {'min': overlap2quantile_vec(o_bounds['max']),
                         'max': overlap2quantile_vec(o_bounds['min'])}
            }


class TestGradientComputation:
    """
    Test cases for the results of gradient computation.
    """
    def test_update_centers(self):
        for random_data in range(10):
            mode = 'lda' if random_data % 2 == 0 else 'c2c'
            learning_rate = np.random.uniform(0.05,0.2)
            linear_penalty_weight = [0,0.3,0.78,1][int(random_data % 4)]
            data = make_random_cluster_quantities()
            i = data['ref_cluster_idx']
            n_clusters = data['centers'].shape[0]
            dim = data['centers'].shape[1]
            all_others_idx = [j for j in range(n_clusters) if j != i]
            
            # job of update centers is to apply the gradient update and
            # then output the loss gradient and loss number
            
            q, q_grad = quantile_gradients(
                    data['centers'][i,:], 
                    data['centers'][all_others_idx,:],
                    i, all_others_idx, cov_list=data['cov_list'],
                    ave_cov_inv_list=(None if mode=='c2c'
                                    else data['ave_cov_inv_list']),
                    axis_deriv_t_list=(
                        data['ave_cov_inv_list'] if mode=='lda'
                            else [np.eye(dim) for i in range(n_clusters)
                                    for j in range(i+1, n_clusters)]),
                    mode=mode,
                    )
            # check if we are in the attractive or repulsive case
            loss_mode = ('attract' if check_for_attraction(
                                      q, max_q=data['q_bounds']['max']) 
                         else 'repel')
            q_thold = (data['q_bounds']['max'] if loss_mode=='attract'
                         else data['q_bounds']['min'])
            grad = loss_gradient_vec(q, q_thold, q_grad, mode=loss_mode,
                                     linear_weight=linear_penalty_weight)
            
            # if we are in the attractive case, subset into the argmax
            # for the separation quantile (otherwise the minimum in the
            # formula won't be felt)
            if loss_mode=='attract':
                argmin_idx = np.argmin(q)
                q = q[:,[argmin_idx]]
                grad = grad[:,[argmin_idx]]
                all_others_idx = [all_others_idx[argmin_idx]]

            # test that we selected the right case for the loss
            assert loss_mode == data['loss_mode']

            # take gradient step manually
            centers_manual_update = np.copy(data['centers'])
            apply_gradient_update(centers_manual_update, grad, i,
                                  all_others_idx, learning_rate,
                                  mode = loss_mode)

            # take the gradient step using 'update_centers'
            update_centers(i, data['centers'], cov_list=data['cov_list'],
                           ave_cov_inv_list=(None if mode=='c2c' else
                                             data['ave_cov_inv_list']),
                           axis_deriv_t_list= (
                                data['ave_cov_inv_list'] if mode=='lda'
                                else [np.eye(dim) for i in range(n_clusters)
                                        for j in range(i+1,n_clusters)]),
                           learning_rate=learning_rate,
                           linear_penalty_weight=linear_penalty_weight,
                           quantile_bounds=data['q_bounds'],
                           mode=mode)
            
            # compare the outcomes of the two gradient steps
            assert np.allclose(centers_manual_update, data['centers'])
            assert centers_manual_update.shape == data['centers'].shape


    def test_apply_gradient_update(self):
        for random_data in range(10):
            mode = 'lda' if random_data % 2 == 0 else 'c2c'
            learning_rate = np.random.uniform(0.05,0.2)

            # initialize centers
            data = make_random_cluster_quantities()
            centers = data['centers']
            n_clusters = centers.shape[0]; dim = centers.shape[1]
            ref_cluster_idx = data['ref_cluster_idx']
            other_cluster_idx = data['other_cluster_idx']
        
            # apply gradient by hand
            q, q_grad = quantile_gradients(
                            centers[ref_cluster_idx,:],
                            centers[other_cluster_idx,:],
                            ref_cluster_idx,
                            other_cluster_idx,
                            cov_list=data['cov_list'],
                            ave_cov_inv_list=data['ave_cov_inv_list'],
                            axis_deriv_t_list=(data['ave_cov_inv_list']
                                if mode=='lda' 
                                else [np.eye(dim) 
                                      for i in range(n_clusters)
                                        for j in range(i+1,n_clusters)]),
                            mode=mode)
            q_thold = (data['q_bounds']['min'] 
                        if data['loss_mode']=='repel'
                        else data['q_bounds']['max'])
            loss_grad = loss_gradient_vec(q, q_thold, q_grad)

            # apply gradient by hand and using 'apply_gradient_update'
            centers_manual_update = np.copy(centers)
            centers_manual_update[ref_cluster_idx,:] -= (learning_rate
                * np.sum(loss_grad,axis=1)) 
            centers_manual_update[other_cluster_idx,:] -= (learning_rate
                * np.transpose(-loss_grad))

            apply_gradient_update(centers, loss_grad,
                                    ref_cluster_idx, other_cluster_idx,
                                    learning_rate=learning_rate, 
                                    mode=data['loss_mode'])
            assert np.array_equal(centers_manual_update, centers)

    def test_quantile_gradients(self):
        for random_data in range(10):
            mode = 'lda' if random_data % 2 == 0 else 'c2c'
            data = make_random_cluster_quantities()
            n_clusters = data['centers'].shape[0]
            dim = data['centers'].shape[1]
            i = data['ref_cluster_idx']
            cov_i = data['cov_list'][i]
            grad_mtx = []
            # Compute gradient matrix by separately computing columns
            for j in data['other_cluster_idx']:
                cov_j = data['cov_list'][j]
                center_diff = data['centers'][i,:] - data['centers'][j,:]
                axis = (center_diff if mode=='c2c'
                            else (np.linalg.inv((cov_i + cov_j)/2)
                                @ center_diff))
                std_i = np.sqrt(np.dot(axis, cov_i @ axis))
                std_j = np.sqrt(np.dot(axis, cov_j @ axis))
                # Compute the j-th gradient vector using product rule
                if mode=='c2c':
                    first_term = 2*center_diff / (std_i + std_j)
                    second_term = (0.5*np.dot(axis, center_diff)
                                    * (-1/((std_i + std_j)**2))
                                    * ((cov_i @ axis) / std_i
                                       + (cov_j @ axis) / std_j))
                    grad_j = first_term + second_term
                elif mode=='lda':
                    first_term = 2*axis / (std_i + std_j)
                    second_term = np.linalg.solve(
                                    (cov_i + cov_j)/2,
                                    (0.5*np.dot(axis, center_diff)
                                        * (-1/((std_i + std_j)**2))
                                        * ((cov_i @ axis) / std_i
                                        + (cov_j @ axis) / std_j)))
                    grad_j = first_term + second_term
                assert grad_j.shape == (dim,)
                grad_mtx.append(grad_j[:,np.newaxis])
            assert [col.shape == (dim,1) for col in grad_mtx]
            desired_result = np.concatenate(grad_mtx, axis=1)
            assert desired_result.dtype != 'object'
            _, computed_result = quantile_gradients(
                ref_center=data['centers'][i,:],
                other_centers=data['centers'][data['other_cluster_idx'],:],
                ref_cluster_idx=data['ref_cluster_idx'],
                other_cluster_idx=data['other_cluster_idx'],
                cov_list = data['cov_list'],
                ave_cov_inv_list = (None if mode=='c2c' 
                                        else data['ave_cov_inv_list']),
                axis_deriv_t_list= (data['ave_cov_inv_list'] if mode=='lda'
                                    else [np.eye(dim) 
                                          for i in range(n_clusters)
                                            for j in range(i+1,n_clusters)]),
                mode=mode
            )
            print(desired_result)
            print(computed_result)
            assert np.allclose(desired_result, computed_result)
            assert desired_result.shape == computed_result.shape

    def test_loss_gradient_vec(self):
        for random_data in range(10):
            data = make_random_cluster_quantities()
            mode = 'lda' if random_data % 2 == 0 else 'c2c'
            i = data['ref_cluster_idx']
            n_clusters = data['centers'].shape[0]
            dim = data['centers'].shape[1]
            linear_penalty_weight = [0, 0.3, 0.75, 1][
                                        int(random_data % 4)
                                    ]

            # compute quantiles
            q = compute_quantiles(i, data['other_cluster_idx'],
                                  centers=data['centers'],
                                  cov_list=data['cov_list'],
                                  ave_cov_inv_list=data['ave_cov_inv_list'],
                                  mode=mode)
            # compute quantiles gradient
            axis_deriv_t_list = (data['ave_cov_inv_list'] if mode=='lda' else
                                    [np.eye(dim) 
                                      for i in range(n_clusters)
                                        for j in range(i+1,n_clusters)])
            
            q, q_grad = quantile_gradients(
                            data['centers'][i,:],
                            data['centers'][data['other_cluster_idx'],:],
                            data['ref_cluster_idx'],
                            data['other_cluster_idx'],
                            cov_list=data['cov_list'],
                            ave_cov_inv_list=data['ave_cov_inv_list'],
                            axis_deriv_t_list=axis_deriv_t_list,
                            mode=mode,
                            )
            # compute loss gradient bottom-up
            if data['loss_mode']=='repel':
                q_thold = data['q_bounds']['min']
                penalty_grad = (
                    (-1)*(linear_penalty_weight*H_vec(q_thold - q)
                        + (1-linear_penalty_weight)*2*ReLU_vec(q_thold - q))
                )
            elif data['loss_mode']=='attract':
                q_thold = data['q_bounds']['max']
                q = q[:,np.argmin(q)]
                penalty_grad = (
                    linear_penalty_weight*H_vec(q - q_thold)
                        + (1-linear_penalty_weight)*2*ReLU_vec(q - q_thold)
                )
            desired_result = penalty_grad * q_grad
            # compute loss gradient top-bown
            q_thold = (data['q_bounds']['max']
                        if data['loss_mode']=='attract'
                        else data['q_bounds']['min'])
            observed_result = loss_gradient_vec(
                                q, q_thold, q_grad,
                                mode=data['loss_mode'], 
                                linear_weight=linear_penalty_weight)
            # compare the results
            assert np.allclose(desired_result, observed_result)
            assert desired_result.shape == observed_result.shape


class TestGradientComputationHelpers:
    """
    Test cases for the helper functions that pre-compute the
    mathematical quantities required for gradient computation in a
    vectorized way.
    """
    def test_matvecprod_vectorized(self):
        matrix_list = [(i % 2)*np.ones(shape=(3,3)) 
                            for i in range(6)]
        matrix_col_idx_list = [1,3,5]
        matrix_col_idx_array = np.array(matrix_col_idx_list,dtype='int')
        vectors_mat = np.array([[1,2,3],[1,2,3],[1,2,3]])
        result_from_list_input = matvecprod_vectorized(
                                    matrix_list, matrix_col_idx_list, 
                                    vectors_mat)
        result_from_array_input = matvecprod_vectorized(
                                    matrix_list, matrix_col_idx_array, 
                                    vectors_mat)
        result_desired = np.array([[3,6,9],[3,6,9],[3,6,9]])

        for i in range(3):
            assert np.array_equal(result_from_list_input[i],
                                  result_from_array_input[i])
            assert np.array_equal(result_from_array_input[i],
                                  result_desired[i])

    def test_cov_transform(self):
        axis_mat = np.array([[1,-1,2,3],[1,-1,2,3],[1,-1,2,3]])
        cov_list = [(i+1)*np.eye(3) for i in range(8)]
        ref_cluster_idx = 2
        other_cluster_idx = np.array([1,4,5,7],dtype='int')
        output_specific = np.array([[2, -5, 12, 24],
                                    [2, -5, 12, 24],
                                    [2, -5, 12, 24]])
        output_reference = np.array([[3, -3, 6, 9],
                                     [3, -3, 6, 9],
                                     [3, -3, 6, 9]])
        result_reference, result_specific = cov_transform(
                                                axis_mat, cov_list,
                                                ref_cluster_idx, 
                                                other_cluster_idx)
        assert np.array_equal(result_reference, output_reference)
        assert np.array_equal(result_specific, output_specific)

    def test_marginal_std(self):
        axis_mat = np.array([[1,0],
                             [0,1],
                             [1,1]])
        ref_transform = np.array([[1,1],
                                  [0,0],
                                  [2,2]])
        specific_transform = np.array([[3,2],
                                       [0,0],
                                       [4,3]])
        (std_ref, std_other) = marginal_std(axis_mat, ref_transform, 
                                            specific_transform)
        assert np.array_equal(std_ref, np.array([[np.sqrt(3), 
                                                  np.sqrt(2)]]))
        assert np.array_equal(std_other, np.array([[np.sqrt(7), 
                                                    np.sqrt(3)]]))

    def test_get_1d_idx(self):
        # i < j
        assert np.all(get_1d_idx(0,np.array([1,2,3,4]),6)
                        == np.array([0,1,2,3]))
        assert np.all(get_1d_idx(1,np.array([3,4,5]),6)
                        == np.array([6,7,8]))
        assert np.all(get_1d_idx(2, np.array([3,5]),6)
                        == np.array([9,11]))
        assert np.all(get_1d_idx(3,np.array([4]),6)
                        == np.array([12]))
        assert np.all(get_1d_idx(4,np.array([5]),6)
                        == np.array([14]))
        # i > j
        assert np.all(get_1d_idx(3,np.array([1,2,4]),6)
                        == np.array([6,9,12]))
        assert np.all(get_1d_idx(4,np.array([3]),6)
                        == np.array([12]))
        assert np.all(get_1d_idx(5,np.array([4]),6)
                        == np.array([14]))


class TestQuantileComputation:
    """ 
    Test cases for functions related to cluster separation quantiles.
    """
    def test_compute_quantiles(self):
        for random_data in range(10):
            data = make_random_cluster_quantities()

            # Compute results using this function
            lda_result = compute_quantiles(
                            data['ref_cluster_idx'],
                            data['other_cluster_idx'], 
                            data['centers'],
                            cov_list=data['cov_list'], 
                            ave_cov_inv_list=data['ave_cov_inv_list'],
                            mode='lda'
                            )
            print("lda result", lda_result)
            
            c2c_result = compute_quantiles(
                            data['ref_cluster_idx'],
                            data['other_cluster_idx'], 
                            data['centers'],
                            cov_list=data['cov_list'], 
                            ave_cov_inv_list=data['ave_cov_inv_list'],
                            mode='c2c'
                            )
            print("c2c result", c2c_result)

            # Compute the ground truth results bottom-up
            numerator_lda = np.sum(data['lda_axis_mat'] 
                                    * data['c2c_axis_mat'], 
                                    axis=0)[np.newaxis,:]
            numerator_c2c = np.sum(data['c2c_axis_mat']
                                    * data['c2c_axis_mat'],
                                    axis=0)[np.newaxis,:]

            ref_tf_lda, other_tf_lda = cov_transform(
                                        data['lda_axis_mat'], 
                                        data['cov_list'],
                                        data['ref_cluster_idx'],
                                        data['other_cluster_idx']
            )
            ref_tf_c2c, other_tf_c2c = cov_transform(
                                        data['c2c_axis_mat'], 
                                        data['cov_list'],
                                        data['ref_cluster_idx'],
                                        data['other_cluster_idx']
            )

            ref_std_lda, other_std_lda = marginal_std(
                                            data['lda_axis_mat'], 
                                            ref_tf_lda, other_tf_lda)
            ref_std_c2c, other_std_c2c = marginal_std(
                                            data['c2c_axis_mat'], 
                                            ref_tf_c2c, other_tf_c2c)
            desired_lda_result = ((numerator_lda 
                                    / (ref_std_lda + other_std_lda))
                                    .flatten())
            desired_c2c_result = ((numerator_c2c
                                    /(ref_std_c2c + other_std_c2c))
                                    .flatten())

            # Compare the two sets of results
            assert np.array_equal(desired_lda_result, lda_result)
            assert np.array_equal(desired_c2c_result, c2c_result)

            # Check output shapes
            assert lda_result.shape == (len(data['other_cluster_idx']),)
            assert c2c_result.shape == (len(data['other_cluster_idx']),)


    def test_mask_repulsive(self):
        assert np.array_equal(mask_repulsive(q = np.array([1,2,0.5]), 
                                             min_q = 0.25),
                              np.array([False,False,False]))
        assert np.array_equal(mask_repulsive(q = np.array([1,2,0.5]), 
                                             min_q = 0.75),
                              np.array([False,False,True]))
        assert np.array_equal(mask_repulsive(o = np.array([0.1,0.3,0.2]), 
                                             max_o = 0.25),
                              np.array([False,True,False]))
        assert np.array_equal(mask_repulsive(o = np.array([0.1,0.1,0.1]), 
                                             max_o = 1e-4),
                              np.array([True,True,True]))

    def test_check_for_attraction(self):
        assert check_for_attraction(q = np.array([1,2,0.5]), 
                                    max_q = 0.25)
        assert not check_for_attraction(q = np.array([1,2,0.5]),
                                    max_q = 0.75)
        assert not check_for_attraction(o = np.array([0.1,0.3,0.2]), 
                                        min_o = 0.25)
        assert not check_for_attraction(o = np.array([0.1,0.1,0.1]),
                                        min_o = 1e-4)
        assert check_for_attraction(o = np.array([0.01,0.05,0.02]),
                                    min_o = 0.75)                          

    def test_assess_obs_separation(self):
        for random_data in range(10):
            # Compute desired results bottom-up
            data = make_random_cluster_quantities()
            mode = 'lda' if random_data % 2 == 0 else 'c2c'
            n_clusters = data['centers'].shape[0]
            quantiles_mtx = []
            for ref_cluster_idx in range(n_clusters):
                other_cluster_idx = np.array(
                                        [i for i in range(n_clusters) 
                                            if i != ref_cluster_idx],
                                        dtype='int')
                quantiles_mtx.append(list(compute_quantiles(
                                            ref_cluster_idx,
                                            other_cluster_idx,
                                            data['centers'],
                                            data['cov_list'],
                                            data['ave_cov_inv_list'],
                                            mode=mode)))
            quantiles_array = np.array(quantiles_mtx)
            desired_min_q_obs = np.min(quantiles_array)
            print(quantiles_array)
            desired_max_q_obs = np.max(np.min(quantiles_array, 
                                              axis=1))
            
            # Compute results top-down and compare
            results = assess_obs_separation(data['centers'],
                                            data['cov_list'],
                                            data['ave_cov_inv_list'],
                                            mode=mode)
            assert np.allclose(results['min'], desired_min_q_obs)
            assert np.allclose(results['max'], desired_max_q_obs)


class TestLossMetrics:
    """ 
    Test cases for functions involved in computing overlap loss.
    """
    def test_ReLU_vec(self):
        x = np.array([0, -1, 2, 3, -0.5, 0])
        assert np.array_equal(ReLU_vec(x), np.array([0,0,2,3,0,0]))

    def test_poly_vec(self):
        x = np.array(np.random.normal(size=100))
        assert np.array_equal(poly_vec(x, linear_weight=0), x**2)
        assert np.array_equal(poly_vec(x, linear_weight=1), np.abs(x))
        assert np.array_equal(poly_vec(x, linear_weight=0.3), 
                                0.3*np.abs(x) + 0.7*(x**2))

    def test_H_vec(self):
        x = np.array(np.random.normal(size=100))
        assert np.array_equal(H_vec(x), (x >= 0).astype('float'))

    def test_single_cluster_loss(self):
        # Example where clusters are too close
        q_bounds = {'min': 1.5, 'max': 3}
        quantiles = np.array([0.5, 1, 2, 4, 1.5])
        assert np.allclose(single_cluster_loss(quantiles, q_bounds, 0.25),
                           (0.25*(1.5 - 0.5) + 0.75*((1.5-0.5)**2)
                            + 0.25*(1.5 - 1) + 0.75*((1.5 - 1)**2)))
        # Example where clusters are too far apart
        q_bounds = {'min': 0.5, 'max': 1}
        quantiles = np.array([2, 1.25, 2, 4, 1.5])
        assert np.allclose(single_cluster_loss(quantiles, q_bounds, 0.4),
                           0.4*(1.25 - 1) + 0.6*((1.25 - 1)**2))

    def test_overlap_loss(self):
        q_bounds = {'min': 1.3, 'max': 2.7}; linear_penalty_weight = 0.1
        for random_data in range(10):
            mode = 'lda' if random_data % 2 == 0 else 'c2c'
            data = make_random_cluster_quantities()
            n_clusters = data['centers'].shape[0]
            # accumulate overlap loss by summing single-cluster losses
            loss_sum = 0
            for ref_cluster_idx in range(n_clusters):
                quantiles = compute_quantiles(
                                ref_cluster_idx, 
                                [i for i in range(n_clusters)
                                    if i != ref_cluster_idx],
                                centers=data['centers'],
                                cov_list=data['cov_list'],
                                ave_cov_inv_list=data['ave_cov_inv_list'],
                                mode=mode)
                ref_cluster_loss = single_cluster_loss(
                                        quantiles,
                                        q_bounds,
                                        linear_penalty_weight
                                        )
                loss_sum += ref_cluster_loss/n_clusters
            assert np.allclose(
                        overlap_loss(
                            data['centers'], q_bounds, 
                            linear_penalty_weight, mode=mode, 
                            cov_list=data['cov_list'],
                            ave_cov_inv_list=data['ave_cov_inv_list']),
                        loss_sum
                        )