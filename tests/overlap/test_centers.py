import pytest
import numpy as np
from scipy.stats import norm, ortho_group

from repliclust import Archetype
from repliclust.utils import assemble_covariance_matrix
from repliclust.overlap.centers import ConstrainedOverlapCenters
from repliclust.overlap.centers import assess_obs_overlap
from repliclust.overlap.centers import overlap2quantile_vec
from repliclust.overlap.centers import quantile2overlap_vec
from repliclust.overlap._gradients import overlap_loss
from repliclust.overlap._gradients import assess_obs_separation
from repliclust.overlap._gradients import compute_quantiles


def make_random_clusters(overlap_mode):
    max_overlap = [1e-3,1e-2,1e-1][np.random.choice(3)]
    min_overlap = max_overlap/2
    mix_model = (Archetype(max_overlap=max_overlap, 
                           min_overlap=min_overlap)
                    .sample_mixture_model())
    centers = (mix_model.centers
                + np.random.normal(size=mix_model.centers.shape))
    cov_list = [ assemble_covariance_matrix(
                    mix_model.axes_list[i],
                    mix_model.axis_lengths_list[i])
                    for i in range(centers.shape[0]) ]
    ave_cov_inv_list = (None if overlap_mode=='c2c' else
        [np.linalg.inv((cov_list[i] + cov_list[j])/2)
            for i in range(centers.shape[0])
                for j in range(i+1, centers.shape[0])])
    axis_deriv_t_list = (None if overlap_mode=='c2c' 
                            else ave_cov_inv_list)

    return {'centers': centers, 'cov_list': cov_list,
            'ave_cov_inv_list': ave_cov_inv_list,
            'axis_deriv_t_list': axis_deriv_t_list,
            'max_overlap': max_overlap,
            'min_overlap': min_overlap}

class TestHelpers:
    def test_overlap2quantile_vec(self):
        # test vectorial input
        overlaps=np.array([0.001,0.05,0.68])
        assert np.allclose(overlap2quantile_vec(overlaps),
                           norm.ppf(1-overlaps/2))
        assert overlap2quantile_vec(overlaps).shape == overlaps.shape
        # test single input
        overlap=0.25
        assert np.allclose(overlap2quantile_vec(overlap),
                           norm.ppf(1-overlap/2))

    def test_quantile2overlap_vec(self):
        # test vectorial input
        quantiles=np.array([0.5,1,2,3,10])
        assert np.allclose(quantile2overlap_vec(quantiles),
                           2*(1-norm.cdf(quantiles)))
        assert (quantile2overlap_vec(quantiles).shape == 
                    norm.cdf(quantiles).shape)
        # test single input
        quantile=2.23
        assert np.allclose(quantile2overlap_vec(quantile),
                           2*(1-norm.cdf(quantile)))

    def test_inverse_relationship(self):
        for random_run in range(10):
            quantiles = np.random.uniform(0,5,size=10)
            assert np.allclose(
                    overlap2quantile_vec(quantile2overlap_vec(quantiles)),
                    quantiles)
            assert (overlap2quantile_vec(
                            quantile2overlap_vec(quantiles)).shape
                        == quantiles.shape)
            overlaps = np.random.uniform(0,1,size=10)
            assert np.allclose(
                    quantile2overlap_vec(overlap2quantile_vec(overlaps)),
                    overlaps)
            assert (quantile2overlap_vec(
                            quantile2overlap_vec(overlaps)).shape
                        == overlaps.shape)            

    def test_assess_obs_overlap(self):
        for random_data in range(20):
            overlap_mode = ['c2c','lda'][int(random_data % 2)]
            # make random cluster centers
            n_clusters = np.random.choice(np.arange(2,30))
            dim = np.random.choice(np.array([2,50,100,500]))
            centers = np.random.normal(size=(n_clusters,dim))

            # make random covariance matrices
            cov_list = [
                (ortho 
                @ np.diag(np.random.exponential(scale=3,size=dim))
                    @ np.transpose(ortho)) 
                        for ortho in [ortho_group.rvs(dim=dim) 
                                      for i in range(n_clusters)]]

            # compute ave_cov_inv_list
            ave_cov_inv_list = (
                None if overlap_mode=='c2c' 
                     else [np.linalg.inv((cov_list[i] + cov_list[j])/2)
                                for i in range(n_clusters)
                                    for j in range(i+1, n_clusters)])   
        
            # compute quantiles using _gradients module
            min_quantiles = []
            for ref_cluster_idx in range(n_clusters):
                q = compute_quantiles(
                        ref_cluster_idx,
                        [j for j in range(n_clusters)
                            if j != ref_cluster_idx],
                        centers=centers, cov_list=cov_list,
                        ave_cov_inv_list=ave_cov_inv_list, 
                        mode=overlap_mode
                        )
                min_quantiles.append(np.min(q))
            minmin_quantile = np.min(min_quantiles)
            maxmin_quantile = np.max(min_quantiles)

            # convert quantiles into overlaps  
            maxmax_overlap = quantile2overlap_vec(minmin_quantile)
            minmax_overlap = quantile2overlap_vec(maxmin_quantile)
            obs_overlap_desired = {'min': minmax_overlap,
                                   'max': maxmax_overlap}

            # compare the result to applying assess_obs_overlaps
            obs_overlap_computed = assess_obs_overlap(
                                    centers, cov_list=cov_list,
                                    ave_cov_inv_list=ave_cov_inv_list,
                                    mode=overlap_mode)
            assert np.allclose(obs_overlap_computed['min'],
                               obs_overlap_desired['min'])
            assert np.allclose(obs_overlap_computed['max'],
                               obs_overlap_desired['max'])

class TestInternalMechanics:
    def test_check_for_continuation(self):
        my_centers = ConstrainedOverlapCenters(max_epoch=133, 
                                               ATOL=1e-6,
                                               RTOL=1e-3)
        # Test responsiveness to epoch count
        assert my_centers._check_for_continuation(epoch=132, 
                                                  loss_curr=10,
                                                  loss_prev=100,
                                                  grad_size=10)
        assert not my_centers._check_for_continuation(epoch=133,
                                                      loss_curr=10,
                                                      loss_prev=100,
                                                      grad_size=10)
        # Test responsiveness to loss reaching zero
        assert not my_centers._check_for_continuation(epoch=1,
                                                  loss_curr=0.0,
                                                  loss_prev=10,
                                                  grad_size=10)
        
        # Test responsiveness to loss approaching zero within tolerance
        my_centers.linear_penalty_weight = 0
        assert not my_centers._check_for_continuation(
                                epoch=1,
                                loss_curr=my_centers.ATOL/2,
                                loss_prev=10,
                                grad_size=10)
        my_centers.linear_penalty_weight = 0.5
        
        # Test responsiveness to loss failing to decrease appreciably
        assert not my_centers._check_for_continuation(
                                    epoch=1,
                                    loss_curr=10*(1-my_centers.RTOL/2),
                                    loss_prev=10,
                                    grad_size=10)
        assert my_centers._check_for_continuation(
                                    epoch=1,
                                    loss_curr=10*(1-2*my_centers.RTOL),
                                    loss_prev=10,
                                    grad_size=10
                                    )

        # Test responsiveness to a small gradient
        assert not my_centers._check_for_continuation(
                                    epoch=1,
                                    loss_curr=1,
                                    loss_prev=10,
                                    grad_size=my_centers.ATOL/2
                                    )
        assert my_centers._check_for_continuation(
                                    epoch=1,
                                    loss_curr=1,
                                    loss_prev=10,
                                    grad_size=2*my_centers.ATOL
                                    )

    def test_optimize_centers(self):
        # Make sure that loss returned at the end is the true loss
        zero_loss_count = 0 # do not expect zero loss in context below
        n_runs = 50
        for random_data in range(n_runs):
            overlap_mode = ['lda', 'c2c'][int(random_data % 2)]
            linear_penalty_weight = [0,.14,.84,1][int(random_data % 4)]
            data = make_random_clusters(overlap_mode=overlap_mode)
            my_centers = ConstrainedOverlapCenters(
                            max_epoch=1,
                            learning_rate=1e-5,
                            max_overlap=data['max_overlap'],
                            min_overlap=data['min_overlap'],
                            overlap_mode=overlap_mode,
                            linear_penalty_weight=linear_penalty_weight)
            returned_loss = my_centers._optimize_centers(
                                data['centers'], 
                                cov_list=data['cov_list'], 
                                ave_cov_inv_list=data['ave_cov_inv_list'], 
                                axis_deriv_t_list=data['axis_deriv_t_list'], 
                                quiet=True, progress_bar=None
                                )
            desired_loss = overlap_loss(
                                data['centers'],
                                my_centers.quantile_bounds, 
                                linear_penalty_weight,
                                mode=overlap_mode,
                                cov_list=data['cov_list'], 
                                ave_cov_inv_list=data['ave_cov_inv_list'])
            
            assert returned_loss == desired_loss
            if returned_loss == 0.0: zero_loss_count += 1

        # make sure we did not always have zero loss
        assert zero_loss_count/n_runs <= 0.5

        # For easy cases, verify that loss goes to zero
        fail_count = 0; total_runs = 20
        for random_data in range(total_runs):
            linear_penalty_weight = [0,0.33,0.61,1][int(random_data % 2)]
            overlap_mode = ['c2c', 'lda'][int(random_data % 2)]
            data = make_random_clusters(overlap_mode=overlap_mode)
            my_centers = ConstrainedOverlapCenters(
                            max_epoch=100,
                            overlap_mode=overlap_mode,
                            max_overlap=data['max_overlap'],
                            min_overlap=data['min_overlap']
                            )
            final_loss = my_centers._optimize_centers(
                            data['centers'], data['cov_list'],
                            ave_cov_inv_list=data['ave_cov_inv_list'],
                            axis_deriv_t_list=data['axis_deriv_t_list'],
                            quiet=False, progress_bar=None
                            )
            # keep track of failures
            if ((linear_penalty_weight > 0 and final_loss > 0)
                or final_loss > my_centers.ATOL):
                fail_count += 1

        # assess overall failures
        assert fail_count/total_runs <= 0.1
 

class TestExposedInterface:
    def test_sample_cluster_centers(self):
        # measure the attained overlaps after sampling cluster centers
        fail_count = 0; n_runs = 100
        for random_data in range(n_runs):
            overlap_mode = ['lda', 'c2c'][int(random_data % 2)]
            n_clusters = np.random.choice(np.arange(2,31))
            max_overlap = [1e-3,1e-2,1e-1,0.2][int(random_data % 4)]
            min_overlap = max_overlap/2
            my_centers = ConstrainedOverlapCenters(
                                max_epoch=200,
                                n_restarts=10,
                                learning_rate=0.6,
                                linear_penalty_weight=0.3,
                                overlap_mode=overlap_mode,
                                max_overlap=max_overlap,
                                min_overlap=min_overlap)
            archetype = Archetype(n_clusters=n_clusters, dim=2)
            axes_list, axis_lengths_list = (archetype
                                            .covariance_sampler
                                            .sample_covariances(
                                                archetype))
            archetype._axes = axes_list
            archetype._lengths = axis_lengths_list
            centers = my_centers.sample_cluster_centers(archetype, 
                                                        quiet=True)
            
            cov_list = [ assemble_covariance_matrix(
                            axes_list[i], axis_lengths_list[i])
                        for i in range(centers.shape[0]) ]
            ave_cov_inv_list = (
                None if overlap_mode=='c2c'
                    else [np.linalg.inv((cov_list[i] + cov_list[j])/2)
                            for i in range(centers.shape[0])
                                for j in range(i+1,centers.shape[0])])
            obs_overlap = assess_obs_overlap(
                                centers=centers, 
                                cov_list=cov_list,
                                ave_cov_inv_list=ave_cov_inv_list, 
                                mode=overlap_mode)
            if ((obs_overlap['min'] < min_overlap) or 
                (obs_overlap['max'] > max_overlap)):
                fail_count += 1

        # Assert that overall, the failure is at most 5%
        assert fail_count/n_runs <= 0.05

        # Make sure we can sample single clusters with no problems
        singlecluster_archetype = Archetype(n_clusters=1, dim=2)
        axes_list, axis_lengths_list = (singlecluster_archetype
                                            .covariance_sampler
                                            .sample_covariances(
                                                singlecluster_archetype))
        archetype._axes = axes_list
        archetype._lengths = axis_lengths_list
        centers = my_centers.sample_cluster_centers(
                                singlecluster_archetype, quiet=True)
        assert np.allclose(centers, np.zeros(centers.shape))