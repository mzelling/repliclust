import pytest
import numpy as np


from repliclust import set_seed
from repliclust.base import Archetype
from repliclust.utils import assemble_covariance_matrix
from repliclust.random_centers import RandomCenters
from repliclust.maxmin.archetype import MaxMinArchetype
from repliclust.overlap.gradients import assess_obs_overlap
from repliclust.overlap.centers import \
    ConstrainedOverlapCenters

RTOL = 0.05

class TestConstrainedOverlapCenters:

    def test_init(self):
        centers = ConstrainedOverlapCenters(0.05,0.04,0.1)

    def test__optimize_centers(self):
        # run a few easy optimization cases; assert that loss goes to 0

        # CASES WITH SPHERICAL CLUSTERS
        for n_clusters, dim, max_overlap, min_overlap, packing in \
                [(6,2,0.01,0.005, 0.1), 
                (2,10,0.05,0.04,0.5), 
                (4,100,0.1,0.09,0.2)]:
            archetype = MaxMinArchetype(n_clusters=n_clusters, dim=dim,
                                        max_overlap=max_overlap, 
                                        min_overlap=min_overlap)
            cov_inv = [np.eye(archetype.dim) for 
                             i in range(archetype.n_clusters)]
            centers = np.random.multivariate_normal(
                            mean=np.zeros(archetype.dim),
                            cov=np.eye(archetype.dim),
                            size = archetype.n_clusters
                            )
            centers_sampler = ConstrainedOverlapCenters(max_overlap,
                                                        min_overlap,
                                                        packing)
            centers_opt = centers_sampler._optimize_centers(
                                centers, cov_inv=cov_inv, max_epoch=100,
                                learning_rate=0.1, verbose=False
                                )
            overlap_obs = assess_obs_overlap(centers_opt, cov_inv)
            assert overlap_obs['min'] >= (1-RTOL)*min_overlap
            assert overlap_obs['max'] <= (1+RTOL)*max_overlap

        # TEST ROBUSTNESS AGAINST DIFFERENT INIT CENTERS
        bp = MaxMinArchetype(n_clusters=13, dim=100,
                                        max_overlap=0.15,
                                        min_overlap=0.1,
                                        packing=0.1)
        axes, axis_lengths = (bp.covariance_sampler
                                .sample_covariances(bp))
        cov_inv = [assemble_covariance_matrix(axes[i],axis_lengths[i],
                                              inverse=True)
                    for i in range(bp.n_clusters)]
        center_sampler = ConstrainedOverlapCenters(
                                max_overlap=bp.max_overlap, 
                                min_overlap=bp.min_overlap, 
                                packing=packing)
        
        for i in range(100):
            centers_init = (RandomCenters(packing=0.1)
                            .sample_cluster_centers(bp))
            centers_opt = center_sampler._optimize_centers(
                            centers_init, cov_inv, learning_rate=0.1,
                            verbose=False)
            obs_overlap = assess_obs_overlap(centers_opt, cov_inv)
            assert obs_overlap['max'] <= (1+RTOL)*bp.max_overlap
            assert obs_overlap['min'] >= (1-RTOL)*bp.min_overlap

        # TEST ROBUSTNESS AGAINST DIFFERENT LEARNING RATES
        for eta in [0.01,0.2,0.5,0.9,1]:
            centers_init = (RandomCenters(packing=0.1)
                            .sample_cluster_centers(bp))
            centers_opt = center_sampler._optimize_centers(
                            centers_init, cov_inv, learning_rate=eta,
                            verbose=False)
            obs_overlap = assess_obs_overlap(centers_opt, cov_inv)
            # just check that low/high learning rates don't cause
            # numerical stability issues; don't require that loss
            # converges for low or high learning rates
            assert 0 < obs_overlap['min']
            assert obs_overlap['min'] <= obs_overlap['max']
            print(eta , 'min', obs_overlap['min'], 'max', 
                    obs_overlap['max'])
            assert obs_overlap['max'] < 1

        # TEST THAT SETTING A SEED WORKS
        centers = np.random.multivariate_normal(
                                mean=np.zeros(archetype.dim),
                                cov=np.eye(archetype.dim),
                                size = archetype.n_clusters
                                )
        set_seed(2)
        centers_opt_1 = centers_sampler._optimize_centers(
                                centers, cov_inv=cov_inv, max_epoch=100,
                                learning_rate=0.1, verbose=False
                                )
        centers_opt_2 = centers_sampler._optimize_centers(
                                centers, cov_inv=cov_inv, max_epoch=100,
                                learning_rate=0.1, verbose=False
                                )
        set_seed(2)
        centers_opt_3 = centers_sampler._optimize_centers(
                                centers, cov_inv=cov_inv, max_epoch=100,
                                learning_rate=0.1, verbose=False
                                )
        assert np.allclose(centers_opt_1, centers_opt_3)



    def test_sample_cluster_centers(self):
        # run a few easy optimization cases; assert that loss goes to 0
        for n_clusters, dim, max_overlap, min_overlap, packing in \
                [(6,2,0.01,0.005, 0.1), 
                 (3,10,0.50,0.49,0.9), 
                 (4,200,0.1,0.09,0.2),
                 (2,500,0.1,0.09,0.01)]:
            archetype = MaxMinArchetype(n_clusters=n_clusters, dim=dim,
                                        max_overlap=max_overlap,
                                        min_overlap=min_overlap,
                                        packing=packing)
            archetype.sample_mixture_model() # load covariance axes
            center_sampler = ConstrainedOverlapCenters(
                                max_overlap=max_overlap, 
                                min_overlap=min_overlap, 
                                packing=packing)
            centers = center_sampler.sample_cluster_centers(archetype,
                                        print_progress=False)
            cov_inv = [assemble_covariance_matrix(archetype._axes[i],
                                                  archetype._lengths[i],
                                                  inverse=True)
                            for i in range(archetype.n_clusters)]
            obs_overlap = assess_obs_overlap(centers, cov_inv)
            assert obs_overlap['max'] <= (1+RTOL)*max_overlap
            assert obs_overlap['min'] >= (1-RTOL)*min_overlap

        # test that setting a seed works
        set_seed(787)
        centers_1 = center_sampler.sample_cluster_centers(archetype,
                                        print_progress=False)
        centers_2 = center_sampler.sample_cluster_centers(archetype,
                                        print_progress=False)
        set_seed(787)
        centers_3 = center_sampler.sample_cluster_centers(archetype,
                                        print_progress=False)

        for i in range(archetype.n_clusters):
            assert (np.allclose(centers_1, centers_3) and 
                        not np.any(centers_1[i] == centers_2[i]))

        # test sampling a single cluster
        archetype_single_cluster = Archetype(n_clusters=1, dim=1000)
        center_sampler = ConstrainedOverlapCenters(
                                max_overlap=max_overlap, 
                                min_overlap=min_overlap, 
                                packing=packing)
        centers = center_sampler.sample_cluster_centers(
                                    archetype_single_cluster,
                                    print_progress=False)
        assert centers.shape == (1,1000)
        assert np.allclose(centers, np.zeros(1000)[np.newaxis,:])
        