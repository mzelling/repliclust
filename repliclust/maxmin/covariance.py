"""
This module provides a class for sampling cluster covariances using
the max-min approach.
"""

import numpy as np

from repliclust import config
from repliclust.utils import make_orthonormal_axes
from repliclust.maxmin.utils import sample_with_maxmin
from repliclust.base import CovarianceSampler

class MaxMinCovarianceSampler():
    """
    Sample covariances for the clusters in a mixture model by specifying
    the max-min ratios for various geometric parameters.

    See documentation of class MaxMinArchetype for more information.

    Attributes
    ----------
    aspect_ref : float, >= 1            
        Reference aspect ratio for clusters in the mixture model.
    aspect_maxmin : float, >= 1
        Max-min ratio for the aspect ratios of clusters in the mixture 
        model.
    radius_maxmin : float, >= 1
        Max-min ratio for the radii of clusters in the mixture model.

    """
    
    def __init__(self, aspect_ref=1.5, aspect_maxmin=2, 
                 radius_maxmin=2):
        """ Instantiate a MaxMinCovarianceSampler object. """
        if (np.min([aspect_ref,aspect_maxmin,radius_maxmin]) < 1):
            raise ValueError('aspect ratio and max-min ratios must be'
                                + ' >= 1.')
        else:
            self.aspect_ref = aspect_ref
            self.aspect_maxmin = aspect_maxmin
            self.radius_maxmin = radius_maxmin
    

    def make_cluster_aspect_ratios(self, n_clusters):
        """
        Sample aspect ratios for all clusters. 
        
        The aspect ratio of a cluster measures how oblong/ellipsoidal 
        the cluster is. It is defined as the ratio between the lengths
        of the cluster's longest and shortest principal axes.

        Parameters
        ----------
        n_clusters : int
            The number of clusters.

        Returns
        -------
        out : ndarray
            The aspect ratios for each cluster.
        """
        n_clusters = self.validate_k(n_clusters)

        ref = self.aspect_ref
        mm = self.aspect_maxmin

        delta = (-(1+mm) + np.sqrt((1+mm)**2 + 4*mm*(ref**2-1)))/2
        max_aspect = 1 + delta
        min_aspect = 1 + (delta/mm)
        # print('ref aspect:', ref, '| nominal mm:', mm, '| effective mm:', 
        #         max_aspect/min_aspect)

        # ref = self.aspect_ref
        # min_aspect = 1 + (ref-1)/self.aspect_maxmin
        # max_aspect = min_aspect*self.aspect_maxmin
        # maxmin_ratio = max_aspect / min_aspect

        # TOL = 0.05
        # def compute_other_aspect_ratio(a):
        #     if (ref - min_aspect < TOL):
        #         return (config._rng.triangular(
        #                         left=ref, mode=ref,right=max_aspect
        #                         )
        #                 if (a <= ref) 
        #                 else (ref + min_aspect)/2)
        #     else:
        #         return ((ref + ((ref - a)/(ref-min_aspect))
        #                             * (max_aspect-ref))
        #                     if (a <= ref)
        #                     else (ref + ((a - ref)/(max_aspect-ref))
        #                                     * (min_aspect - ref))
        #                     )

        # result = sample_with_maxmin(n_clusters, 
        #                             ref, min_aspect, maxmin_ratio, 
        #                             compute_other_aspect_ratio)

        result = sample_with_maxmin(n_clusters, ref, min_aspect,
                                    max_aspect/min_aspect,
                                    lambda a: (ref**2)/a)

        return result


    def validate_k(self, n_clusters):
        """
        Make sure the number of clusters is valid.
        """
        if (n_clusters < 1):
            raise ValueError('number of clusters must be >= 1.')
        else:
            return int(n_clusters)

        
    def make_cluster_radii(self, n_clusters, ref_radius, dim):
        """ 
        Sample cluster radii using pairwise max-min sampling.

        Sampling constrains the arithmetic mean of cluster volumes to
        equal the reference volume (namely ref_radius**dim power). 
        The minimum and maximum cluster radii of the resulting sample
        average to the reference radius.

        Parameters
        ----------
        n_clusters : int
            The number of clusters.
        ref_radius : float
            The reference radius for the clusters.
        dim : int
            The number of dimensions.

        Returns
        -------
        radii : ndarray
            Radii for all the clusters.
        """
        n_clusters = self.validate_k(n_clusters)
        log_min_radius = (np.log(ref_radius) 
                            - np.log(self.radius_maxmin)/2)
        f_constraint = lambda log_r: 2*np.log(ref_radius) - log_r
        log_max_radius = (np.log(ref_radius) 
                            + np.log(self.radius_maxmin)/2)
                            
        if (self.radius_maxmin == 1):
            maxmin_log_ratio = 1
        else:
            maxmin_log_ratio = log_max_radius/log_min_radius

        return np.exp(sample_with_maxmin(
                        n_clusters, np.log(ref_radius), log_min_radius,
                        maxmin_log_ratio, f_constraint))
    

    def make_axis_lengths(self, n_axes, reference_length, aspect_ratio):
        """
        Sample the lengths of all principal axes for a single cluster.

        Parameters
        ----------
        n_axes : int
            The number of principal axes (same as the dimensionality).
        reference_length : float
            Desired geometric mean of the lengths.
        aspect_ratio : float
            Desired ratio between longest and shortest lengths.

        Returns
        -------
        lengths : ndarray
            Lengths of the principal axes for this cluster.
        """
        if ((n_axes < 1) or (reference_length <= 0) 
                or (aspect_ratio < 1)):
            raise ValueError('number of axes must be >= 1, and'
                    + ' reference length must be > 0, and aspect ratio'
                    + ' must be >= 1.')
        else: 
            n_axes = int(n_axes)

        min_length = reference_length/np.sqrt(aspect_ratio)
        f_constraint = lambda s: (reference_length**2)/s
        return sample_with_maxmin(
                    n_axes, reference_length, min_length, 
                    aspect_ratio, f_constraint
                    )
        

    def sample_covariances(self, archetype):
        """
        Compute the principal axes and their lengths for each cluster
        in a mixture model.

        Parameters
        ----------
        archetype : Archetype
            Archetype for a mixture model.

        Returns
        -------
        (axes_list, axis_lengths_list) : Tuple[List[ndarray], List[ndarray]]
            Tuple with two components. The first component, `axes_list`,
            is a list whose `i`-th element stores the principal axes
            of the `i`-th cluster as a matrix (each row is an 
            axis). The second component, `axis_lengths_list`, is a
            list whose i-th element stores the lengths of the i-th 
            cluster's principal axes as a vector. In particular, for
            any cluster `i` and axis `j`, the number 
            `axis_lengths_list[i][j]` is the length corresponding
            to the principal axis `axes_list[i][j,:]`.
        """
        axes_list = list()
        axis_lengths_list = list()

        n_clusters = archetype.n_clusters
        dim = archetype.dim
        scale = archetype.scale
        
        cluster_radii = self.make_cluster_radii(n_clusters, scale, dim)
        cluster_aspects = self.make_cluster_aspect_ratios(n_clusters)
        
        # For each cluster, sample principal axes and their lengths.
        for clust in range(n_clusters):
            axes = make_orthonormal_axes(dim, dim)
            axis_lengths = self.make_axis_lengths(dim, 
                                                  cluster_radii[clust], 
                                                  cluster_aspects[clust]
                                                  )
            axes_list.append(axes)
            axis_lengths_list.append(axis_lengths)

        return (axes_list, axis_lengths_list)
