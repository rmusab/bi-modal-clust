# BiModalClust: Fused Data and Neighborhood Variation for Advanced K-means Big Data Clustering
# Original paper:
# ---

# BiModalClust parameters:
# points : The input dataset.
# n_centers : The desired number of clusters.
# sample_size : The number of data points to be randomly selected from the input dataset at each iteration of the HPClust.
# max_iter : The maximum number of samples to be processed.
# t_max : The time limit for the search process (in seconds); a zero or negative value means no limit.
# local_max_iters : The maximum number of K-means iterations before declaring convergence and stopping the clustering process for each sample.
# local_tol : The threshold below which the relative change in the objective function between two iterations must fall to declare convergence of K-means.
# n_candidates : The number of candidate centers to choose from at each stage of the K-means++ initialization algorithm.

import math
import time

import numpy as np
import numba as nb
from numba import njit, prange, objmode


def normalization(X):
    X_min = np.amin(X, axis=0)
    X = X - X_min
    X_max = np.amax(X, axis=0)
    if X_max.ndim == 1:
        X_max[X_max == 0.0] = 1.0
    elif X_max.ndim == 0:
        if X_max == 0.0:
            X_max = 1.0
    else:
        X_max = 1.0
    X = X / X_max
    return X


@njit
def empty_state(n_samples, n_features, n_clusters):
    sample_membership = np.full(n_samples, -1)
    centroids = np.full((n_clusters, n_features), np.nan)
    centroid_sums = np.full((n_clusters, n_features), np.nan)
    centroid_counts = np.full(n_clusters, 0.0)
    return sample_membership, centroids, centroid_sums, centroid_counts


@njit(parallel = False)
def distance_mat(X,Y):
    out = np.dot(X, Y.T)
    NX = np.sum(X*X, axis=1)
    NY = np.sum(Y*Y, axis=1)
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            out[i,j] = NX[i] - 2.*out[i,j] + NY[j]
    return out


@njit(parallel = True)
def distance_mat_parallel(X, Y):
    X_rows, X_cols = X.shape
    Y_rows, Y_cols = Y.shape
    out = np.zeros((X_rows, Y_rows))
    NX = np.sum(X*X, axis=1)
    NY = np.zeros(Y_rows)
    for i in prange(Y_rows):
        for j in range(Y_cols):
            NY[i] += Y[i, j] * Y[i, j]        
        for j in range(X_rows):        
            for k in range(X_cols):
                out[j, i] += X[j, k] * Y[i, k]
            out[j, i] = NX[j] - 2 * out[j, i] + NY[i]
    return out


@njit(parallel = False)
def kmeans_plus_plus(points, centers, n_new_centers=2, n_candidates=6):
    n_points, n_features = points.shape
    n_centers = centers.shape[0]
    n_dists = 0
    center_inds = np.full(n_new_centers, -1)
    if n_points > 0 and n_features > 0 and n_new_centers > 1 and n_candidates > 0:
        if n_centers == 0:
            center_inds[0] = np.random.randint(n_points)
            closest_dist_sq = distance_mat(points[center_inds[0:1]], points)[0]
            n_dists += n_points
            n_added_centers = 1
        else:           
            dist_mat = distance_mat(centers, points)
            n_dists += n_centers * n_points
            closest_dist_sq = np.empty(n_points)
            for j in range(n_points):
                min_dist = dist_mat[0, j]
                for i in range(1, n_centers):
                    if dist_mat[i, j] < min_dist:
                        min_dist = dist_mat[i, j]
                closest_dist_sq[j] = min_dist
            n_added_centers = 0
        current_pot = np.sum(closest_dist_sq)
        for c in range(n_added_centers, n_new_centers):
            rand_vals = np.random.random_sample(n_candidates) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
            dists = distance_mat(points[candidate_ids], points)
            n_dists += dists.size
            dists = np.minimum(dists, closest_dist_sq)
            candidates_pot = np.sum(dists, axis=1)
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = dists[best_candidate]
            center_inds[c] = candidate_ids[best_candidate]
    return center_inds, n_dists


@njit(parallel = True)
def kmeans_plus_plus_parallel(points, centers, n_new_centers=2, n_candidates=6):
    n_points, n_features = points.shape
    n_centers = centers.shape[0]
    n_dists = 0
    center_inds = np.full(n_new_centers, -1)
    if n_points > 0 and n_features > 0 and n_new_centers > 1 and n_candidates > 0:
        if n_centers == 0:
            center_inds[0] = np.random.randint(n_points)
            closest_dist_sq = distance_mat_parallel(points[center_inds[0:1]], points)[0]
            n_dists += n_points
            n_added_centers = 1
        else:           
            dist_mat = distance_mat_parallel(centers, points)
            n_dists += n_centers * n_points
            closest_dist_sq = np.empty(n_points)
            for j in prange(n_points):
                min_dist = dist_mat[0, j]
                for i in range(1, n_centers):
                    if dist_mat[i, j] < min_dist:
                        min_dist = dist_mat[i, j]
                closest_dist_sq[j] = min_dist
            n_added_centers = 0
        current_pot = np.sum(closest_dist_sq)
        for c in range(n_added_centers, n_new_centers):
            rand_vals = np.random.random_sample(n_candidates) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
            dists = distance_mat_parallel(points[candidate_ids], points)
            n_dists += dists.size
            dists = np.minimum(dists, closest_dist_sq)
            candidates_pot = np.sum(dists, axis=1)
            best_candidate = np.argmin(candidates_pot)
            current_pot = candidates_pot[best_candidate]
            closest_dist_sq = dists[best_candidate]
            center_inds[c] = candidate_ids[best_candidate]
    return center_inds, n_dists


@njit(parallel = False)
def kmeans(points, centers, max_iters = -1, tol=0.0, use_inner_product = True):
    def dist2(point1, point2):
        if use_inner_product:
            s1 = s2 = s3 = 0.0
            for i in range(point1.shape[0]):
                s1 += point1[i]*point1[i]
                s2 += point1[i]*point2[i]
                s3 += point2[i]*point2[i]
            return s1 - 2*s2 + s3            
        else:
            d = 0.0
            for i in range(point1.shape[0]):
                d += (point1[i]-point2[i])**2
            return d   
    assert points.ndim == 2
    m, n = points.shape  
    assert (centers.ndim == 2) and (centers.shape[1] == n)
    k = centers.shape[0]
    assignment = np.full(m, -1)
    center_sums = np.empty((k, n))
    center_counts = np.zeros(k)
    f = np.inf
    n_iters = 0
    if (m > 0) and (n > 0) and (k > 0):
        objective_previous = np.inf
        tolerance = np.inf
        while True:            
            f = 0.0 # assignment step
            n_changed = 0
            for i in range(m):
                min_d = np.inf
                min_ind = -1
                for j in range(k):
                    d = dist2(points[i], centers[j])
                    if d < min_d:
                        min_d = d
                        min_ind = j
                if assignment[i] != min_ind:
                    n_changed += 1
                    assignment[i] = min_ind
                f += min_d
            n_iters += 1
            tolerance = 1 - f/objective_previous
            objective_previous = f
            
            if ((max_iters >= 0) and (n_iters >= max_iters)) or (n_changed == 0) or ((tol > 0.0) and (tolerance <= tol)):
                break
            for i in range(k): # update step
                center_counts[i] = 0.0
                for j in range(n):
                    center_sums[i,j] = 0.0
                    centers[i,j] = np.nan
            for i in range(m):
                center_ind = assignment[i]
                if center_ind > -1:
                    for j in range(n):
                        center_sums[center_ind,j] += points[i,j]
                    center_counts[center_ind] += 1.0                    
            for i in range(k):
                if center_counts[i] > 0.0:
                    for j in range(n):
                        centers[i,j] = center_sums[i,j] / center_counts[i]
    return f, n_iters, assignment, n_iters*k*m


@njit(parallel = True)
def kmeans_parallel(points, centers, max_iters = -1, tol=0.0, use_inner_product = True):
    def dist2(point1, point2):
        if use_inner_product:
            s1 = s2 = s3 = 0.0
            for i in range(point1.shape[0]):
                s1 += point1[i]*point1[i]
                s2 += point1[i]*point2[i]
                s3 += point2[i]*point2[i]
            return s1 - 2*s2 + s3            
        else:
            d = 0.0
            for i in range(point1.shape[0]):
                d += (point1[i]-point2[i])**2
            return d   
    assert points.ndim == 2
    m, n = points.shape  
    assert (centers.ndim == 2) and (centers.shape[1] == n)
    k = centers.shape[0]
    assignment = np.full(m, -1)
    center_sums = np.empty((k, n))
    center_counts = np.zeros(k)
    f = np.inf
    n_iters = 0
    if (m > 0) and (n > 0) and (k > 0):
        objective_previous = np.inf
        tolerance = np.inf
        while True:            
            f = 0.0 # assignment step
            n_changed = 0
            for i in prange(m):
                min_d = np.inf
                min_ind = -1
                for j in range(k):
                    d = dist2(points[i], centers[j])
                    if d < min_d:
                        min_d = d
                        min_ind = j
                if assignment[i] != min_ind:
                    n_changed += 1
                    assignment[i] = min_ind
                f += min_d
            n_iters += 1
            tolerance = 1 - f/objective_previous
            objective_previous = f
            
            if ((max_iters >= 0) and (n_iters >= max_iters)) or (n_changed == 0) or ((tol > 0.0) and (tolerance <= tol)):
                break
            for i in range(k): # update step
                center_counts[i] = 0.0
                for j in range(n):
                    center_sums[i,j] = 0.0
                    centers[i,j] = np.nan
            for i in range(m):
                center_ind = assignment[i]
                if center_ind > -1:
                    for j in range(n):
                        center_sums[center_ind,j] += points[i,j]
                    center_counts[center_ind] += 1.0                    
            for i in range(k):
                if center_counts[i] > 0.0:
                    for j in range(n):
                        centers[i,j] = center_sums[i,j] / center_counts[i]
    return f, n_iters, assignment, n_iters*k*m


@njit(parallel = True)
def bi_modal_clust(X, k=3, s=100, p_max=5, max_iter=10000, t_max=10, init_mode=1, local_max_iters=300, local_tol=0.0001, n_candidates=3, printing=False):
    m, n = X.shape
    assert s <= m, "Sample size cannot exceed the size of the entire dataset!"
    if printing:
        with objmode:
            print ('%-30s%-15s%-15s' % ('objective', 'n_iter', 'cpu_time'))
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
    cpu_time = 0.0
    
    membership = np.full(s, -1)
    centsums = np.empty((k, n))
    centnums = np.zeros(k)
    weights = np.empty(0, dtype=np.float64)  # no weights
    
    centroids = np.full((k, n), np.nan)
    objective = np.inf
    n_dists = 0
    n_iter = 0
    best_time = 0.0
    best_n_dists = 0
    best_n_iter = 0
    p = 1  # shaking power
    
    while (n_iter < max_iter) and (cpu_time < t_max):
        sample = X[np.random.choice(m, s, replace=False)]
        centers = np.copy(centroids)
        
        degenerate_mask = np.sum(np.isnan(centers), axis = 1) > 0
        # Reinitialize all degenerate cluster centers
        n_degenerate = np.sum(degenerate_mask)
        if n_degenerate > 0:
            if init_mode == 0:
                new_centers = np.random.choice(s, n_degenerate, replace=False)
            else:
                new_centers, n_d = kmeanspp(sample, centers, n_degenerate, n_candidates)
                n_dists += n_d
            centers[degenerate_mask, :] = sample[new_centers, :]
        # Reinitialize some centers up to shaking power
        shaken_mask = np.random.choice(k, min(p, k), replace=False)
        if init_mode == 0:
            new_centers = np.random.choice(s, min(p, s), replace=False)
        else:
            new_centers, n_d = kmeanspp(sample, centers, min(p, k), n_candidates)
            n_dists += n_d
        centers[shaken_mask, :] = sample[new_centers, :]
        
        # Local search phase
        obj, n_it = k_means(sample, weights, membership, centers, centsums, centnums, local_max_iters, local_tol, True)
        n_dists += n_it*s*k
        
        with objmode(cpu_time = 'float64'):
            cpu_time = time.perf_counter() - start_time
        
        # Update incumbent solution
        n_iter += 1
        if obj < objective:
            objective = obj
            centroids = np.copy(centers)
            if printing:
                with objmode:
                    print ('%-30f%-15i%-15.2f' % (objective, n_iter, cpu_time))
            best_time = cpu_time
            best_n_dists = n_dists
            best_n_iter = n_iter
        
        p += 1
        if p > p_max:
            p = 1
            
    return centroids, objective, best_time, best_n_iter, best_n_dists


# BiModalClust with 'Hybrid Parallelism (competitive + collective)': 
# This parallelization approach involves two consecutive phases: competitive and collective.
# During the first phase, each worker tries independently to obtain its own best solution. 
# Then, during the second phase, the workers begin sharing information about the best solutions 
# with each other and try to improve them. 
# Finally, the best solution among all workers is selected as the final result.
# Additional Parameters:
# p_max: Maximum shaking power in the VNS metaheuristic.
# init_mode: Reinitialization mode used in the shaking procedure (0 => random, 1 => K-means++).
# max_iter1 : Maximum number of samples to be processed for the first phase.
# max_iter2 : Maximum number of samples to be processed for the second phase.
# t_max1 : The time limit for the first phase (in seconds).
# t_max2 : The time limit for the second phase (in seconds).
@njit(parallel = True)
def bi_modal_clust_hybrid(points, n_centers=3, sample_size=100, p_max=5, init_mode=1, max_iter1=10000, max_iter2=10000, t_max1=10.0, t_max2=10.0, local_max_iters=300, local_tol=0.0001, n_candidates=3, printing=False):
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
    
    n_points, n_features = points.shape
    n_threads = nb.get_num_threads()
    assert sample_size <= n_points
    assert max_iter1 > 0 and max_iter2 > 0 and t_max1 > 0.0 and t_max2 > 0.0
    if printing:
        with objmode:
            print ('%-30s%-15s%-15s' % ('sample objective', 'n_iter', 'cpu_time'))

    centers = np.full((n_threads, n_centers, n_features), np.nan)
    objectives = np.full((n_threads, max_iter1 + max_iter2), np.inf)
    best_objectives = np.full(n_threads, np.inf)
    n_dists = np.full(n_threads, 0)
    n_iters = np.full(n_threads, 0)
    running_time = np.full(n_threads, 0.0)
    objective_times = np.full((n_threads, max_iter1 + max_iter2), 0.0)
    best_times = np.full(n_threads, 0.0)
    best_n_iters = np.full(n_threads, 0)
    ps = np.full(n_threads, 1)  # shaking power per worker
    
    for t in prange(n_threads):
        while (np.sum(n_iters) < max_iter1) and (running_time[t] < t_max1):        
            sample = points[np.random.choice(n_points, sample_size, replace=False)]
            best = np.argmin(best_objectives)
            best_objective = best_objectives[best]
            new_centers = centers[t].copy()
            p = ps[t]
            # Reinitialize all degenerate centroids
            degenerate_mask = np.sum(np.isnan(new_centers), axis = 1) > 0
            n_degenerate = np.sum(degenerate_mask)
            if n_degenerate > 0:
                center_inds, num_dists = kmeans_plus_plus(sample, new_centers[~degenerate_mask], n_degenerate, n_candidates)
                n_dists[t] += num_dists
                new_centers[degenerate_mask,:] = sample[center_inds,:]
            # Reinitialize some centers up to the shaking power
            # shaken_mask = np.random.choice(n_centers, min(p, n_centers), replace=False)
            shaken_mask = np.array([True] * min(p, n_centers) + [False] * (n_centers - min(p, n_centers)))[np.random.permutation(n_centers)]
            if init_mode == 0:
                center_inds = np.random.choice(sample_size, min(p, n_centers), replace=False)
            else:
                center_inds, num_dists = kmeans_plus_plus(sample, new_centers[~shaken_mask], min(p, n_centers), n_candidates)
                # center_inds, num_dists = kmeans_plus_plus(sample, new_centers, min(p, n_centers), n_candidates)
                n_dists[t] += num_dists
            new_centers[shaken_mask,:] = sample[center_inds,:]
            # Local search phase
            new_objective, _, _, num_dists = kmeans(sample, new_centers, local_max_iters, local_tol, True)
            n_dists[t] += num_dists
            with objmode(time_now = 'float64'):
                time_now = time.perf_counter() - start_time
            running_time[t] = time_now
            n_iters[t] += 1
            # Neighborhood change step
            if new_objective < best_objectives[t]:
                objectives[t, n_iters[t] - 1] = new_objective
                best_objectives[t] = new_objective
                centers[t] = new_centers.copy()
                objective_times[t, n_iters[t] - 1] = time_now
                best_times[t] = time_now
                best_n_iters[t] = np.sum(n_iters)
                if printing:
                    if new_objective < best_objective:
                        with objmode:
                            print ('%-30f%-15i%-15.2f' % (new_objective, best_n_iters[t], time_now))
                ps[t] = 1
            else:
                ps[t] = max(ps[t] + 1, p_max)
            # ps[t] += 1
            # if ps[t] > p_max:
            #     ps[t] = 1
                            
    for t in prange(n_threads):
        while (np.sum(n_iters) < max_iter1 + max_iter2) and (running_time[t] < t_max1 + t_max2):        
            sample = points[np.random.choice(n_points, sample_size, replace=False)]
            best = np.argmin(best_objectives)
            best_objective = best_objectives[best]
            new_centers = centers[best].copy()
            p = ps[t]
            # Reinitialize all degenerate centroids
            degenerate_mask = np.sum(np.isnan(new_centers), axis = 1) > 0
            n_degenerate = np.sum(degenerate_mask)
            if n_degenerate > 0:
                center_inds, num_dists = kmeans_plus_plus(sample, new_centers[~degenerate_mask], n_degenerate, n_candidates)
                n_dists[t] += num_dists
                new_centers[degenerate_mask,:] = sample[center_inds,:]
            # Reinitialize some centers up to the shaking power
            # shaken_mask = np.random.choice(n_centers, min(p, n_centers), replace=False)
            shaken_mask = np.array([True] * min(p, n_centers) + [False] * (n_centers - min(p, n_centers)))[np.random.permutation(n_centers)]
            if init_mode == 0:
                center_inds = np.random.choice(sample_size, min(p, n_centers), replace=False)
            else:
                center_inds, num_dists = kmeans_plus_plus(sample, new_centers[~shaken_mask], min(p, n_centers), n_candidates)
                # center_inds, num_dists = kmeans_plus_plus(sample, new_centers, min(p, n_centers), n_candidates)
                n_dists[t] += num_dists
            new_centers[shaken_mask,:] = sample[center_inds,:]
            # Local search phase
            new_objective, _, _, num_dists = kmeans(sample, new_centers, local_max_iters, local_tol, True)
            n_dists[t] += num_dists
            with objmode(time_now = 'float64'):
                time_now = time.perf_counter() - start_time
            running_time[t] = time_now
            n_iters[t] += 1
            # Neighborhood change step
            if new_objective < best_objective:
                objectives[t, n_iters[t] - 1] = new_objective
                best_objectives[t] = new_objective
                centers[t] = new_centers.copy()
                objective_times[t, n_iters[t] - 1] = time_now
                best_times[t] = time_now
                best_n_iters[t] = np.sum(n_iters)
                if printing:
                    with objmode:
                        print ('%-30f%-15i%-15.2f' % (new_objective, best_n_iters[t], time_now))
                ps[t] = 1
            else:
                ps[t] = max(ps[t] + 1, p_max)
            # ps[t] += 1
            # if ps[t] > p_max:
            #     ps[t] = 1

    best_ind = np.argmin(best_objectives)
    final_centers = centers[best_ind].copy()

    # When 'max_iters = 0' is used for K-means, only the assignment step will be performed
    full_objective, _, assignment, full_num_dists = kmeans_parallel(points, final_centers, 0, 0.0, True)

    return final_centers, full_objective, assignment, np.sum(n_iters), best_n_iters[best_ind], best_times[best_ind], np.sum(n_dists)+full_num_dists, objectives, objective_times


# BiModalClust with "Inner Parallelism":
# Separate data samples are clustered sequentially one-by-one, but the clustering process itself 
# is parallelized on the level of internal implementation of the K-means and K-means++ functions.
@njit(parallel = True)
def big_vns_clust_inner(points, n_centers = 3, sample_size = 100, p_max=5, init_mode=1, max_iter = 10000, t_max = 10.0, local_max_iters=300, local_tol=0.0001, n_candidates = 3, printing=False):
    n_points, n_features = points.shape
    assert sample_size <= n_points
    if printing:
        with objmode:
            print ('%-30s%-15s%-15s' % ('sample objective', 'n_iter', 'cpu_time'))
    with objmode(start_time = 'float64'):
        start_time = time.perf_counter()
    cpu_time = 0.0

    centers = np.full((n_centers, n_features), np.nan)
    objectives = np.full(max_iter, np.inf)
    objective = np.inf
    n_dists = 0
    n_iter = 0
    objective_times = np.full(max_iter, 0.0)
    best_time = 0.0
    best_n_iter = 0
    p = 1  # shaking power
    while (n_iter < max_iter or max_iter <= 0) and (cpu_time < t_max or t_max <= 0.0):
        sample = points[np.random.choice(n_points, sample_size, replace=False)]
        new_centers = np.copy(centers)
        degenerate_mask = np.sum(np.isnan(new_centers), axis = 1) > 0
        n_degenerate = np.sum(degenerate_mask)
        # Reinitialize all degenerate centroids
        if n_degenerate > 0:
            center_inds, num_dists = kmeans_plus_plus_parallel(sample, new_centers[~degenerate_mask], n_degenerate, n_candidates)
            n_dists += num_dists
            new_centers[degenerate_mask,:] = sample[center_inds,:]
        # Reinitialize some centers up to the shaking power
        shaken_mask = np.array([True] * min(p, n_centers) + [False] * (n_centers - min(p, n_centers)))[np.random.permutation(n_centers)]
        if init_mode == 1:
            center_inds, num_dists = kmeans_plus_plus(sample, new_centers[~shaken_mask], min(p, n_centers), n_candidates)
            n_dists += num_dists
        else:
            center_inds = np.random.choice(sample_size, min(p, n_centers), replace=False)
        new_centers[shaken_mask,:] = sample[center_inds,:]
        # Local search phase
        new_objective, _, _, num_dists = kmeans_parallel(sample, new_centers, local_max_iters, local_tol, True)
        n_dists += num_dists
        with objmode(cpu_time = 'float64'):
            cpu_time = time.perf_counter() - start_time
        n_iter += 1
        # Neighborhood change step
        if new_objective < objective:
            objectives[n_iter - 1] = new_objective
            objective = new_objective
            centers = np.copy(new_centers)
            if printing:
                with objmode:
                    print ('%-30f%-15i%-15.2f' % (objective, n_iter, cpu_time))
            objective_times[n_iter - 1] = cpu_time
            best_time = cpu_time
            best_n_iter = n_iter
        p += 1
        if p > p_max:
            p = 1
       
    # When 'max_iters = 0' is used for kmeans, only the assignment step will be performed
    full_objective, _, assignment, num_dists = kmeans_parallel(points, centers, 0, 0.0, True)
    n_dists += num_dists
    return centers, full_objective, assignment, n_iter, best_n_iter, best_time, n_dists, objectives, objective_times