# cython: embedsignature=True, profile=True

from libc.math cimport log, sqrt
from libc.stdlib cimport rand, srand
from libc.time cimport time, time_t
import numpy as np
cimport numpy as np
cimport cython
from cython.view cimport array

srand(time(<time_t*>0))
def seed(int value):
    srand(value)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float[:] _empty1f(int n):
    return array(shape=(n,), itemsize=sizeof(float), format=b"f")

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int[:] _empty1i(int n):
    return array(shape=(n,), itemsize=sizeof(int), format=b"i")

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float[:] _cross3(const float[:] p0, const float[:] p1, const float[:] p2):
    cdef float a0 = p1[0] - p0[0], a1 = p1[1] - p0[1], a2 = p1[2] - p0[2]
    cdef float b0 = p2[0] - p0[0], b1 = p2[1] - p0[1], b2 = p2[2] - p0[2]

    cdef float[:] result = _empty1f(3)
    result[0] = a1*b2 - a2*b1
    result[1] = a2*b0 - a0*b2
    result[2] = a0*b1 - a1*b0
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float _dot3(const float[:] a, const float[:] b) nogil:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float _norm3(const float[:] elems) nogil:
    return sqrt(_dot3(elems, elems))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float _div3(float[:] arr, const float div) nogil:
    arr[0] = arr[0] / div
    arr[1] = arr[1] / div
    arr[2] = arr[2] / div

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int[:] _choice3(int total_size, int choice_size): # without replace
    cdef int[:] result = _empty1i(choice_size)
    cdef bint duplicate = False

    for i in range(choice_size):
        while True:
            result[i] = rand() % total_size

            duplicate = False
            for j in range(i):
                if result[j] == result[i]:
                    duplicate = True
                    break
            if not duplicate:
                break

    return result

@cython.cdivision(True)
def plane_ransac(np.ndarray[float, ndim=2] data, int max_iter = 100, float max_err = 0.1, float p = 0):
    # constants
    cdef int sample_size = 3
    cdef float eps = 1e-6

    # previous declaration
    cdef float log_probability = log(1 - p)
    cdef int ninliers, niter = 0, k = max_iter
    cdef float d, norm
    cdef int[:] samples_idx
    cdef float[:] p0, p1, p2, abc
    cdef np.ndarray samples, err, inliers

    cdef int best_count = 0, 
    cdef tuple best_coeff = ([0, 0, 0], float('inf'))
    cdef np.ndarray best_inliers = np.zeros(len(data), dtype=bool)
    while True:
        # sample data
        samples_idx = _choice3(data.shape[0], sample_size)

        # calculate plane coefficients
        p0 = data[samples_idx[0], :3]
        p1 = data[samples_idx[1], :3]
        p2 = data[samples_idx[2], :3]

        abc = _cross3(p0, p1, p2)
        norm = _norm3(abc)
        if norm < eps:
            continue

        _div3(abc, norm)
        d = -_dot3(abc, p0)

        # count inliers
        err = np.abs(data[:,:3].dot(np.asarray(abc)) + d)
        inliers = err < max_err
        ninliers = np.sum(inliers)
        if ninliers > best_count:
            best_count = ninliers
            best_inliers = inliers
            best_coeff = (abc, d)

            # update k
            if p > 0:
                p_no_outliers = 1 - (float(best_count) / data.shape[0]) ** sample_size
                k = int(log_probability / log(min(max(p_no_outliers, eps), 1 - eps)))

        # exit conditions
        niter += 1
        if niter > max_iter:
            break
        if p > 0 and niter > k:
            break

    return best_inliers, best_coeff

def plane_ransac_batch(np.ndarray data, int max_iter = 100, float max_err = 0.1, float p=0, int batch_size=10000, int batch_count=10):
    cdef np.ndarray best_inliers = np.zeros(len(data), bool)
    cdef np.ndarray batch_indices, batch_data, inliers
    cdef int best_count = 0, ninliers

    for i in range(batch_count):
        batch_indices = np.random.choice(len(data), batch_size, replace=False)
        batch_data = data[batch_indices]
        _, (abc, d) = plane_ransac(batch_data, max_iter=max_iter, max_err=max_err, p=p)

        err = np.abs(data[:,:3].dot(abc) + d)
        inliers = err < max_err
        ninliers = np.sum(inliers)

        if ninliers > best_count:
            best_count = ninliers
            best_inliers = inliers
    
    return best_inliers
