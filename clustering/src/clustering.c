/* * This file is part of PyEMMA.
 *
 * Copyright (c) 2015, 2014 Computational Molecular Biology Group
 *
 * PyEMMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define NO_IMPORT_ARRAY
#include <clustering.h>
#include <assert.h>

#ifdef USE_OPENMP
    #include <omp.h>
#endif

float euclidean_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n, float *buffer_a, float *buffer_b)
{
    double sum;
    size_t i;

    sum = 0.0;
    for(i=0; i<n; ++i) {
        sum += (a[i]-b[i])*(a[i]-b[i]);
    }
    return sqrt(sum);
}

float minRMSD_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n, float *SKP_restrict buffer_a, float *SKP_restrict buffer_b)
{
    float msd;
    float trace_a, trace_b;

    memcpy(buffer_a, a, n*sizeof(float));
    memcpy(buffer_b, b, n*sizeof(float));

    inplace_center_and_trace_atom_major(buffer_a, &trace_a, 1, n/3);
    inplace_center_and_trace_atom_major(buffer_b, &trace_b, 1, n/3);
    msd = msd_atom_major(n/3, n/3, buffer_a, buffer_b, trace_a, trace_b, 0, NULL);
    return sqrt(msd);
}

int c_assign(float *chunk, float *centers, npy_int32 *dtraj, char* metric,
             Py_ssize_t N_frames, Py_ssize_t N_centers, Py_ssize_t dim, int n_threads) {
    int ret;
    float d, mindist;
    size_t argmin;
    float *buffer_a, *buffer_b;
    float (*distance)(float*, float*, size_t, float*, float*);

    buffer_a = NULL; buffer_b = NULL;
    ret = ASSIGN_SUCCESS;

    /* init metric */
    if(strcmp(metric,"euclidean")==0) {
        distance = euclidean_distance;
    } else if(strcmp(metric,"minRMSD")==0) {
        distance = minRMSD_distance;
        buffer_a = malloc(dim*sizeof(float));
        buffer_b = malloc(dim*sizeof(float));
        if(!buffer_a || !buffer_b) {
            ret = ASSIGN_ERR_NO_MEMORY; goto error;
        }
    } else {
        ret = ASSIGN_ERR_INVALID_METRIC;
        goto error;
    }

    /* Do the assignment in parallel with OpenMP. Each thread finds the minimum
     * distance for a couple of frames index by i.
     */
    #ifdef USE_OPENMP
    omp_set_num_threads(n_threads);
    #endif
    #pragma omp parallel
    {
        #ifdef USE_OPENMP
        printf("n threads in c_assign: %i\n", n_threads);
        assert(omp_get_num_threads() == n_threads);
        #endif

        int i, j;

        #pragma omp for private(i, j, argmin, mindist, d) schedule(static, 10)
        for(i = 0; i < N_frames; ++i) {
            mindist = FLT_MAX;
            argmin = -1;
            for(j = 0; j < N_centers; ++j) {
                d = distance(&chunk[i*dim], &centers[j*dim], dim, buffer_a, buffer_b);

            	{
                	if(d < mindist) { mindist = d; argmin = j; }
            	}
            }
            dtraj[i] = argmin;
        }
    }

error:
    free(buffer_a);
    free(buffer_b);
    return ret;
}

PyObject *assign(PyObject *self, PyObject *args) {

    PyObject *py_centers, *py_res;
    PyArrayObject *np_chunk, *np_centers, *np_dtraj;
    Py_ssize_t N_centers, N_frames, dim;
    float *chunk;
    float *centers;
    npy_int32 *dtraj;
    char *metric;
    int n_threads;

    py_centers = NULL; py_res = NULL;
    np_chunk = NULL; np_dtraj = NULL;
    centers = NULL; metric=""; chunk = NULL; dtraj = NULL; n_threads = -1;

    if (!PyArg_ParseTuple(args, "O!OO!si", &PyArray_Type, &np_chunk, &py_centers, &PyArray_Type, &np_dtraj, &metric, &n_threads)) goto error; /* ref:borr. */

    /* import chunk */
    if(PyArray_TYPE(np_chunk)!=NPY_FLOAT32) { PyErr_SetString(PyExc_ValueError, "dtype of \"chunk\" isn\'t float (32)."); goto error; };
    if(!PyArray_ISCARRAY_RO(np_chunk) ) { PyErr_SetString(PyExc_ValueError, "\"chunk\" isn\'t C-style contiguous or isn\'t behaved."); goto error; };
    if(PyArray_NDIM(np_chunk)!=2) { PyErr_SetString(PyExc_ValueError, "Number of dimensions of \"chunk\" isn\'t 2."); goto error;  };
    N_frames = np_chunk->dimensions[0];
    dim = np_chunk->dimensions[1];
    if(dim==0) {
        PyErr_SetString(PyExc_ValueError, "chunk dimension must be larger than zero.");
        goto error;
    }
    chunk = PyArray_DATA(np_chunk);

    /* import dtraj */
    if(PyArray_TYPE(np_dtraj)!=NPY_INT32) { PyErr_SetString(PyExc_ValueError, "dtype of \"dtraj\" isn\'t int (32)."); goto error; };
    if(!PyArray_ISBEHAVED_RO(np_dtraj) ) { PyErr_SetString(PyExc_ValueError, "\"dtraj\" isn\'t behaved."); goto error; };
    if(PyArray_NDIM(np_dtraj)!=1) { PyErr_SetString(PyExc_ValueError, "Number of dimensions of \"dtraj\" isn\'t 1."); goto error; };
    if(np_chunk->dimensions[0]!=N_frames) {
        PyErr_SetString(PyExc_ValueError, "Size of \"dtraj\" differs from number of frames in \"chunk\".");
        goto error;
    }
    dtraj = (npy_int32*)PyArray_DATA(np_dtraj);

    /* import list of cluster centers */
    np_centers = (PyArrayObject*)PyArray_ContiguousFromAny(py_centers, NPY_FLOAT32, 2, 2);
    if(!np_centers) {
        PyErr_SetString(PyExc_ValueError, "Could not convert \"centers\" to two-dimensional C-contiguous behaved ndarray of float (32).");
        goto error;
    }
    N_centers = np_centers->dimensions[0];
    if(N_centers==0) {
        PyErr_SetString(PyExc_ValueError, "centers must contain at least one element.");
        goto error;
    }
    if(np_centers->dimensions[1]!=dim) {
        PyErr_SetString(PyExc_ValueError, "Dimension of cluster centers doesn\'t match dimension of frames.");
        goto error;
    }
    centers = (float*)PyArray_DATA(np_centers);

    /* do the assignment */
    switch(c_assign(chunk, centers, dtraj, metric, N_frames, N_centers, dim, n_threads)) {
        case ASSIGN_ERR_INVALID_METRIC:
            PyErr_SetString(PyExc_ValueError, "metric must be one of \"euclidean\" or \"minRMSD\".");
            goto error;
        case ASSIGN_ERR_NO_MEMORY:
            PyErr_NoMemory();
            goto error;
    }

    py_res = Py_BuildValue(""); /* =None */
    /* fall through */
error:
    return py_res;
}

