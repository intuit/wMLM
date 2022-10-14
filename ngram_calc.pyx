#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
cimport numpy as np
import cython

ctypedef np.int32_t cINT32
ctypedef np.double_t cDOUBLE
ctypedef np.float64_t FLOAT_t

cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object unicode)

from libc.stdlib cimport malloc, free
from libc.string cimport strcmp

cdef char ** to_cstring_array(list_str):
    cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
    for i in xrange(len(list_str)):
        ret[i] = PyUnicode_AsUTF8(list_str[i])
    return ret

cpdef dict ngram_calc(list token_list,cINT32 length, cINT32 skip_distance):
    
    cdef dict loop_dict = {}
    cdef char **c_token_list = to_cstring_array(token_list)
    cdef int i
    cdef int j
    cdef str w1
    cdef str w2

    for i in range(length):
        for j in range(length):
            if((i<j) and abs(i-j)<=(skip_distance+1)):
                w1=c_token_list[i].decode('UTF-8')
                w2=c_token_list[j].decode('UTF-8')
                key=(w1,w2)

                if(key in loop_dict):
                    loop_dict[key]=loop_dict[key]+1
                else:
                    loop_dict[key]=1

    return loop_dict