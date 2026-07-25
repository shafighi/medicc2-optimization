#cython: c_string_encoding=utf8, c_string_type=unicode, language_level=3, nonecheck=True
# distutils: language=c++

## Imports.

# Cython operator workarounds.
from cython.operator cimport address as addr       # &foo
from cython.operator cimport dereference as deref  # *foo
from cython.operator cimport preincrement as inc   # ++foo

## my imports
from libcpp.vector cimport vector
from .pywrapfst cimport *
from .cops cimport *
import numpy as np

## additional constants
DELTA = fst.kDelta
SHORTEST_DELTA = fst.kShortestDelta


def runme():
    print('This works')

cpdef MutableFst align(Fst model, Fst ifst1, Fst ifst2):
  
  cdef unique_ptr[fst.VectorFstClass] tfst
  tfst.reset(new fst.VectorFstClass(ifst1.arc_type()))

  align_std_impl(deref(model._fst), deref(ifst1._fst), deref(ifst2._fst), tfst.get())

  return _init_MutableFst(tfst.release())

cpdef Weight score_std(Fst model, Fst ifst1, Fst ifst2):
  cdef fst.WeightClass distance
  with nogil:
    distance = score_std_impl(deref(model._fst), deref(ifst1._fst), deref(ifst2._fst))
  retval = Weight(model._fst.get().WeightType(), distance.ToString())
  return retval

cpdef Weight score_log(Fst model, Fst ifst1, Fst ifst2):
  cdef fst.WeightClass distance
  with nogil:
    distance = score_log_impl(deref(model._fst), deref(ifst1._fst), deref(ifst2._fst))
  retval = Weight(model._fst.get().WeightType(), distance.ToString())
  return retval

cpdef Weight kernel_score_std(Fst model, Fst ifst1, Fst ifst2):
  cdef fst.WeightClass distance
  with nogil:
    distance = kernel_score_std_impl(deref(model._fst), deref(ifst1._fst), deref(ifst2._fst))
  retval = Weight(model._fst.get().WeightType(), distance.ToString())
  return retval

cpdef Weight kernel_score_log(Fst model, Fst ifst1, Fst ifst2):
  cdef fst.WeightClass distance
  with nogil:
    distance = kernel_score_log_impl(deref(model._fst), deref(ifst1._fst), deref(ifst2._fst))
  retval = Weight(model._fst.get().WeightType(), distance.ToString())
  return retval

cpdef Weight multi_score_std(Fst loh, Fst wgd, Fst gl, Fst ifst1, Fst ifst2):
  cdef fst.WeightClass distance
  with nogil:
    distance = multi_score_std_impl(deref(loh._fst), deref(wgd._fst), deref(gl._fst), deref(ifst1._fst), deref(ifst2._fst))
  retval = Weight(loh._fst.get().WeightType(), distance.ToString())
  return retval

cpdef Weight multi_kernel_score_std(Fst loh, Fst wgd, Fst gain, Fst loss, Fst ifst1, Fst ifst2):
  cdef fst.WeightClass distance
  with nogil:
    distance = multi_kernel_score_std_impl(deref(loh._fst), deref(wgd._fst), deref(gain._fst), deref(loss._fst), deref(ifst1._fst), deref(ifst2._fst))
  retval = Weight(loh._fst.get().WeightType(), distance.ToString())
  return retval
