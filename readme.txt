* The current version requires the GNU Scientific Library (GSL).  The
  code only uses the library for sorting and random number
  generation.  This dependency will probably be removed in the future.

* There are two versions of the exact search method for the RBC.
  searchExactManyCores(..) seems to perform much better on systems
  with many cores (e.g., I was experimenting on a system with 48
  cores, 24 of which I was using), where as searchExact(..) seems to
  work better on systems that have only around four cores.  
