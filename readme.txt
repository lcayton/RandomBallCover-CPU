***[cpu] Random Ball Cover (RBC) v0.1***
Lawrence Cayton
lcayton@tuebingen.mpg.de

(C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 
This program is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

---------------------------------------------------------------------
SUMMARY

This is a C implementation of the Random Ball Cover data structure for
fast nearest neighbor (NN) search, designed for shared-memory systems.
All parallelization is handled through OpenMP.  This code contains
both the one-shot (approximate) search algorithm, and exact the exact 
search algorithm.  

There is a different implementation available that runs on a GPU;
visit my (Lawrence Cayton) webpage for details.


---------------------------------------------------------------------
FILES

* brute.{c,h} -- implementation of brute force NN search and many
  variants.  These routines perform virtually all the work.
* driver.{c,h} -- definitions of constants and macros, including the
  distance metric.
* rbc.{c,h} -- the core RBC algorithms.  This includes build and
  search routines for exact and approximate search.  The searchExact
  method comes in two forms, one with ManyCores appended to the
  function name; see below for discussion.  
* utils.{c,h} -- supporting code, including the implementations of
  some basic data structures and various routines useful for
  debugging.  


---------------------------------------------------------------------
COMPILATION

This code currently requires the GNU Scientific Library (GSL), which
is available for free on the web (or through a Linux package
manager).  It also requires the OpenMP libraries, and GCC.  

To build the code, type make in a shell.  

The code has been tested under Linux and MacOS X.  


---------------------------------------------------------------------
MISC NOTES ON THE CODE


* The current version requires the GNU Scientific Library (GSL).  The
  code only uses the library for sorting and random number
  generation.  This dependency will probably be removed in the future.

* For exact search, there are separate 1-NN and K-NN functions
  (distinguishable by the function names).  One can run the K-NN
  functions with K=1 and get the same answer as the 1-NN functions,
  but the 1-NN functions are slightly faster, and easier to
  understand.  

* There are two versions of the exact search method for the RBC,
  searchExact(..) and searchExactManyCores(..), plus K-NN versions of
  both.  searchExact(..) is somewhat faster for systems with a small
  number of cores.  I recommend using searchExactManyCores(..) for
  systems with more than 4 cores, though you might try both methods.
