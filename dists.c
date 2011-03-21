/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef DISTS_C
#define DISTS_C

#include "dists.h"
#include "defs.h"

real distVec(matrix x, matrix y, unint k, unint l){
  /* unint i,j; */
  /* real ans[VEC_LEN]; */
  /* real sum=0; */

  /* for(i=0; i<VEC_LEN; i++) */
  /*   ans[i]=0; */
  
  /* for(i=0; i<x.pc; i+=VEC_LEN){ */
  /*   for(j=0; j<VEC_LEN; j++) */
  /*     ans[j] += DIST( x.mat[IDX(k,i+j,x.ld)], y.mat[IDX(l,i+j,x.ld)] ); */
  /* } */
  /* for(i=0; i<VEC_LEN; i++) */
  /*   sum += ans[i]; */
  /* return DIST_EXP(sum); */
  
  unint i, j;
  real sum=0;
  
  for(i=0; i<x.pc; i+=VEC_LEN){
    for(j=0; j<VEC_LEN; j++)
     sum += DIST( x.mat[IDX(k,i+j,x.ld)], y.mat[IDX(l,i+j,x.ld)] );
  }
  return DIST_EXP(sum);

}



real distVecLB(matrix x, matrix y, unint k, unint l, real lb){
 /* unint i,j; */
 /*  real ans[VEC_LEN]; */
 /*  real sum=0; */

 /*  for(i=0; i<VEC_LEN; i++) */
 /*    ans[i]=0; */
  
 /*  for(i=0; i<x.pc; i+=VEC_LEN){ */
 /*    for(j=0; j<VEC_LEN; j++) */
 /*      ans[j] += DIST( x.mat[IDX(k,i+j,x.ld)], y.mat[IDX(l,i+j,x.ld)] ); */
 /*  } */
 /*  for(i=0; i<VEC_LEN; i++) */
 /*    sum += ans[i]; */
 /*  return DIST_EXP(sum); */

  
  unint i, j;
  real sum=0;
  real lb2 = lb==MAX_REAL? lb : DIST_ROOT(lb);

  for(i=0; i<x.pc; i+=VEC_LEN){
    for(j=0; j<VEC_LEN; j++)
      sum += DIST( x.mat[IDX(k,i+j,x.ld)], y.mat[IDX(l,i+j,x.ld)] );

    if( sum > lb2 ) {
      //      if( fabs(DIST_EXP(sum)-lb)<FLOAT_TOL)
      //	printf("%6.9f %6.9f \n",DIST_EXP(sum),lb);
      return MAX_REAL;
    }
  }
  return DIST_EXP(sum);

}


#endif
