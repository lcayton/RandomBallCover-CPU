/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef BRUTE_C
#define BRUTE_C

#include "utils.h"
#include "defs.h"
#include "brute.h"
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<gsl/gsl_sort.h>


void brutePar(matrix X, matrix Q, unint *NNs, real *dToNNs){
  real temp[CL];
  int i, j, k,t ;
  
#pragma omp parallel for private(t,k,j,temp) 
  for( i=0; i<Q.pr/CL; i++ ){
    t = i*CL;
    for(j=0;j<CL;j++){
      dToNNs[t+j] = MAX_REAL;
      NNs[t+j] = 0;
    }
    for(j=0; j<X.r; j++ ){
      for(k=0; k<CL; k++){
	temp[k] = distVec( Q, X, t+k, j );
      }
      for(k=0; k<CL; k++){
	if( temp[k] < dToNNs[t+k]){
	  NNs[t+k] = j;
	  dToNNs[t+k] = temp[k];
	}
      }
    }
  }
}


void bruteKHeap(matrix X, matrix Q, unint **NNs, real **dToNNs, unint K){
  real temp[CL];
  int i, j, k,t;

  int nt = omp_get_max_threads();
  heap **hp;
  hp = (heap**)calloc(nt, sizeof(*hp));
  for(i=0; i<nt; i++){
    hp[i] = (heap*)calloc(CL, sizeof(**hp));
    for(j=0; j<CL; j++)
      createHeap(&hp[i][j],K);
  }
  
#pragma omp parallel for private(t,k,j,temp) 
  for( i=0; i<Q.pr/CL; i++ ){
    t = i*CL;
    unint tn = omp_get_thread_num();
    heapEl newEl;

    for(j=0; j<X.r; j++ ){
      for(k=0; k<CL; k++)
	temp[k] = distVec( Q, X, t+k, j );
      
      for(k=0; k<CL; k++){
	if( temp[k] < hp[tn][k].h[0].val ){
	  newEl.id=j;
	  newEl.val=temp[k];
	  replaceMax( &hp[tn][k], newEl);
	}
      }
    }
    for(j=0; j<CL; j++)
      heapSort(&hp[tn][j], NNs[t+j], dToNNs[t+j]);
      
    for(j=0; j<CL; j++)
      reInitHeap(&hp[tn][j]);
  }
  
  for(i=0; i<nt; i++){
    for(j=0; j<CL; j++)
      destroyHeap(&hp[i][j]);
    free(hp[i]);
  }
  free(hp);
  
}


void bruteK(matrix x, matrix q, size_t **NNs, unint k){
  int i, j, l;
  int nt = omp_get_max_threads();

  float ***d;
  size_t ***t;
  d = (float***)calloc(nt, sizeof(*d));
  t = (size_t***)calloc(nt, sizeof(*t));
  for(i=0; i<nt; i++){
    d[i] = (float**)calloc(CL, sizeof(**d));
    t[i] = (size_t**)calloc(CL, sizeof(**t));
    for(j=0; j<CL; j++){
      d[i][j] = (float*)calloc(x.pr, sizeof(***d));
      t[i][j] = (size_t*)calloc(x.pr, sizeof(***t));
    }
  }
      
#pragma omp parallel for private(j,l) shared(d,t,k)
  for( i=0; i<q.pr/CL; i++){
    int row = i*CL;
    int tn = omp_get_thread_num(); //thread num

    for( j=0; j<x.r; j++){
      for( l=0; l<CL; l++){
	d[tn][l][j] =  distVec( q, x, row+l, j);
      }
    }
    for(l=0; l<CL; l++)
      gsl_sort_float_smallest_index(t[tn][l], k, d[tn][l], 1, x.r);
    
    for(l=0; l<CL; l++){
      for(j=0; j<k; j++){
	NNs[row+l][j] = t[tn][l][j];
      }
    }
  }

  for(i=0; i<nt; i++){
    for(j=0; j<CL; j++){
      free(d[i][j]);  free(t[i][j]);
    }
    free(d[i]);  free(t[i]);
  }  
  free(t); free(d);
}


void bruteKDists(matrix x, matrix q, size_t **NNs, real **D, unint k){
  int i, j;

  float **d;
  d = (float**)calloc(q.pr, sizeof(*d));
  size_t **t;
  t = (size_t**)calloc(q.pr, sizeof(*t));
  for( i=0; i<q.pr; i++){
    d[i] = (float*)calloc(x.pr, sizeof(**d));
    t[i] = (size_t*)calloc(x.pr, sizeof(**t));
  }

#pragma omp parallel for private(j)
  for( i=0; i<q.r; i++){
    for( j=0; j<x.r; j++)
      d[i][j] = distVec( q, x, i, j );
    gsl_sort_float_index(t[i], d[i], 1, x.r);
    for ( j=0; j<k; j++){
      NNs[i][j]=t[i][j];
      D[i][j]=d[i][t[i][j]];
    }
  }

  for( i=0; i<q.pr; i++){
    free(t[i]);
    free(d[i]);
  }
  free(t);
  free(d);
}


void bruteMap(matrix X, matrix Q, rep *ri, unint* qMap, unint *NNs, real *dToNNs){
  unint i, j, k;
  
  size_t *qSort = (size_t*)calloc(Q.pr, sizeof(*qSort));
  gsl_sort_uint_index(qSort,qMap,1,Q.r);

#pragma omp parallel for private(j,k)
  for( i=0; i<Q.pr/CL; i++ ){
    unint row = i*CL;
    for(j=0; j<CL; j++){
      dToNNs[qSort[row+j]] = MAX_REAL;
      NNs[qSort[row+j]] = 0;
    }
    
    real temp;
    rep rt[CL];
    unint maxLen = 0;
    for(j=0; j<CL; j++){
      rt[j] = ri[qMap[qSort[row+j]]];
      maxLen = MAX(maxLen, rt[j].len);
    }  
    
    for(k=0; k<maxLen; k++ ){
      for(j=0; j<CL; j++ ){
	if( k<rt[j].len ){
	  temp = distVec( Q, X, qSort[row+j], rt[j].lr[k] );
	  if( temp < dToNNs[qSort[row+j]]){
	    NNs[qSort[row+j]] = rt[j].lr[k];
	    dToNNs[qSort[row+j]] = temp;
	  }
	}
      }
    }
  }
  free(qSort);
}


void rangeCount(matrix X, matrix Q, real *ranges, unint *counts){
  real temp;
  unint i, j, k;

#pragma omp parallel for private(j,k,temp) shared(counts,ranges)
  for( i=0; i<Q.pr/CL; i++ ){
    unint row = i*CL;
    for( j=0; j<CL; j++)
      counts[row+j] = 0;
    for(k=0; k<X.r; k++ ){
      for( j=0; j<CL; j++){
	temp = distVec( Q, X, row+j, k );
	counts[row+j] += ( temp < ranges[row+j] );
      }
    }
  }
}


void bruteList(matrix X, matrix Q, rep *ri, intList *toSearch, unint numReps, unint *NNs, real *dToNNs){
  real temp;
  unint i, j, k, l;
  
  for(i=0; i<Q.r; i++){
    dToNNs[i] = MAX_REAL;
    NNs[i] = 0;
  }

#pragma omp parallel for private(j,k,l,temp)
  for( i=0; i<numReps; i++ ){
    for( j=0; j< toSearch[i].len/CL; j++){  //toSearch is assumed to be padded
      unint row = j*CL;
      unint qInd[CL];
      for(k=0; k<CL; k++)
	qInd[k] = toSearch[i].x[row+k];
      rep rt = ri[i];
      unint curMinInd[CL];
      real curMinDist[CL];
      for(k=0; k<CL; k++)
	curMinDist[k] = MAX_REAL;
      for(k=0; k<rt.len; k++){
	for(l=0; l<CL; l++ ){
	  if(qInd[l]!=DUMMY_IDX){
	    temp = distVec( Q, X, qInd[l], rt.lr[k] );
	    if( temp < curMinDist[l] ){
	      curMinInd[l] = rt.lr[k];
	      curMinDist[l] = temp;
	    }
	  }
	}
      }
#pragma omp critical
      {
	for(k=0; k<CL; k++){
	  if( qInd[k]!=DUMMY_IDX && curMinDist[k] < dToNNs[qInd[k]]){
	    NNs[qInd[k]] = curMinInd[k];
	    dToNNs[qInd[k]] = curMinDist[k];
	  }
	}
      }
    }
  }
}


// Merges at the end, getting rid of the critical section
void bruteList2(matrix X, matrix Q, rep *ri, intList *toSearch, unint numReps, unint *NNs, real *dToNNs){
  real temp;
  unint i, j, k, l;
  
  unint nt = omp_get_max_threads();
  unint m = Q.r;

  real **d = calloc(nt, sizeof(*d));
  unint **nn = calloc(nt, sizeof(*nn));
  for(i=0; i<nt; i++){
    d[i] = calloc(m, sizeof(**d));
    nn[i] = calloc(m, sizeof(**nn));
  }
  
  for(i=0; i<nt; i++){
    for(j=0; j<m; j++)
      d[i][j] = MAX_REAL;
  }
  for(i=0; i<m; i++)
    dToNNs[i] = MAX_REAL;

#pragma omp parallel for private(j,k,l,temp)
  for( i=0; i<numReps; i++ ){
    unint tn = omp_get_thread_num();

    for( j=0; j< toSearch[i].len/CL; j++){  //toSearch is assumed to be padded
      unint row = j*CL;
      unint qInd[CL];
      for(k=0; k<CL; k++)
	qInd[k] = toSearch[i].x[row+k];
      rep rt = ri[i];
      unint curMinInd[CL];
      real curMinDist[CL];
      for(k=0; k<CL; k++)
	curMinDist[k] = MAX_REAL;
      for(k=0; k<rt.len; k++){
	for(l=0; l<CL; l++ ){
	  if(qInd[l]!=DUMMY_IDX){
	    temp = distVec( Q, X, qInd[l], rt.lr[k] );
	    if( temp < curMinDist[l] ){
	      curMinInd[l] = rt.lr[k];
	      curMinDist[l] = temp;
	    }
	  }
	}
      }
      for(k=0; k<CL; k++){
	if(qInd[k]!=DUMMY_IDX && curMinDist[k] < d[tn][qInd[k]]){
	  nn[tn][qInd[k]] = curMinInd[k];
	  d[tn][qInd[k]] = curMinDist[k];
	}
      }
    }
  }

  // Make this a parallel reduce
  for( i=0; i<nt; i++){
    for( j=0; j<m; j++){
      if( d[i][j] < dToNNs[j] ){
	dToNNs[j] = d[i][j];
	NNs[j] = nn[i][j];
      }
    }
  }
}


// Merges at the end, getting rid of the critical section
void bruteListK(matrix X, matrix Q, rep *ri, intList *toSearch, unint numReps, unint **NNs, real **dToNNs, unint K){
  real temp;
  unint i, j, k, l;
  
  unint nt = omp_get_max_threads();
  unint m = Q.r;

  heap **hp;
  hp = (heap**)calloc(nt, sizeof(*hp));
  for(i=0; i<nt; i++){
    hp[i] = (heap*)calloc(m, sizeof(**hp));
    for(j=0; j<m; j++)
      createHeap(&hp[i][j],K);
  }      
  
#pragma omp parallel for private(j,k,l,temp)
  for( i=0; i<numReps; i++ ){
    unint tn = omp_get_thread_num();
    heapEl newEl;
    //printf("%u %u \n",i,toSearch[i].len);
    for( j=0; j< toSearch[i].len/CL; j++){  //toSearch is assumed to be padded
      unint row = j*CL;
      unint qInd[CL];
      for(k=0; k<CL; k++)
	qInd[k] = toSearch[i].x[row+k];
      rep rt = ri[i];
      for(k=0; k<rt.len; k++){
	for(l=0; l<CL; l++ ){
	  if(qInd[l]!=DUMMY_IDX){
	    temp = distVec( Q, X, qInd[l], rt.lr[k] );
	    if( temp < hp[tn][qInd[l]].h[0].val ){
	      newEl.id = rt.lr[k];
	      newEl.val = temp;
	      replaceMax( &hp[tn][qInd[l]], newEl );
	    }
	  }
	}
      }
    }
  }

  unint **tInds;
  real **tVals;
  tInds = (unint**)calloc(nt, sizeof(*tInds));
  tVals = (real**)calloc(nt, sizeof(*tVals));
  for( i=0; i<nt; i++ ){
    tInds[i] = (unint*)calloc(K, sizeof(**tInds) );
    tVals[i] = (real*)calloc(K, sizeof(**tVals) );
  }
  
  size_t *indVec = (size_t*)calloc(nt*K, sizeof(*indVec));
  size_t *tempInds = (size_t*)calloc(nt*K, sizeof(*indVec));
  real *valVec = (real*)calloc(nt*K, sizeof(*valVec));

  // Make this a parallel reduce
  for( i=0; i<m; i++){
    for( j=0; j<nt; j++){
      heapSort( &hp[j][i], tInds[j], tVals[j] );
      for( k=0; k<K; k++){
	indVec[j*K + k] = tInds[j][k];
	valVec[j*K + k] = tVals[j][k];
	/* if(tInds[j][k]!=DUMMY_IDX) */
	/*   printf("%u %6.2f ",tInds[j][k],tVals[j][k]); */
      }
    }      
    gsl_sort_float_index(tempInds, valVec, 1, nt*K);
    for( j=0; j<K; j++ ){
      dToNNs[i][j] = valVec[tempInds[j]];
      NNs[i][j] = indVec[tempInds[j]];
    }
  }


  free(tempInds);
  free(indVec);
  free(valVec);
  for( i=0; i<nt; i++ ){
    free(tInds[i]);
    free(tVals[i]);
  }
  free(tInds); 
  free(tVals);
  for(i=0; i<nt; i++){
    for(j=0; j<m; j++)
      destroyHeap(&hp[i][j]);
    free(hp[i]);
  }
  free(hp);
}
#endif
