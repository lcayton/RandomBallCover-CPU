#ifndef RBC_C
#define RBC_C

#include "rbc.h"
#include "defs.h"
#include "utils.h"
#include "brute.h"
#include<omp.h>
#include<gsl/gsl_rng.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_sort.h>
#include<sys/time.h>
#include<stdio.h>
#include<math.h>
#include<stdint.h>

void furthestFirst(matrix x, matrix r){
  //the following simply generates a rand int.  It should be moved
  gsl_rng * rng;
  const gsl_rng_type *rngT;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  gsl_rng_env_setup();
  rngT = gsl_rng_default;
  rng = gsl_rng_alloc(rngT);
  gsl_rng_set(rng,tv.tv_usec);
  unint first = (unint)floor(gsl_ran_flat(rng, 0.0, (double)r.r));
  gsl_rng_free(rng);
  //end rand int generation

  unint i, j;

  matrix dummy;
  dummy.r=1; dummy.pr=PAD(dummy.r); dummy.c=x.c; dummy.pc=x.pc; dummy.ld=x.ld;
  dummy.mat = (real*)calloc(dummy.pr*dummy.pc, sizeof(*dummy.mat));

  real *dists = (real*)calloc(x.pr, sizeof(*dists));
  real *oldDists = (real*)calloc(x.pr, sizeof(*oldDists));
  unint *ids = (unint*)calloc(x.pr, sizeof(*ids));

  copyVector(dummy.mat, &x.mat[IDX(first, 0, x.ld)], x.c);
  copyVector(r.mat, &x.mat[IDX(first, 0, x.ld)], x.c);
  for(i=0; i<x.c; i++)
    oldDists[i] = MAX_REAL;
  for(i=1; i<r.r; i++){
    //find farthest
    brutePar(dummy, x, ids, dists);
    for(j=0; j<x.r; j++)
      oldDists[j] = dists[j] = MIN(dists[j], oldDists[j]);
    real max=-1.0;
    unint maxInd = DUMMY_IDX;
    for(j=0; j<x.r; j++){
      maxInd = MAXI( dists[j], max, j, maxInd );
      max = MAX( dists[j], max );
    }
    copyVector(dummy.mat, &x.mat[IDX(maxInd, 0, x.ld)], x.c);
    copyVector(&r.mat[IDX(i, 0, r.ld)], &x.mat[IDX(maxInd, 0, x.ld)], x.c);
  }



  free(dummy.mat);
  free(dists); free(oldDists); free(ids); 
}

//Builds the RBC for hash-based search.
void buildBit(matrix x, matrix *r, real *repWidth, uint32_t **xb, unint nbits){
  unint n = x.r;
  unint i;

  r->c=x.c; r->pc=x.pc; r->r=nbits; r->pr=CPAD(nbits); r->ld=r->pc;
  r->mat = (real*)calloc( r->pc*r->pr, sizeof(*r->mat) );
 
  //pick r random reps
  pickReps(x,r);
  //furthestFirst(x, *r);

  //Compute the rep for each x
  unint *repID = (unint*)calloc(x.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(x.pr, sizeof(*dToReps));
 
  brutePar(*r,x,repID,dToReps);
  
  for(i=0; i<n; i++)
    repWidth[repID[i]] = MAX( repWidth[repID[i]], dToReps[i] );
  
  //build bit representations
  getBitRep(x, *r, repWidth, xb, nbits);

  free(dToReps);
  free(repID);
}


void getBitRep(matrix x, matrix r, real *repWidth, uint32_t **xb, unint nbits){
  unint i, j, k;
  unint nwords = nbits/32;

#pragma omp parallel for private(j,k)
  for(i=0; i<x.pr; i++){
    for(j=0; j<nwords; j++){
      for(k=0; k<32; k++){
	  if (distVec(x, r, i, j*32+k) < repWidth[j*32+k])
	    xb[i][j] |= GETBIT(k);
      }
    }
  }
}


void searchBit(uint32_t **xb, uint32_t **qb, unint n, unint m, unint maxHamm , intList *l, unint nbits){
  unint i,j;

#pragma omp parallel for private(j)
  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      if(hamm(xb[j],qb[i],nbits/32) < maxHamm)
	addToList(&l[i], j);
    }
  }
}


//Builds the RBC for exact (1- or K-) NN search.
void buildExact(matrix x, matrix *r, rep *ri, unint numReps){
  unint n = x.r;
  unint i, j;
  unint longestLength = 0;

  r->c=x.c; r->pc=x.pc; r->r=numReps; r->pr=CPAD(numReps); r->ld=r->pc;
  r->mat = (real*)calloc( r->pc*r->pr, sizeof(*r->mat) );
 
  //pick r random reps
  pickReps(x,r);

  //Compute the rep for each x
  unint *repID = (unint*)calloc(x.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(x.pr, sizeof(*dToReps));

  brutePar(*r,x,repID,dToReps);

  //gather the rep info & store it in struct
  for(i=0; i<numReps; i++){
    ri[i].len = 0;
    ri[i].radius = 0;
}    
  
  for(i=0; i<n; i++){
    ri[repID[i]].radius = MAX( dToReps[i], ri[repID[i]].radius );
    ri[repID[i]].len++;
  }
  
  unint **tempI = (unint**)calloc(numReps, sizeof(*tempI));
  real **tempD = (real**)calloc(numReps, sizeof(*tempD));
  for(i=0; i<numReps; i++){
    tempI[i] = (unint*)calloc(ri[i].len, sizeof(**tempI));
    tempD[i] = (real*)calloc(ri[i].len, sizeof(**tempD));
    ri[i].lr = (unint*)calloc(ri[i].len, sizeof(*ri[i].lr));
    ri[i].dists = (real*)calloc(ri[i].len, sizeof(*ri[i].dists));
    longestLength = MAX( longestLength, ri[i].len );
  }
  
  unint *tempCount = (unint*)calloc(numReps, sizeof(*tempCount));
  for(i=0; i<n; i++){
    tempI[repID[i]][tempCount[repID[i]]] = i;
    tempD[repID[i]][tempCount[repID[i]]++] = dToReps[i];
    //    ri[repID[i]].dists[tempCount[repID[i]]++] = dToReps[i];
    //    ri[repID[i]].lr[tempCount[repID[i]]++] = i;
  }

  //  for(i=0; i<numReps; i++)
  //    tempCount[i]=0;

  size_t *p = (size_t*)calloc(longestLength, sizeof(*p));
  for(i=0; i<numReps; i++){
    gsl_sort_float_index( p, tempD[i], 1, ri[i].len );
    for(j=0; j<ri[i].len; j++){
      ri[i].dists[j] = tempD[i][p[j]];
      ri[i].lr[j] = tempI[i][p[j]];
    }
  }
  free(p);
  for(i=0;i<numReps; i++){
    free(tempI[i]);
    free(tempD[i]);
  }
  free(tempI);
  free(tempD);
  free(tempCount);
  free(dToReps);
  free(repID);
}

//Experimental build exact method
//ol == overlap factor
void buildExactExp(matrix x, matrix *r, rep *ri, unint numReps, unint ol){
  unint n = x.r;
  unint i,j,k;

  r->c=x.c; r->pc=x.pc; r->r=numReps; r->pr=CPAD(numReps); r->ld=r->pc;
  r->mat = (real*)calloc( r->pc*r->pr, sizeof(*r->mat) );

  //this initialization encourages the OS to put the r matrix in a "good"
  //portion of memory.  
#pragma omp parallel for private(j,k)
  for(i=0; i<numReps/CL; i++){
    for(j=0; j<CL; j++){
      for(k=0; k<r->c; k++)
	r->mat[IDX(i*CL+j, k, r->ld)] = 0.1;
      for(k=r->c; k<r->pc; k++)
	r->mat[IDX(i*CL+j, k, r->ld)] = 0;
    }
  }

  //pick r random reps
  pickReps(x,r);

  //Compute the rep for each x
  unint **repID = (unint**)calloc(x.pr, sizeof(*repID));
  real **dToReps = (real**)calloc(x.pr, sizeof(*dToReps));
  for(i=0; i<x.pr; i++){
    repID[i] = (unint*)calloc(ol, sizeof(**repID));
    dToReps[i] = (real*)calloc(ol, sizeof(**dToReps));
  }

  bruteKHeap(*r,x,repID,dToReps,ol);

  //gather the rep info & store it in struct
  for(i=0; i<numReps; i++){
    ri[i].len = 0;
    ri[i].radius = 0;
  }    
  
  for(i=0; i<n; i++){
    for(j=0; j<ol; j++){
      ri[repID[i][j]].radius = MAX( dToReps[i][j], ri[repID[i][j]].radius );
      ri[repID[i][j]].len++;
    }
  }
  
  for(i=0; i<numReps; i++){
    ri[i].lr = (unint*)calloc(ri[i].len, sizeof(*ri[i].lr));
  }
  
  unint *tempCount = (unint*)calloc(numReps, sizeof(*tempCount));
  for(i=0; i<n; i++){
    for(j=0; j<ol; j++)
      ri[repID[i][j]].lr[tempCount[repID[i][j]]++] = i;
  }

  for(i=0; i<x.pr; i++){
    free(repID[i]);
    free(dToReps[i]);
  }
  free(dToReps);
  free(repID);
  free(tempCount);
}


// Reshuffles X and stores the result in Y.  This
// improves memory locality.
void reshuffleX(matrix y, matrix x, rep *ri, unint numReps){
  unint i,j;

  ri[0].start = 0;
  for( i=1; i<numReps; i++ )
    ri[i].start = ri[i-1].start + ri[i-1].len;
  unint t;
  //#pragma omp parallel for private(j,t)
  for( i=0; i<numReps; i++ ){
    t = ri[i].start;
    for( j=0; j<ri[i].len; j++ )
      copyRow( &y, &x, t++, ri[i].lr[j] );
  }
}


//Exact 1-NN search with the RBC.
void searchExact(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  unint i, j, k;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));
  intList *toSearch = (intList*)calloc(r.pr, sizeof(*toSearch));
  for(i=0;i<r.pr;i++)
    createList(&toSearch[i]);
  int nt = omp_get_max_threads();
  
  float ***d;  //d is indexed by: thread, cache line #, rep #
  d = (float***)calloc(nt, sizeof(*d));
  for(i=0; i<nt; i++){
    d[i] = (float**)calloc(CL, sizeof(**d));
    for(j=0; j<CL; j++){
      d[i][j] = (float*)calloc(r.pr, sizeof(***d));
    }
  }
  
#pragma omp parallel for private(j,k)
  for(i=0; i<q.pr/CL; i++){
    unint row = i*CL;
    unint tn = omp_get_thread_num();
    
    unint minID[CL];
    real minDist[CL];
    for(j=0;j<CL;j++)
      minDist[j] = MAX_REAL;
    
    for( j=0; j<r.r; j++ ){
      for(k=0; k<CL; k++){
	d[tn][k][j] = distVec(q, r, row+k, j);
	if(d[tn][k][j] < minDist[k]){
	  minDist[k] = d[tn][k][j]; //gamma
	  minID[k] = j;
	}
      }
    }

    for( j=0; j<CL; j++ )
      dToReps[row+j] = minDist[j];

    for(j=0; j<r.r; j++ ){
      for(k=0; k<CL; k++ ){
	real temp = d[tn][k][j];
	if( row + k<q.r && minDist[k] >= temp - ri[j].radius && 3.0*minDist[k] >= temp ){
#pragma omp critical
	  {
	    addToList(&toSearch[j], row+k);
	  }
	}
      }
    }
  }
  for(i=0; i<r.r; i++){
    while(toSearch[i].len % CL != 0)
      addToList(&toSearch[i],DUMMY_IDX);
  }

  bruteList(x,q,ri,toSearch,r.r,NNs,dToReps);

  for(i=0;i<r.pr;i++)
    destroyList(&toSearch[i]);
  free(toSearch);
  free(repID);
  free(dToReps);
  for(i=0; i<nt; i++){
    for(j=0; j<CL; j++)
      free(d[i][j]); 
    free(d[i]);
  }
  free(d);
}


//Exact k-NN search with the RBC
void searchExactK(matrix q, matrix x, matrix r, rep *ri, unint **NNs, unint K){
  unint i, j, k;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real **dToReps = (real**)calloc(q.pr, sizeof(*dToReps));
  for(i=0; i<q.pr; i++)
    dToReps[i] = (real*)calloc(K, sizeof(**dToReps));
  intList *toSearch = (intList*)calloc(r.pr, sizeof(*toSearch));
  for(i=0;i<r.pr;i++)
    createList(&toSearch[i]);
  int nt = omp_get_max_threads();

  float ***d;  //d is indexed by: thread, cache line #, rep #
  d = (float***)calloc(nt, sizeof(*d));
  for(i=0; i<nt; i++){
    d[i] = (float**)calloc(CL, sizeof(**d));
    for(j=0; j<CL; j++){
      d[i][j] = (float*)calloc(r.pr, sizeof(***d));
    }
  }
  
  heap **hp;
  hp = (heap**)calloc(nt, sizeof(*hp));
  for(i=0; i<nt; i++){
    hp[i] = (heap*)calloc(CL, sizeof(**hp));
    for(j=0; j<CL; j++)
      createHeap(&hp[i][j],K);
  }
  
#pragma omp parallel for private(j,k)
  for(i=0; i<q.pr/CL; i++){
    unint row = i*CL;
    unint tn = omp_get_thread_num();
    heapEl newEl; 

    for( j=0; j<r.r; j++ ){
      for(k=0; k<CL; k++){
	d[tn][k][j] = distVec(q, r, row+k, j);
	if( d[tn][k][j] < hp[tn][k].h[0].val ){
	  newEl.id = j;
	  newEl.val = d[tn][k][j];
	  replaceMax( &hp[tn][k], newEl );
	}
      }
    }
    for(j=0; j<r.r; j++ ){
      for(k=0; k<CL; k++ ){
	real minDist = hp[tn][k].h[0].val;
	real temp = d[tn][k][j];
	if( row + k<q.r && minDist >= temp - ri[j].radius && 3.0*minDist >= temp ){
#pragma omp critical
	  {
	    addToList(&toSearch[j], row+k);
	  }
	}
      }
    }
    for(j=0; j<CL; j++)
      reInitHeap(&hp[tn][j]);
  }

  for(i=0; i<r.r; i++){
    while(toSearch[i].len % CL != 0)
      addToList(&toSearch[i],DUMMY_IDX);
  }

  bruteListK(x,q,ri,toSearch,r.r,NNs,dToReps,K);

  
  for(i=0; i<nt; i++){
    for(j=0; j<CL; j++)
      destroyHeap(&hp[i][j]);
    free(hp[i]);
  }
  free(hp);
  for(i=0;i<r.pr;i++)
    destroyList(&toSearch[i]);
  free(toSearch);
  free(repID);
  for(i=0;i<q.pr; i++)
    free(dToReps[i]);
  free(dToReps);
  for(i=0; i<nt; i++){
    for(j=0; j<CL; j++)
      free(d[i][j]); 
    free(d[i]);
  }
  free(d);
}

// Exact 1-NN search with the RBC.  This version works better on computers
// with a high core count (say > 8)
void searchExactManyCores(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  unint i, j, k;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));
  intList *toSearch = (intList*)calloc(r.pr, sizeof(*toSearch));
  struct timeval tvB,tvE;
  for(i=0;i<r.pr;i++)
    createList(&toSearch[i]);

  gettimeofday(&tvB,NULL);
  brutePar(r,q,repID,dToReps);
  gettimeofday(&tvE,NULL);
  printf("....exact[pt1] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

  gettimeofday(&tvB,NULL);
#pragma omp parallel for private(j,k)
  for(i=0; i<r.pr/CL; i++){
    unint row = CL*i;
    real temp[CL];
    
    for(j=0; j<q.r; j++ ){
      for(k=0; k<CL; k++){
	temp[k] = distVec( q, r, j, row+k );
      }
      for(k=0; k<CL; k++){
	//dToRep[j] is current UB on dist to j's NN
	//temp - ri[i].radius is LB to dist belonging to rep i
	if( row+k<r.r && dToReps[j] >= temp[k] - ri[row+k].radius && 3.0*dToReps[j] >= temp[k] )
	  addToList(&toSearch[row+k], j); //need to search rep 
      }
    }
    for(j=0;j<CL;j++){
      if(row+j<r.r){
	while(toSearch[row+j].len % CL != 0)
	  addToList(&toSearch[row+j],DUMMY_IDX);	
      }
    }
  }

  gettimeofday(&tvE,NULL);
  printf("....exact[pt2] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );


  //Most of the time is spent in this method
  gettimeofday(&tvB,NULL);
  bruteList(x,q,ri,toSearch,r.r,NNs,dToReps);
  gettimeofday(&tvE,NULL);
  printf("....exact[pt3] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );


  for(i=0;i<r.pr;i++)
    destroyList(&toSearch[i]);
  free(toSearch);
  free(repID);
  free(dToReps);
}

void searchExactManyCores2(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  unint i, j, k;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));
  intList *toSearch = (intList*)calloc(r.pr, sizeof(*toSearch));
  struct timeval tvB,tvE;
  for(i=0;i<r.pr;i++)
    createList(&toSearch[i]);

  gettimeofday(&tvB,NULL);
  brutePar(r,q,repID,dToReps);
  gettimeofday(&tvE,NULL);
  printf("....exact[pt1] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

  gettimeofday(&tvB,NULL);
#pragma omp parallel for private(j,k)
  for(i=0; i<r.pr/CL; i++){
    unint row = CL*i;
    real temp[CL];
    
    for(j=0; j<q.r; j++ ){
      for(k=0; k<CL; k++){
	temp[k] = distVec( q, r, j, row+k );
      }
      for(k=0; k<CL; k++){
	//dToRep[j] is current UB on dist to j's NN
	//temp - ri[i].radius is LB to dist belonging to rep i
	if( row+k<r.r && dToReps[j] >= temp[k] - ri[row+k].radius && 3.0*dToReps[j] >= temp[k] )
	  addToList(&toSearch[row+k], j); //need to search rep 
      }
    }
    for(j=0;j<CL;j++){
      if(row+j<r.r){
	while(toSearch[row+j].len % CL != 0)
	  addToList(&toSearch[row+j],DUMMY_IDX);	
      }
    }
  }

  gettimeofday(&tvE,NULL);
  printf("....exact[pt2] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );


  //Most of the time is spent in this method
  gettimeofday(&tvB,NULL);
  bruteList2(x,q,ri,toSearch,r.r,NNs,dToReps);
  gettimeofday(&tvE,NULL);
  printf("....exact[pt3] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );


  for(i=0;i<r.pr;i++)
    destroyList(&toSearch[i]);
  free(toSearch);
  free(repID);
  free(dToReps);
}



// Exact k-NN search with the RBC.  This version works better on computers
// with a high core count (say > 8)
void searchExactManyCoresK(matrix q, matrix x, matrix r, rep *ri, unint **NNs, unint K){
  unint i, j, k;
  unint **repID = (unint**)calloc(q.pr, sizeof(*repID));
  for(i=0; i<q.pr; i++)
    repID[i] = (unint*)calloc(K, sizeof(**repID));
  real **dToReps = (real**)calloc(q.pr, sizeof(*dToReps));
  for(i=0; i<q.pr; i++)
    dToReps[i] = (real*)calloc(K, sizeof(**dToReps));
  intList *toSearch = (intList*)calloc(r.pr, sizeof(*toSearch));
  for(i=0;i<r.pr;i++)
    createList(&toSearch[i]);
  
  bruteKHeap(r,q,repID,dToReps,K);
  
#pragma omp parallel for private(j,k)
  for(i=0; i<r.pr/CL; i++){
    unint row = CL*i;
    real temp[CL];
    
    for(j=0; j<q.r; j++ ){
      for(k=0; k<CL; k++){
	temp[k] = distVec( q, r, j, row+k );
      }
      for(k=0; k<CL; k++){
	//dToRep[j] is current UB on dist to j's NN
	//temp - ri[i].radius is LB to dist belonging to rep i
	if( row+k<r.r && dToReps[j][K-1] >= temp[k] - ri[row+k].radius && 3.0*dToReps[j][K-1] >= temp[k] )
	  addToList(&toSearch[row+k], j); //need to search rep 
      }
    }
    for(j=0;j<CL;j++){
      if(row+j<r.r){
	while(toSearch[row+j].len % CL != 0)
	  addToList(&toSearch[row+j],DUMMY_IDX);	
      }
    }
  }

  bruteListK(x,q,ri,toSearch,r.r,NNs,dToReps,K);
  
  for(i=0;i<q.pr;i++)
    free(dToReps[i]);
  free(dToReps);
  for(i=0;i<r.pr;i++)
    destroyList(&toSearch[i]);
  free(toSearch);
  for(i=0;i<q.pr;i++)
    free(repID[i]);
  free(repID);
}


//Builds the RBC for the One-shot (inexact) method.
void buildOneShot(matrix x, matrix *r, rep *ri, unint numReps, unint s){
  unint ps = CPAD(s);
  unint i, j;
  
  r->c=x.c; r->pc=x.pc; r->r=numReps; r->pr=CPAD(numReps); r->ld=r->pc;
  r->mat = (real*)calloc( r->pc*r->pr, sizeof(*r->mat) );
 
  //pick r random reps
  pickReps(x,r); 

  //Compute the ownership lists
  unint **repID = (unint**)calloc(r->pr, sizeof(*repID));
  real **dToNNs = (real**)calloc(r->pr, sizeof(*dToNNs));
  for( i=0; i<r->pr; i++){
    repID[i] = (unint*)calloc(CPAD(ps), sizeof(**repID));
    dToNNs[i] = (real*)calloc(CPAD(ps), sizeof(**dToNNs));
  }

  //need to find the radius such that each rep contains s points
  bruteKHeap(x,*r,repID,dToNNs,s);
  
  for( i=0; i<r->pr; i++){
    ri[i].lr = (unint*)calloc(ps, sizeof(*ri[i].lr));
    ri[i].len = s;
    for (j=0; j<s; j++){
      ri[i].lr[j] = repID[i][j];
    }
    ri[i].radius = distVec( *r, x, i, ri[i].lr[s-1]);  //Not needed by one-shot alg
    // printf("%6.2f \n",ri[i].radius);
  }
  
  for( i=0; i<r->pr; i++){
    free(dToNNs[i]);
    free(repID[i]);
  }
  free(dToNNs);
  free(repID);
}


// Performs (approx) 1-NN search with the RBC One-shot algorithm.
void searchOneShot(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));
  
  // Determine which rep each query is closest to.
  struct timeval tvB, tvE;
  gettimeofday(&tvB,NULL);
  brutePar(r,q,repID,dToReps);
  gettimeofday(&tvE,NULL);
  printf("....one-shot[pt1] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

  gettimeofday(&tvB,NULL);
  // Search that rep's ownership list.
  bruteMap(x,q,ri,repID,NNs,dToReps);
  gettimeofday(&tvE,NULL);
  printf("....one-shot[pt2] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );
  

  free(repID);
  free(dToReps);
}


void searchOneShot2(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));
  
  // Determine which rep each query is closest to.
  struct timeval tvB, tvE;
  gettimeofday(&tvB,NULL);
  brutePar2(r,q,repID,dToReps);
  gettimeofday(&tvE,NULL);
  printf("....one-shot[pt1] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

  gettimeofday(&tvB,NULL);
  // Search that rep's ownership list.
  bruteMap2(x,q,ri,repID,NNs,dToReps, r.r);
  gettimeofday(&tvE,NULL);
  printf("....one-shot[pt2] time elapsed = %6.4f \n", timeDiff(tvB,tvE) );
  

  free(repID);
  free(dToReps);
}


// Performs (approx) 1-NN search with the RBC One-shot algorithm.
void searchOneShotK(matrix q, matrix x, matrix r, rep *ri, unint **NNs, unint K){
  int i;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real **dToReps = (real**)calloc(q.pr, sizeof(*dToReps));
  for(i=0; i<q.pr; i++)
    dToReps[i] = (real*)calloc(K, sizeof(**dToReps));
  real *dT = (real*)calloc(q.pr, sizeof(*dT));
  // Determine which rep each query is closest to.
  brutePar(r,q,repID,dT);
  
  // Search that rep's ownership list.
  bruteMapK(x,q,ri,repID,NNs,dToReps,K);
  
  free(repID);
  for(i=0; i<q.pr; i++)
    free(dToReps[i]);
  free(dToReps);
  free(dT);
}


// Chooses representatives at random from x and stores them in r.
void pickReps(matrix x, matrix *r){
  unint n = x.r;
  unint i, j;

  unint *shuf = (unint*)calloc(n, sizeof(*shuf));
  for(i=0; i<n; i++)
    shuf[i]=i;


  struct timeval tv;
  gettimeofday(&tv,NULL);
  gsl_rng * rng;
  const gsl_rng_type *rngT;
  
  gsl_rng_env_setup();
  rngT = gsl_rng_default;
  rng = gsl_rng_alloc(rngT);
  gsl_rng_set(rng,tv.tv_usec);
  
  gsl_ran_shuffle(rng, shuf, n, sizeof(*shuf));
  gsl_rng_free(rng);
 
  for(i=0; i<r->r; i++){
    for(j=0; j<r->c; j++){
      r->mat[IDX( i, j, r->ld )] = x.mat[IDX( shuf[i], j, x.ld )];
    }
  }
  free(shuf);
}


// Determines the total number of computations needed by RBC to 
// find the the NNS.  This function is useful mainly for 
// evaluating the effectiveness of the RBC.  
void searchStats(matrix q, matrix x, matrix r, rep *ri, double *avgDists){
  unint i, j;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));

  brutePar(r,q,repID,dToReps);

  //for each q, need to determine which reps to examine
  size_t numAdded=0;
  size_t totalComp=0;
#pragma omp parallel for private(j) reduction(+:numAdded,totalComp)
  for(i=0; i<q.r; i++){
    for(j=0; j<r.r; j++ ){
      real temp = distVec( q, r, i, j );
      //dToRep[i] is current UB on dist to i's NN
      //temp - ri[j].radius is LB to dist belonging to rep j
      if( dToReps[i] >= temp - ri[j].radius && temp <= 3.0*dToReps[i] ){
	numAdded++;
	totalComp+=ri[j].len;
      }
    }
  }
  
  *avgDists = ((double)totalComp)/((double)q.r);
  free(repID);
  free(dToReps);
}


#endif
