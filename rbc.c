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
	  minDist[k] = d[tn][k][j];
	  minID[k] = j;
	}
      }
    }
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


void searchExactK(matrix q, matrix x, matrix r, rep *ri, unint *NNs, unint K){
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
	  minDist[k] = d[tn][k][j];
	  minID[k] = j;
	}
      }
    }
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


void searchExactManyCores(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  unint i, j, k;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));
  intList *toSearch = (intList*)calloc(r.pr, sizeof(*toSearch));
  for(i=0;i<r.pr;i++)
    createList(&toSearch[i]);
  
  brutePar(r,q,repID,dToReps);
  
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

  bruteList(x,q,ri,toSearch,r.r,NNs,dToReps);
  
  for(i=0;i<r.pr;i++)
    destroyList(&toSearch[i]);
  free(toSearch);
  free(repID);
  free(dToReps);
}



void buildExact(matrix x, matrix *r, rep *ri, unint numReps){
  unint n = x.r;
  unint i,j ;

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
  
  for(i=0; i<numReps; i++){
    ri[i].lr = (unint*)calloc(ri[i].len, sizeof(*ri[i].lr));
  }
  
  unint *tempCount = (unint*)calloc(numReps, sizeof(*tempCount));
  for(i=0; i<n; i++)
    ri[repID[i]].lr[tempCount[repID[i]]++] = i;

  free(tempCount);
  free(dToReps);
  free(repID);
}


void buildOneShot(matrix x, matrix *r, rep *ri, unint numReps, unint s){
  unint n = x.r;
  unint ps = CPAD(s);
  unint i, j;
  
  r->c=x.c; r->pc=x.pc; r->r=numReps; r->pr=CPAD(numReps); r->ld=r->pc;
  r->mat = (real*)calloc( r->pc*r->pr, sizeof(*r->mat) );
 
  //pick r random reps
  pickReps(x,r); 

  //Compute the ownership lists
  size_t **repID = (size_t**)calloc(r->pr, sizeof(*repID));
  for( i=0; i<r->pr; i++)
    repID[i] = (size_t*)calloc(CPAD(ps), sizeof(**repID));

  //need to find the radius such that each rep contains s points
  bruteK(x,*r,repID,s);
  
  for( i=0; i<r->pr; i++){
    ri[i].lr = (unint*)calloc(ps, sizeof(*ri[i].lr));
    ri[i].len = s;
    for (j=0; j<s; j++)
      ri[i].lr[j] = repID[i][j];
    //    ri[i].radius = distVec( r, x, i, ri[i].lr[s-1]);  Not needed by one-shot alg
  }
  
  for( i=0; i<r->pr; i++)
    free(repID[i]);
  free(repID);
}


void searchOneShot(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  int i;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));

  brutePar(r,q,repID,dToReps);
  bruteMap(x,q,ri,repID,NNs,dToReps);
  
  free(repID);
  free(dToReps);
}


void pickReps(matrix x, matrix *r){
  unint n = x.r;
  unint i, j;

  unint *shuf = (unint*)calloc(n, sizeof(*shuf));
  for(i=0; i<n; i++)
    shuf[i]=i;

  gsl_rng * rng;
  const gsl_rng_type *rngT;
  
  gsl_rng_env_setup();
  rngT = gsl_rng_default;
  rng = gsl_rng_alloc(rngT);
  
  gsl_ran_shuffle(rng, shuf, n, sizeof(*shuf));
  gsl_rng_free(rng);
 
  for(i=0; i<r->r; i++){
    for(j=0; j<r->c; j++){
      r->mat[IDX( i, j, r->ld )] = x.mat[IDX( shuf[i], j, x.ld )];
    }
  }
  free(shuf);
}


//Determines the total number of computations needed by RBC to find the the NNS.
void searchStats(matrix q, matrix x, matrix r, rep *ri, double *avgDists){
  unint i, j;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));

  brutePar(r,q,repID,dToReps);

  //for each q, need to determine which reps to examine
  size_t numAdded=0;
  size_t totalComp=0;
  size_t numSaved=0;
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
