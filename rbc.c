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

void searchStats(matrix q, matrix x, matrix r, rep *ri, double *avgDists){
  unint i, j;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));


  brutePar2(r,q,repID,dToReps);


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


void searchExact2(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  unint i, j;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));
  intList *toSearch = (intList*)calloc(r.pr, sizeof(*toSearch));
  for(i=0;i<r.pr;i++)
    createList(&toSearch[i]);
  
  struct timeval tvB,tvE;

  gettimeofday(&tvB,NULL);
  brutePar2(r,q,repID,dToReps);
  gettimeofday(&tvE,NULL);
  printf("[SE]brutePar2 time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

  gettimeofday(&tvB,NULL);
#pragma omp parallel for private(j)
  for(i=0; i<r.r; i++){
    for(j=0; j<q.r; j++ ){
      real temp = distVec( q, r, j, i );
      //dToRep[j] is current UB on dist to j's NN
      //temp - ri[i].radius is LB to dist belonging to rep i
      if( dToReps[j] >= temp - ri[i].radius)
	addToList(&toSearch[i], j); //need to search rep i
    }
    while(toSearch[i].len % CL != 0)
      addToList(&toSearch[i],DUMMY_IDX);
  }
  gettimeofday(&tvE,NULL);
  printf("[SE]loop time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

  gettimeofday(&tvB,NULL);
  bruteListRev(x,q,ri,toSearch,r.r,NNs,dToReps);
  gettimeofday(&tvE,NULL);
  printf("[SE]bruteListRev time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

  for(i=0;i<r.pr;i++)
    destroyList(&toSearch[i]);
  free(toSearch);
  free(repID);
  free(dToReps);
}


void searchExact3(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  unint i, j, k;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));
  intList *toSearch = (intList*)calloc(r.pr, sizeof(*toSearch));
  for(i=0;i<r.pr;i++)
    createList(&toSearch[i]);
  int nt = omp_get_max_threads();
  
  float ***d;
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

  bruteListRev(x,q,ri,toSearch,r.r,NNs,dToReps);

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


void searchExact4(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  unint i, j, k;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));
  intList *toSearch = (intList*)calloc(r.pr, sizeof(*toSearch));
  for(i=0;i<r.pr;i++)
    createList(&toSearch[i]);
  
  struct timeval tvB,tvE;

  gettimeofday(&tvB,NULL);
  brutePar2(r,q,repID,dToReps);
  gettimeofday(&tvE,NULL);
  printf("[SE]brutePar2 time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

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
	  addToList(&toSearch[row+k], j); //need to search rep i
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
  printf("[SE]loop time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

  gettimeofday(&tvB,NULL);
  bruteListRev(x,q,ri,toSearch,r.r,NNs,dToReps);
  gettimeofday(&tvE,NULL);
  printf("[SE]bruteListRev time elapsed = %6.4f \n", timeDiff(tvB,tvE) );

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

  brutePar2(*r,x,repID,dToReps);

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


void buildDefeat(matrix x, matrix *r, rep *ri, unint numReps, unint s){
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
  bruteK2(x,*r,repID,s);
  
  for( i=0; i<r->pr; i++){
    ri[i].lr = (unint*)calloc(ps, sizeof(*ri[i].lr));
    ri[i].len = s;
    for (j=0; j<s; j++)
      ri[i].lr[j] = repID[i][j];
    //    ri[i].radius = distVec( r, x, i, ri[i].lr[s-1]);  Not needed by defeat alg
  }
  
  for( i=0; i<r->pr; i++)
    free(repID[i]);
  free(repID);
}


void searchDefeat(matrix q, matrix x, matrix r, rep *ri, unint *NNs){
  int i;
  unint *repID = (unint*)calloc(q.pr, sizeof(*repID));
  real *dToReps = (real*)calloc(q.pr, sizeof(*dToReps));

  brutePar2(r,q,repID,dToReps);
  bruteMapSort(x,q,ri,repID,NNs,dToReps);
  
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


void dimEst(matrix x, unint s){
  unint n = x.r;
  unint i,j ;
  int t = 3;
  long unsigned int curMax, oldMax; 

  matrix r;
  r.c=x.c; r.pc=x.pc; r.r=s; r.pr=CPAD(s); r.ld=r.pc;
  r.mat = (real*)calloc( r.pc*r.pr, sizeof(*r.mat) );
  
  size_t** NNs = (size_t**)calloc(r.r, sizeof(*NNs));
  real** dToNNs = (real**)calloc(r.r, sizeof(*dToNNs));
  for(i=0;i<r.pr;i++){
    NNs[i] = (size_t*)calloc(t, sizeof(**NNs));
    dToNNs[i] = (real*)calloc(t, sizeof(**dToNNs));
  }
  
  real* ranges = (real*)calloc(r.pr, sizeof(*ranges));
  unint *counts = (unint*)calloc(r.pr,sizeof(*counts));

  size_t *p = (size_t*)calloc(r.pr, sizeof(*p));
  //pick r random reps
  pickReps(x,&r);

  //printf("calling bruteKdists\n");
  bruteKDists(x,r,NNs,dToNNs,t); 
  //  printf("done\n");
  for(i=0;i<r.r;i++)
    ranges[i] = dToNNs[i][t-1];

  for(i=0; i<10; i++){
    rangeCount(x,r,ranges,counts);
 
    //gsl_sort_uint_index(p,counts,1,r.r);
    //printf("80 = %u \n",counts[p[5*r.r/10]]);
    for(j=0; j<r.r; j++)
      ranges[j]*=2.0;
    curMax = 0;
    unint avg = 0;
    for(j=0; j<r.r; j++){
      //      printf("%u ",counts[j]);
      curMax = MAX(counts[j],curMax);
      avg +=counts[j];
    }
    //    printf("\n");
    //    printf("avg = %6.4f \n",((double)avg)/((double)r.r));
    //    printf("%lu \n",curMax);
  }
  
  for(i=0;i<r.r;i++){
    free(NNs[i]);
    free(dToNNs[i]);
  }
  free(NNs);
  free(dToNNs);
  free(r.mat);
  free(ranges);
  free(counts);
  free(p);
}


#endif
