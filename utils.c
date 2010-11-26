/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef UTILS_C
#define UTILS_C

#include<sys/time.h>
#include<stdio.h>
#include<stdlib.h>
#include "utils.h"
#include "defs.h"

//Returns a length l subset of a random permutation of [0,...,n-1]
//using the knuth shuffle.
//Input variable x is assumed to be alloced and of size l.
void subRandPerm(unint l, unint n, unint *x){
  unint i,ri, *y;
  y = (unint*)calloc(n,sizeof(*y));
    
  struct timeval t3;
  gettimeofday(&t3,NULL);
  srand(t3.tv_usec);
  
  for(i=0;i<n;i++)
    y[i]=i;
  
  for(i=0;i<MIN(l,n-1);i++){  //The n-1 bit is necessary because you can't swap the last 
                              //element with something larger.
    ri=randBetween(i+1,n);
    swap(&y[i],&y[ri]);
  }
  
  for(i=0;i<l;i++)
    x[i]=y[i];
  free(y);
}

//Generates a random permutation of 0, ... , n-1 using the knuth shuffle.
//This should probably be merged with subRandPerm. 
void randPerm(unint n, unint *x){
  unint i,ri;
  
  struct timeval t3;
  gettimeofday(&t3,NULL);
  srand(t3.tv_usec);
  
  for(i=0;i<n;i++){
    x[i]=i;
  }
  
  for(i=0;i<n-1;i++){
    ri=randBetween(i+1,n);
    swap(&x[i],&x[ri]);
  }
}

void swap(unint *a, unint *b){
  unint t;
  t=*a; *a=*b; *b=t;
}

//generates a rand int in rand [a,b) 
unint randBetween(unint a, unint b){
  unint val,c;

  if(b<=a){
    fprintf(stderr,"misuse of randBetween.. exiting\n");
    exit(1);
  }
  c= b-a;

  while(c<= (val= rand()/(int)(((unsigned)RAND_MAX + 1) / c)));
  val=val+a;
  
  return val;
}


void printMat(matrix A){
  unint i,j;
  for(i=0;i<A.r;i++){
    for(j=0;j<A.c;j++)
      printf("%6.4f ",(float)A.mat[IDX(i,j,A.ld)]);
    printf("\n");
  }
}


void printMatWithIDs(matrix A, unint *id){
  unint i,j;
  for(i=0;i<A.r;i++){
    for(j=0;j<A.c;j++)
      printf("%6.4f ",(float)A.mat[IDX(i,j,A.ld)]);
    printf("%d ",id[i]);
    printf("\n");
  }
}


void printCharMat(charMatrix A){
  unint i,j;
  for(i=0;i<A.r;i++){
    for(j=0;j<A.c;j++)
      printf("%d ",(char)A.mat[IDX(i,j,A.ld)]);
    printf("\n");
  }
}

void printIntMat(intMatrix A){
  unint i,j;
  for(i=0;i<A.r;i++){
    for(j=0;j<A.c;j++)
      printf("%u ",(unint)A.mat[IDX(i,j,A.ld)]);
    printf("\n");
  }
}

void printVector(real *x, unint d){
  unint i;

  for(i=0 ; i<d; i++)
    printf("%6.2f ",x[i]);
  printf("\n");
}


void copyVector(real *x, real *y, unint d){
  unint i;
  
  for(i=0;i<d;i++)
    x[i]=y[i];
}


void copyMat(matrix *x, matrix *y){
  unint i,j;
  
  x->r=y->r; x->pr=y->pr; x->c=y->c; x->pc=y->pc; x->ld=y->ld;
  for(i=0; i<y->r; i++){
    for(j=0; j<y->c; j++){
      x->mat[IDX( i, j, x->ld )] = y->mat[IDX( i, j, y->ld )];
    }
  }
}


real distVec(matrix x, matrix y, unint k, unint l){
  unint i,j;
  real ans[VEC_LEN];
  real sum=0;

  for(i=0;i<VEC_LEN;i++)
    ans[i]=0;
  
  for(i=0;i<x.pc/VEC_LEN;i+=VEC_LEN){
    for(j=0; j<VEC_LEN; j++)
      ans[j] += DIST( x.mat[IDX(k,i+j,x.ld)], y.mat[IDX(l,i+j,x.ld)] );
  }
  for(i=0; i<VEC_LEN; i++)
    sum += ans[i];
  return sum;
}


double timeDiff(struct timeval start, struct timeval end){
  return (double)(end.tv_sec+end.tv_usec/1e6 - start.tv_sec - start.tv_usec/1e6); 
}


void addToList(intList *l, unint newMem){
  if (l->len < l->maxLen)
    l->x[l->len++] = newMem;
  else{
    unint *y = (unint*)calloc(2*l->maxLen, sizeof(*y));
    unint i;
    for(i=0; i<l->len; i++)
      y[i] = l->x[i];
    free(l->x);
    l->x = y;
    l->x[l->len++] = newMem;
    l->maxLen *= 2;
    
  }
}

void createList(intList *l){
  l->maxLen = DEF_LIST_SIZE;
  l->x = (unint*)calloc(DEF_LIST_SIZE, sizeof(*l->x));
  l->len = 0;
}


void destroyList(intList *l){
  free(l->x);
}

void printList(intList *l){
  unint i;
  printf("len= %d maxLen= %d\n",l->len,l->maxLen);
  for(i=0;i<l->len;i++)
    printf("%d ",l->x[i]);
  printf("\n");
}

void createHeap(heap *hp, unint k){
  int i;
  hp->len = k;
  hp->h = (heapEl*)calloc(k, sizeof(*hp->h));

  heapEl t;
  t.val = MAX_REAL;
  t.id = DUMMY_IDX;
  for(i=0;i<k;i++)
    hp->h[i] = t;
}

void destroyHeap(heap *hp){
  free(hp->h);
}

// This method replaces the top of the heap (ie the max-element)
// with newEl, then re-orders the heap as necessary.  
void replaceMax(heap *hp, heapEl newEl){
  heapEl t;
  hp->h[0] = newEl;
  unint swapInd;
  unint i = 0;
  char done = 0;

  while( 2*i+1 < hp->len && !done){
    if( 2*i+2 < hp->len )
      swapInd = MAXI(hp->h[2*i+1].val, hp->h[2*i+2].val, 2*i+1, 2*i+2);
    else
      swapInd = 2*i+1;
    
    if(hp->h[i].val < hp->h[swapInd].val){
      t.id = hp->h[swapInd].id;
      t.val = hp->h[swapInd].val;
      hp->h[swapInd].id = hp->h[i].id;
      hp->h[swapInd].val = hp->h[i].val;
      hp->h[i].id = t.id;
      hp->h[i].val = t.val;
      i = swapInd;
    }
    else
      done = 1;
  }
  
}

void heapSort(heap *hp, unint *sortInds, real *sortVals){
  unint i,j, swapInd;
  heapEl t;

  for(i=0; i<hp->len; i++){
    sortInds[hp->len-i-1] = hp->h[0].id;
    sortVals[hp->len-i-1] = hp->h[0].val;
    hp->h[0].id = DUMMY_IDX;
    hp->h[0].val = MIN_REAL;

    j=0;
    while(2*j+1 < hp->len ){
      if( 2*j+2 < hp->len )
	swapInd = MAXI(hp->h[2*j+1].val, hp->h[2*j+2].val, 2*j+1, 2*j+2);
      else
	swapInd = 2*j+1;
      if(hp->h[swapInd].id==DUMMY_IDX)
	break;
      t.id = hp->h[swapInd].id;
      t.val = hp->h[swapInd].val;
      hp->h[swapInd].id = hp->h[j].id;
      hp->h[swapInd].val = hp->h[j].val;
      hp->h[j].id = t.id;
      hp->h[j].val = t.val;
      j = swapInd;
    }
  }
}
#endif
