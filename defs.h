#ifndef DEFS_H
#define DEFS_H

#include<float.h>
#include<limits.h>
#include<math.h>

#define FLOAT_TOL 1e-7
#define VEC_LEN 4 //for SSE instructions
#define CL 16 //related in some vague way to cache line size
#define DEF_LIST_SIZE 1024

//The distance measure that is used.  This macro returns the 
//distance for a single coordinate.
#define DIST(i,j) ( fabs((i)-(j)) )  // L_1
#define DIST_EXP(x) ( (x) ); //L_1
//#define DIST(i,j) ( ( (i)-(j) )*( (i)-(j) ) )  // L_2
//#define DIST_EXP(x) ( sqrt(x) ); //L_2

// Format that the data is manipulated in:
typedef float real;
#define MAX_REAL FLT_MAX
#define MIN_REAL (-1.0*FLT_MAX)

// To switch to double precision, comment out the above 
// 2 lines and uncomment the following two lines. 

//typedef double real;
//#define MAX_REAL DBL_MAX

#define DUMMY_IDX UINT_MAX

//Row major indexing
#define IDX(i,j,ld) (((i)*(ld))+(j))

//increase an int to the next multiple of VEC_LEN
#define PAD(i) ( ((i)%VEC_LEN)==0 ? (i):((i)/VEC_LEN)*VEC_LEN+VEC_LEN ) 

//ditto but for CL
#define CPAD(i) ( ((i)%CL)==0 ? (i):((i)/CL)*CL+CL )

//decrease an int to the next multiple of VEC_LEN
#define DPAD(i) ( ((i)%VEC_LEN)==0 ? (i):((i)/VEC_LEN)*VEC_LEN ) 

#define MAX(i,j) ((i) > (j) ? (i) : (j))
#define MAXI(i,j,k,l) ((i) > (j) ? (k) : (l))
#define MIN(i,j) ((i) < (j) ? (i) : (j))

typedef unsigned int unint;


typedef struct {
  real *mat;
  unint r; //rows
  unint c; //cols
  unint pr; //padded rows
  unint pc; //padded cols
  unint ld; //the leading dimension (in this code, this is the same as pc)
} matrix;


typedef struct {
  char *mat;
  unint r;
  unint c;
  unint pr;
  unint pc;
  unint ld;
} charMatrix;


typedef struct {
  unint *mat;
  unint r;
  unint c;
  unint pr;
  unint pc;
  unint ld;
} intMatrix;


typedef struct {
  unint* lr; //list of owned DB points
  unint len;  //length of lr
  real radius;
} rep;


typedef struct { //very simple list data type
  unint* x;
  unint len;
  unint maxLen;
} intList;

typedef struct { //array-based heap element
  unint id;
  real val;
} heapEl; 

typedef struct { //array-based heap
  heapEl *h;
  unint len;
} heap;
#endif
