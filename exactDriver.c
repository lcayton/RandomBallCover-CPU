#include<omp.h>
#include<string.h>
#include<stdarg.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<math.h>
#include "defs.h"
#include "utils.h" 
#include "brute.h"
#include "rbc.h"
void testHam(matrix,matrix);
void parseInput(int,char**);
void readData(char*,unint,unint,real*);
void readDataText(char*,unint,unint,real*);
void orgData(real*,unint,unint,matrix,matrix);
void evalApprox(matrix,matrix,unint*);
double evalApproxK(matrix,matrix,unint**,unint);
void writeDoubs(int, char*, double,...);

char *dataFile, *outFile;
unint n=0, m=0, d=0, numReps=0, s=0, D=0, runBrute=0;
unint K = 1;

int main(int argc, char**argv){
  real *data;
  matrix x, q;
  unint i;
  struct timeval tvB,tvE;

  printf("********************************\n");
  printf("RANDOM BALL COVER -- CPU version\n");
  printf("********************************\n");

  parseInput(argc,argv);

  unint pm = CPAD(m);
  data  = (real*)calloc( (n+m)*d, sizeof(*data) );
  x.r = n; x.c = D; x.pr = CPAD(n); x.pc = PAD(D); x.ld = x.pc;
  q.r = m; q.c = D; q.pr = CPAD(m); q.pc = PAD(D); q.ld = q.pc;

  /* x.mat = (real*)calloc( PAD(n)*PAD(d), sizeof(*(x.mat)) ); */
  /* q.mat = (real*)calloc( PAD(m)*PAD(d), sizeof(*(q.mat)) ); */

  //alocates memory in an aligned way
  if( posix_memalign((void**)&x.mat,64,x.pr*x.pc*sizeof(*x.mat)) || 
      posix_memalign((void**)&q.mat,64,q.pr*q.pc*sizeof(*q.mat)) ){
    fprintf(stderr, "memory allocation failure .. exiting \n");
    exit(1);
  }
  
  readData(dataFile, (n+m), d, data);
  orgData(data, (n+m), d, x, q);
  free(data);
  
  unint *NNs = calloc(pm, sizeof(*NNs));; 

  int threadMax = omp_get_max_threads();
  printf("number of threads = %d \n",threadMax);

  if(runBrute){
    unint *NNsBrute = (unint*)calloc( pm, sizeof(*NNsBrute) );
    real *dToNNs = (real*)calloc( pm, sizeof(*dToNNs) );;
    gettimeofday(&tvB,NULL);
    brutePar(x,q,NNsBrute,dToNNs);
    gettimeofday(&tvE,NULL);
    double bruteTime = timeDiff(tvB,tvE);
    printf("brute time elapsed = %6.4f \n", bruteTime );
    if(outFile)
      writeDoubs(1,outFile,bruteTime);
    free(NNsBrute);
    free(dToNNs);
  }

 
  matrix rE;
  rep *riE = (rep*)calloc( CPAD(numReps), sizeof(*riE) );
  
  gettimeofday(&tvB,NULL);
  buildExact(x, &rE, riE, numReps);
  gettimeofday(&tvE,NULL);
  double buildTime =  timeDiff(tvB,tvE);
  printf("exact build time elapsed = %6.4f \n", buildTime );

  gettimeofday(&tvB,NULL);
  searchExactManyCores(q, x, rE, riE, NNs);
  gettimeofday(&tvE,NULL);
  double searchTime =  timeDiff(tvB,tvE);
  printf("one-shot search time elapsed = %6.4f \n", searchTime );

  double avgDists;
  searchStats(q,x,rE,riE,&avgDists);

  
  if(outFile)
    writeDoubs(5,outFile,(double)numReps,((double)D),buildTime,searchTime,avgDists);

  free(rE.mat);
  for(i=0; i<rE.pr; i++)
    free(riE[i].lr);
  free(riE);
  free(NNs);
  free(x.mat);
  free(q.mat);
  return 0;
}





void parseInput(int argc, char **argv){
  int i=1;
  if(argc <= 1){
    printf("\nusage: \n  testRBC -f datafile (bin) -n numPts (DB) -m numQueries -d dim -r numReps -s numPtsPerRep [-D rDim] [-o outFile] [-b] [-k neighbs]\n\n");
    printf("\tdatafile     = binary file containing the data\n");
    printf("\tnumPts       = size of database\n");
    printf("\tnumQueries   = number of queries\n");
    printf("\tdim          = dimensionality\n");
    printf("\tnumReps      = number of representatives\n");
    printf("\tnumPtsPerRep = number of points assigned to each representative\n");
    printf("\toutFile      = output file (optional); stored in text format\n");
    printf("\trDim         = reduced dimensionality\n"); 
    printf("\tneighbs      = num neighbors\n"); 
    printf("\n\n");
    exit(0);
  }
  
  while(i<argc){
    if(!strcmp(argv[i], "-f"))
      dataFile = argv[++i];
    else if(!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-m"))
      m = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-d"))
      d = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-D"))
      D = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-r"))
      numReps = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-s"))
      s = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-b"))
      runBrute = 1;
    else if(!strcmp(argv[i], "-o"))
      outFile = argv[++i];
    else if(!strcmp(argv[i], "-k"))
      K = atoi(argv[++i]);
    else{
      fprintf(stderr,"%s : unrecognized option.. exiting\n",argv[i]);
      exit(1);
    }
    i++;
  }

  if( !n || !m || !d || !numReps || !s || !dataFile ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  
  if( !D )
    D = d;
  /* if(numReps>n){ */
  /*   fprintf(stderr,"can't have more representatives than points.. exiting\n"); */
  /*   exit(1); */
  /* } */
}


void readData(char *dataFile, unint rows, unint cols, real *data){
  FILE *fp;
  unint numRead;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }
    
  numRead = fread(data,sizeof(real),rows*cols,fp);
  if(numRead != rows*cols){
    fprintf(stderr,"error reading file.. exiting \n");
    exit(1);
  }
  fclose(fp);
}


void readDataText(char *dataFile, unint rows, unint cols, real *data){
  FILE *fp;
  real t;
  int i,j;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }
    
  for(i=0; i<rows; i++){
    for(j=0; j<cols; j++){
      if(fscanf(fp,"%f ", &t)==EOF){
	fprintf(stderr,"error reading file.. exiting \n");
	exit(1);
      }
      data[IDX( i, j, cols )]=(real)t;
    }
  }
  fclose(fp);
}

//This function splits the data into two matrices, x and q, of 
//their specified dimensions.  The data is split randomly.
//It is assumed that the number of rows of data (the parameter n)
//is at least as large as x.r+q.r
void orgData(real *data, unint n, unint d, matrix x, matrix q){
   
  unint i,fi,j;
  unint *p;
  p = (unint*)calloc(n,sizeof(*p));
  
  randPerm(n,p);

  for(i=0,fi=0 ; i<x.r ; i++,fi++){
    for(j=0;j<x.c;j++){
      x.mat[IDX(i,j,x.ld)] = data[IDX(p[fi],j,d)];
    }
  }

  for(i=0 ; i<q.r ; i++,fi++){
    for(j=0;j<q.c;j++){
      q.mat[IDX(i,j,q.ld)] = data[IDX(p[fi],j,d)];
    } 
  }

  free(p);
}


void evalApprox(matrix q, matrix x, unint *NNs){
  real *ranges = (real*)calloc(q.pr, sizeof(*ranges));
  unint *counts = (unint*)calloc(q.pr,sizeof(*counts));
  unint i;

  for(i=0; i<q.r; i++)
    ranges[i] = distVec(q,x,i,NNs[i]);

  rangeCount(x,q,ranges,counts);

  double avgCount = 0.0;
  for(i=0; i<q.r; i++)
    avgCount += ((double)counts[i]);
  avgCount/=q.r;
  printf("average num closer = %6.5f \n",avgCount);

  free(counts);
  free(ranges);
}


double evalApproxK(matrix q, matrix x, unint **NNs, unint K){
  unint i,j,k;
  struct timeval tvB, tvE;
  unint **nnCorrect = (unint**)calloc(q.pr, sizeof(*nnCorrect));
  real **dT = (real**)calloc(q.pr, sizeof(*dT));
  for(i=0; i<q.pr; i++){
    nnCorrect[i] = (unint*)calloc(K, sizeof(**nnCorrect));
    dT[i] = (real*)calloc(K, sizeof(**dT));
  }

  gettimeofday(&tvB,NULL); 
  bruteK(x, q, nnCorrect, dT, K);
  gettimeofday(&tvE,NULL); 
  
  unsigned long ol = 0;
  for(i=0; i<q.r; i++){
    for(j=0; j<K; j++){
      for(k=0; k<K; k++){
	ol += (NNs[i][j] == nnCorrect[i][k]);
      }
    }
  }
  printf("avg overlap = %6.4f / %d\n", ((double)ol)/((double)q.r), K);
  printf("(bruteK took %6.4f seconds) \n",timeDiff(tvB,tvE));
  
  for(i=0; i<q.pr; i++){
    free(nnCorrect[i]);
    free(dT[i]);
  }
  free(nnCorrect);
  free(dT);
  
  return ((double)ol)/((double)q.r);
}


void writeDoubs(int num, char* file, double x,...){
  double z;
  int i;

  FILE*fp = fopen(file,"a");
  va_list s;
  va_start(s,x);
  
  for(z=x, i=0; i<num; z=va_arg(s,double),i++)
    fprintf(fp,"%6.5f ",z);
  fprintf(fp,"\n");
  fclose(fp);
}
