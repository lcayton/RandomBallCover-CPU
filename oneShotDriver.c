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
#include "dists.h"


void parseInput(int,char**);
void readData(char*,matrix);
void readDataText(char*,matrix);
void evalApprox(matrix,matrix,unint*);
double evalApproxK(matrix,matrix,unint**,unint);
void writeDoubs(int, char*, double,...);
void writeNeighbs(char*,char*,unint**,real**);

char *dataFileX, *dataFileQ, *dataFileXtxt, *dataFileQtxt, *outFile, *outFiletxt;
unint n=0, m=0, d=0, numReps=0, runBrute=0, runEval=0;
unint K = 1;

int main(int argc, char**argv){
  matrix x, q;
  unint i;
  struct timeval tvB,tvE;

  printf("********************************\n");
  printf("RANDOM BALL COVER -- CPU version\n");
  printf("********************************\n");
  
  parseInput(argc,argv);

  int threadMax = omp_get_max_threads();
  printf("number of threads = %d \n",threadMax);
  
  initMat( &x, n, d );
  initMat( &q, m, d );
  x.mat = (real*)calloc( sizeOfMat(x), sizeof(*(x.mat)) );
  q.mat = (real*)calloc( sizeOfMat(q), sizeof(*(q.mat)) );
  if( !x.mat || !q.mat ){
    fprintf(stderr, "memory allocation failure .. exiting \n");
    exit(1);
  }
  
  // Read data:
  if(dataFileXtxt)
    readDataText(dataFileXtxt, x);
  else
    readData(dataFileX, x);
  if(dataFileQtxt)
    readDataText(dataFileQtxt, q);
  else
    readData(dataFileQ, q);
 
  
  unint **NNs = (unint**)calloc( m, sizeof(*NNs) ); //indices of NNs
  real **dNNs = (real**)calloc( m, sizeof(*dNNs) ); //dists to NNs
  for(i=0;i<m; i++){
    NNs[i] = (unint*)calloc(K, sizeof(**NNs));
    dNNs[i] = (real*)calloc(K, sizeof(**dNNs) );
  }

  matrix r; //matrix of representatives
  rep *ri = (rep*)calloc( CPAD(numReps), sizeof(*ri) ); //data struct for RBC
  
  // ******** builds the RBC
  gettimeofday(&tvB,NULL);
  buildOneShot(x, &r, ri, numReps);
  gettimeofday(&tvE,NULL);
  double buildTime =  timeDiff(tvB,tvE);
  printf("one-shot build time elapsed = %6.4f \n", buildTime );
  

  // ******** queries the RBC
  gettimeofday(&tvB,NULL);
  searchOneShotK(q, x, r, ri, NNs, dNNs, K);
  gettimeofday(&tvE,NULL);
  double searchTime =  timeDiff(tvB,tvE);
  printf("one-shot k-nn search time elapsed = %6.4f \n", searchTime );
  
  if( runEval )
    evalApproxK(q, x, NNs, K);

  // ******** runs brute force search.
  if(runBrute){
    printf("running brute force search..\n");
    unint **NNsBrute = (unint**)calloc( m, sizeof(*NNsBrute) );
    real **dToNNsBrute = (real**)calloc( m, sizeof(*dToNNsBrute) );;
    for(i=0; i<m; i++){
      NNsBrute[i] = (unint*)calloc( K, sizeof(**NNsBrute) );
      dToNNsBrute[i] = (real*)calloc( K, sizeof(**dToNNsBrute) );
    }
    gettimeofday(&tvB,NULL);
    bruteK(x,q,NNsBrute,dToNNsBrute,K);
    gettimeofday(&tvE,NULL);
    double bruteTime = timeDiff(tvB,tvE);
    printf("brute time elapsed = %6.4f \n", bruteTime );
        
    free(NNsBrute);
    free(dToNNsBrute);
  }
  
  if( outFile || outFiletxt )
    writeNeighbs( outFile, outFiletxt, NNs, dNNs );
  

  //clean up
  freeRBC( r, ri );

  for(i=0;i<m; i++){
    free(NNs[i]);    free(dNNs[i]);
  }
  free(NNs); free(dNNs);
  free(x.mat);
  free(q.mat);
  
  return 0;
}





void parseInput(int argc, char **argv){
  int i=1;
  if(argc <= 1){
    printf("\nusage: \n  oneShotRBC -x datafileX -q datafileQ -n numPts (DB) -m numQueries -d dim -r numReps [-o outFile] [-b] [-e] [-k neighbs]\n\n");
    printf("\tdatafileX    = binary file containing the database\n");
    printf("\tdatafileQ    = binary file containing the queries\n");
    printf("\tnumPts       = size of database\n");
    printf("\tnumQueries   = number of queries\n");
    printf("\tdim          = dimensionality\n");
    printf("\tnumReps      = number of representatives\n");
    printf("\toutFile      = binary output file (optional)\n");
    printf("\tneighbs      = num neighbors (optional; default is 1)\n"); 
    printf("\n\tuse -b to run brute force search (in addition to the RBC)\n");
    printf("\n\tuse -e to run the evaluation procedure.  \n\tNOTE: this procedure takes as long as brute force.\n");
    printf("\n\n\tTo input/output data in text format (instead of bin), use the \n\t-X and -Q and -O switches in place of -x and -q and -o (respectively).\n");
    printf("\n\n");
    exit(0);
  }
  
  while(i<argc){
    if(!strcmp(argv[i], "-x"))
      dataFileX = argv[++i];
    else if(!strcmp(argv[i], "-q"))
      dataFileQ = argv[++i];
    else if(!strcmp(argv[i], "-X"))
      dataFileXtxt = argv[++i];
    else if(!strcmp(argv[i], "-Q"))
      dataFileQtxt = argv[++i];
    else if(!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-m"))
      m = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-d"))
      d = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-r"))
      numReps = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-b"))
      runBrute = 1;
    else if(!strcmp(argv[i], "-e"))
      runEval = 1;
    else if(!strcmp(argv[i], "-o"))
      outFile = argv[++i];
    else if(!strcmp(argv[i], "-O"))
      outFiletxt = argv[++i];
    else if(!strcmp(argv[i], "-k"))
      K = atoi(argv[++i]);
    else{
      fprintf(stderr,"%s : unrecognized option.. exiting\n",argv[i]);
      exit(1);
    }
    i++;
  }

  if( !n || !m || !d || !numReps ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  if( (!dataFileX && !dataFileXtxt) || (!dataFileQ && !dataFileQtxt) ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  if( (dataFileX && dataFileXtxt) || (dataFileQ && dataFileQtxt) ){
    fprintf(stderr,"you can only give one database file and one query file and one ouput file.. exiting\n");
    exit(1); 
  }
  if(numReps>n){ 
    fprintf(stderr,"can't have more representatives than points.. exiting\n"); 
    exit(1); 
  } 
  if(K>n){
    fprintf(stderr,"K can't be greater than the number of points.. exiting\n"); 
    exit(1); 
  }
}


// loads data from a binary file into data in row-major order.
void readData(char *dataFile, matrix x){
  FILE *fp;
  unint numRead;
  unint i;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }
    
  for( i=0; i<x.r; i++ ){ //can't load everything in one fread
                           //because matrix is padded.
    numRead = fread( &x.mat[IDX(i, 0, x.ld)], sizeof(real), x.c, fp);
    if(numRead != x.c){
      fprintf(stderr,"error reading file.. exiting \n");
      exit(1);
    }
  }
  fclose(fp);
}


void readDataText(char *dataFile, matrix x){
  FILE *fp;
  real t;
  int i,j;

  fp = fopen(dataFile,"r");
  if(fp==NULL){
    fprintf(stderr,"error opening file.. exiting\n");
    exit(1);
  }
    
  for(i=0; i<x.r; i++){
    for(j=0; j<x.c; j++){
      if(fscanf(fp,"%f ", &t)==EOF){
	fprintf(stderr,"error reading file.. exiting \n");
	exit(1);
      }
      x.mat[IDX( i, j, x.ld )]=(real)t;
    }
  }
  fclose(fp);
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


void writeNeighbs(char *file, char *filetxt, unint **NNs, real **dNNs){
  unint i,j;
  
  if( filetxt ) { //write text

    FILE *fp = fopen(filetxt,"w");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }
    
    for( i=0; i<m; i++ ){
      for( j=0; j<K; j++ )
	fprintf( fp, "%u ", NNs[i][j] );
      fprintf(fp, "\n");
    }
    
    for( i=0; i<m; i++ ){
      for( j=0; j<K; j++ )
	fprintf( fp, "%f ", dNNs[i][j] );
      fprintf(fp, "\n");
    }
    fclose(fp);
    
  }

  if( file ){ //write binary

    FILE *fp = fopen(file,"wb");
    if( !fp ){
      fprintf(stderr, "can't open output file\n");
      return;
    }
    
    for( i=0; i<m; i++ )
      fwrite( NNs[i], sizeof(*NNs[i]), K, fp );
    for( i=0; i<m; i++ )
      fwrite( dNNs[i], sizeof(*dNNs[i]), K, fp );
    
    fclose(fp);
  }
}
