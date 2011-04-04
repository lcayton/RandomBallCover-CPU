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
void readData(char*,unint,unint,real*);
void readDataText(char*,unint,unint,real*);
void evalApprox(matrix,matrix,unint*);
double evalApproxK(matrix,matrix,unint**,unint);
void writeDoubs(int, char*, double,...);
void writeNeighbs(char*,unint**);


char *dataFileX, *dataFileQ, *dataFileXtxt, *dataFileQtxt, *outFile;
unint n=0, m=0, d=0, numReps=0, D=0, runBrute=0;
unint K = 1;

int main(int argc, char**argv){
  matrix x, q;
  unint i,j;
  struct timeval tvB,tvE;

  printf("********************************\n");
  printf("RANDOM BALL COVER -- CPU version\n");
  printf("********************************\n");
  
  parseInput(argc,argv);

  int threadMax = omp_get_max_threads();
  printf("number of threads = %d \n",threadMax);
  
  initMat( &x, n, D );
  initMat( &q, m, D );
  x.mat = (real*)calloc( sizeOfMat(x), sizeof(*(x.mat)) );
  q.mat = (real*)calloc( sizeOfMat(q), sizeof(*(q.mat)) );
  if( !x.mat || !q.mat ){
    fprintf(stderr, "memory allocation failure .. exiting \n");
    exit(1);
  }

  // If you are on a unix-based machine, I recommend using the following 
  // to alocate memory in an aligned way.  
  /* if( posix_memalign((void**)&x.mat, 64, sizeOfMat(x)*sizeof(*x.mat)) || */
  /*     posix_memalign((void**)&q.mat, 64, sizeOfMat(q)*sizeof(*q.mat)) ){ */
  /*   fprintf(stderr, "memory allocation failure .. exiting \n"); */
  /*   exit(1); */
  /* } */
  
  // Read data:
  if(dataFileXtxt)
    readDataText(dataFileXtxt, n, D, x.mat);
  else
    readData(dataFileX, n, D, x.mat);
  if(dataFileQtxt)
    readDataText(dataFileQtxt, m, D, q.mat);
  else
    readData(dataFileQ, m, D, q.mat);
 
  
  unint **NNs = (unint**)calloc( m, sizeof(*NNs) ); //indices of NNs
  real **dNNs = (real**)calloc( m, sizeof(*dNNs) ); //dists to NNs
  for(i=0;i<m; i++){
    NNs[i] = (unint*)calloc(K, sizeof(**NNs));
    dNNs[i] = (real*)calloc(K, sizeof(**dNNs) );
  }

  matrix rE; //matrix of representatives
  rep *riE = (rep*)calloc( CPAD(numReps), sizeof(*riE) ); //data struct for RBC
  
  // ******** builds the RBC
  gettimeofday(&tvB,NULL);
  buildExact(x, &rE, riE, numReps);
  gettimeofday(&tvE,NULL);
  double buildTime =  timeDiff(tvB,tvE);
  printf("exact build time elapsed = %6.4f \n", buildTime );
  

  // ******** queries the RBC
  gettimeofday(&tvB,NULL);
  if( threadMax<8 )
    searchExactK(q, x, rE, riE, NNs, dNNs, K);
  else
    searchExactManyCoresK(q, x, rE, riE, NNs, dNNs, K);
  gettimeofday(&tvE,NULL);
  double searchTime =  timeDiff(tvB,tvE);
  printf("exact k-nn search time elapsed = %6.4f \n", searchTime );

  //Uncomment the following to compute some stats about the search. 
  /* double avgDists; */
  /* searchStats(q,x,rE,riE,&avgDists); */
  /* printf("avgDists = %6.2f \n", avgDists); */
  

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

    if(outFile)
      writeDoubs(2, outFile, ((double)D), bruteTime );
    
    for(i=0; i<q.r; i++){
      for(j=0; j<K; j++){
	if(dNNs[i][j]!=dToNNsBrute[i][j]){
	  printf("%d,%d: %d %d %6.5f %6.5f \n", i, j, NNs[i][j], NNsBrute[i][j], distVec(q,x,i,NNs[i][j]), distVec(q,x,i,NNsBrute[i][j]));
	}
      }
    }
    
    free(NNsBrute);
    free(dToNNsBrute);
  }
  
  if(outFile)
    writeDoubs( 4, outFile, (double)numReps, ((double)D), buildTime, searchTime );


  //clean up
  freeRBC( rE, riE );

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
    printf("\nusage: \n  testRBC -x datafileX -q datafileQ -n numPts (DB) -m numQueries -d dim -r numReps [-o outFile] [-b] [-k neighbs]\n\n");
    printf("\tdatafileX    = binary file containing the database\n");
    printf("\tdatafileQ    = binary file containing the queries\n");
    printf("\tnumPts       = size of database\n");
    printf("\tnumQueries   = number of queries\n");
    printf("\tdim          = dimensionality\n");
    printf("\tnumReps      = number of representatives\n");
    printf("\toutFile      = output file (optional); stored in text format\n");
    //printf("\trDim         = reduced dimensionality\n"); 
    printf("\tneighbs      = num neighbors (optional; default is 1)\n"); 
    printf("\n\tuse -b option to run brute force search (in addition to the RBC)\n");
    printf("\n\nTo input data in text format (instead of bin):\n\tuse the -X and -Q switches in place of -x and -q\n");
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
    else if(!strcmp(argv[i], "-D")) 
      D = atoi(argv[++i]);
    else if(!strcmp(argv[i], "-r"))
      numReps = atoi(argv[++i]);
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

  if( !n || !m || !d || !numReps ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  if( (!dataFileX && !dataFileXtxt) || (!dataFileQ && !dataFileQtxt) ){
    fprintf(stderr,"more arguments needed.. exiting\n");
    exit(1);
  }
  if( (dataFileX && dataFileXtxt) || (dataFileQ && dataFileQtxt)){
    fprintf(stderr,"you can only give one database file and one query file.. exiting\n");
    exit(1); 
  }
  if( !D )
    D = d;
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


void writeNeighbs(char *file, unint **NNs){
  unint i,j;
  FILE *fp = fopen(file,"w");
  if( !fp ){
    fprintf(stderr, "can't open output file\n");
    return;
  }
  
  for( i=0; i<m; i++ ){
    for( j=0; j<K; j++ )
      fprintf( fp, "%u ", NNs[i][j] );
    fprintf(fp, "\n");
  }
  fclose(fp);

}
