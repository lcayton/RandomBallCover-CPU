#ifndef RBC_H
#define RBC_H

#include "defs.h"
void buildBit(matrix,matrix*,real*,unsigned long*,unint);
void getBitRep(matrix, matrix, real*, unsigned long*);
void searchBit(unsigned long*, unsigned long*, unint, unint, unint, intList*);

void buildExact(matrix,matrix*,rep*,unint);
void searchExact(matrix,matrix,matrix,rep*,unint*);
void searchExactK(matrix,matrix,matrix,rep*,unint**,unint);
void searchExactManyCores(matrix,matrix,matrix,rep*,unint*);
void searchExactManyCoresK(matrix,matrix,matrix,rep*,unint**,unint);

void buildOneShot(matrix,matrix*,rep*,unint,unint);
void searchOneShot(matrix,matrix,matrix,rep*,unint*);
void searchOneShotK(matrix,matrix,matrix,rep*,unint**,unint);

void pickReps(matrix,matrix*);

void searchStats(matrix,matrix,matrix,rep*,double*);
#endif
