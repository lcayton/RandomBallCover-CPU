#ifndef RBC_H
#define RBC_H

#include<stdint.h>
#include "defs.h"

void furthestFirst(matrix,matrix);
void buildBit(matrix,matrix*,real*,uint32_t**,unint);
void getBitRep(matrix, matrix, real*, uint32_t**,unint);
void searchBit(uint32_t**, uint32_t**, unint, unint, unint, intList*,unint);
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
