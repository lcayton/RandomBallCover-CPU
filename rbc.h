#ifndef RBC_H
#define RBC_H

#include "defs.h"
void buildExact(matrix,matrix*,rep*,unint);
void searchStats(matrix,matrix,matrix,rep*,double*);
void searchExact2(matrix,matrix,matrix,rep*,unint*);
void searchExact3(matrix,matrix,matrix,rep*,unint*);
void searchExact4(matrix,matrix,matrix,rep*,unint*);
void buildDefeat(matrix,matrix*,rep*,unint,unint);
void searchDefeat(matrix,matrix,matrix,rep*,unint*);
void pickReps(matrix,matrix*);
void dimEst(matrix,unint);
#endif
