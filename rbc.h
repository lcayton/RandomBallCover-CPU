#ifndef RBC_H
#define RBC_H

#include "defs.h"
void buildExact(matrix,matrix*,rep*,unint);
void searchStats(matrix,matrix,matrix,rep*,double*);
void searchExact(matrix,matrix,matrix,rep*,unint*);
void searchExactK(matrix,matrix,matrix,rep*,unint*,unint);
void searchExactManyCores(matrix,matrix,matrix,rep*,unint*);
void buildOneShot(matrix,matrix*,rep*,unint,unint);
void searchOneShot(matrix,matrix,matrix,rep*,unint*);
void pickReps(matrix,matrix*);
#endif
