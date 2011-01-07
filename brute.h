/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef BRUTE_H
#define BRUTE_H

#include<stdlib.h>
#include "defs.h"

void brutePar(matrix,matrix,unint*,real*);
void bruteK(matrix,matrix,unint**,real**,unint);
void bruteKHeap(matrix, matrix,unint**,real**, unint);
void bruteMap(matrix,matrix,rep*,unint*,unint*,real*);
void bruteMapK(matrix,matrix,rep*,unint*,unint**,real**,unint);
void bruteListK(matrix,matrix,rep*,intList*,unint,unint**,real**,unint);
void rangeCount(matrix,matrix,real*,unint*);
#endif
