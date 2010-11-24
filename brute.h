/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef BRUTE_H
#define BRUTE_H

#include<stdlib.h>
#include "defs.h"

void brute(matrix,matrix,unint*,real*);
void bruteCache(matrix,matrix,unint*,real*);
void brutePar(matrix,matrix,unint*,real*);
void brutePar2(matrix,matrix,unint*,real*);
void brutePar3(matrix,matrix,unint*,real*);
void bruteK(matrix,matrix,size_t**,unint);
void bruteK2(matrix,matrix,size_t**,unint);
void bruteKDists(matrix,matrix,size_t**,real**,unint);
void bruteMap(matrix,matrix,rep*,unint*,unint*,real*);
void bruteMap2(matrix,matrix,rep*,unint*,unint*,real*);
void bruteMapSort(matrix,matrix,rep*,unint*,unint*,real*);
void bruteList(matrix,matrix,rep*,intList*,unint*,real*);
void bruteListRev(matrix,matrix,rep*,intList*,unint,unint*,real*);
void rangeCount(matrix,matrix,real*,unint*);
void rangeCount2(matrix,matrix,real*,unint*);
#endif
