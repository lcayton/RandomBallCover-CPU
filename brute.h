/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef BRUTE_H
#define BRUTE_H

#include<stdlib.h>
#include "defs.h"

void brutePar(matrix,matrix,unint*,real*);
void bruteK(matrix,matrix,size_t**,unint);
void bruteKDists(matrix,matrix,size_t**,real**,unint);
void bruteMap(matrix,matrix,rep*,unint*,unint*,real*);
void bruteList(matrix,matrix,rep*,intList*,unint,unint*,real*);
void rangeCount(matrix,matrix,real*,unint*);
#endif
