/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef UTILS_H
#define UTILS_H

#include "defs.h"
#include<sys/time.h>

void swap(unint*,unint*);
void randPerm(unint,unint*);
unint randBetween(unint,unint);
void printMat(matrix);
void printMatWithIDs(matrix,unint*);
void printCharMat(charMatrix);
void printIntMat(intMatrix);
void printVector(real*,unint);
void copyVector(real*,real*,unint);
real distVec(matrix,matrix,unint,unint);
double timeDiff(struct timeval,struct timeval);
void copyMat(matrix*,matrix*);

void addToList(intList*,unint);
void createList(intList*);
void destroyList(intList*);
void printList(intList*);

void createHeap(heap*,unint);
void destroyHeap(heap*);
void replaceMax(heap*,heapEl);
void heapSort(heap*,unint*,real*);
void reInitHeap(heap*);

unint countBits(unsigned long);
#endif
