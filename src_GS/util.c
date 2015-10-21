//
// Created by Regina on 18/05/2015.
//

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "util.h"
#include <gsl/gsl_linalg.h>
#include <time.h>


const char *method = "a";

//TODO convert p->stuff to defined stuff
Parameters* setParameters(){
  Parameters* p;
  p = malloc(sizeof(Parameters));

  p->ratio = RATIO;
  p->tau = TAU;

  p->lrWidth = LRWIDTH;
  p->lrPatchWidth = LRPW;
  p->lrOverlap = LRO;
  p->hrWidth = p->lrWidth * p->ratio;                 //HR training images is (LR images * ratio)
  p->hrPatchWidth = p->lrPatchWidth * p->ratio;
  p->hrOverlap = p->lrOverlap * p->ratio;

  p->numTrainImages = NUMTRAINIMAGES;
  p->numTrainImagesToUse = MM;
  p->method = 'a';
  p->maxiter = MAXITER;        // max cg iterations
  return p;
}

Patch* divideToPatches(PGMData *data, const int numImages, Parameters* p, const char mode) {
  int width,pw,overlap;

  if (mode == 'l'){
    width = p->lrWidth;
    pw = p->lrPatchWidth;
    overlap = p->lrOverlap;
  }else if (mode == 'h'){
    width = p->hrWidth;
    pw = p->hrPatchWidth;
    overlap = p->hrOverlap;
  }else{
    fprintf(stderr, "Divide to patches: wrong mode!\n");
    exit(EXIT_FAILURE);
  }

  int numPatchX = (int)ceil((double)(width-overlap)/(double)(pw-overlap));// number of patches in x direction
  int numPatchXY = numPatchX*numPatchX;  // number of patches in one face image
  int step = pw - overlap;

  //    printf("dividing to patches, mode: %c\n", mode);
  //    printf("width: %d\n", width);
  //    printf("overlap: %d\n", overlap);
  //    printf("pw: %d\n", pw);
  //
  //    printf("num patch x: %d\n", numPatchX);
  //    fflush(stdout);

  Patch* patches;
  int n,i,j, patchNumber;
  patches = malloc(numPatchXY*sizeof(Patch));

  for (patchNumber=0;patchNumber<numPatchXY;patchNumber++) {
    //        printf("creating patch %d\n", patchNumber);

    patches[patchNumber].numPatchX = numPatchX;
    patches[patchNumber].row = numImages;
    patches[patchNumber].col = pw*pw;
    patches[patchNumber].matrix = allocate_dynamic_matrix_float(numImages, pw * pw);
    //Given a matrix for each patch, each row represents pixels of that patch of a single image,
    //each column represents the same pixel for different images.

    // start xy of each patch in each image
    int startX = ((patchNumber%numPatchX) * step) % width;
    int startY = floor(patchNumber / numPatchX) * step;

    // copy data from each original image to the patched version
    for (n = 0; n < numImages; n++) {
      for (j = 0; j < pw; j++) {
	for (i = 0; i < pw; i++) {
	  if (((startX + i) < width) && ((startY + j) < width))
	    patches[patchNumber].matrix[n][j * pw + i] = data[n].matrix[startX + i][startY + j];
	  else
	    patches[patchNumber].matrix[n][j * pw + i] = 0;     // zero pad right/bottom edges patches

	}
      }
    } // end for numImages
  } // end for patch number

  return patches;
}

// TODO perhaps consider numImages instead of just constrained to one patched image?
PGMData combinePatches(Patch* patches, Parameters* p) {
  printf("Combining patches\n");
  fflush(stdout);

  PGMData xHR;
  int width = p->hrWidth;
  int overlap = p->hrOverlap;
  int pw = p->hrPatchWidth;

  int numPatchX = (int)ceil((double)(width-overlap)/(double)(pw-overlap)); // number of patches in x direction
  int step = pw - overlap;

  int i, j, patch, startX, startY;
  int** counter;  // count how many patches contributed to the pixel, used as divider when averaging
  float** xHRmatrix;
  xHRmatrix = allocate_dynamic_matrix_float(width, width);
  counter = allocate_dynamic_matrix(width, width);

  for(patch=0;patch<numPatchX*numPatchX;patch++) {
    //        printf("\n patch %d\n", patch);
    for (i = 0; i < pw * pw; i++) {             //For each pixel
      //            printf("i %d\n", i);

      startX = ((patch % numPatchX) * step) % width;
      startY = floor(patch / numPatchX) * step;

      int ystep = floor(i / pw);

      //            printf("start X+i_pw is %d\n", startX + i % pw);
      //            printf("startY + ystep is %d\n", startY + ystep);
      //            printf("patches :%f\n",patches[patch].matrix[0][i]);
      //            assert(startX + i % pw < width);
      //            assert(startY + ystep < width);
      if ((startX + i % pw < width) && (startY + ystep < width)){         //Within the range
	xHRmatrix[startX + i % pw][startY + ystep] += patches[patch].matrix[0][i];
	counter[startX + i % pw][startY + ystep]++;
      }
    }
  }

  for (i = 0; i < width; i++) {
    for(j = 0; j < width; j++) {
      xHRmatrix[i][j] = xHRmatrix[i][j] / counter[i][j];          //Average out
      if(xHRmatrix[i][j]>1)
	xHRmatrix[i][j] = 1;
      else if(xHRmatrix[i][j]<0)
	xHRmatrix[i][j] = 0;
    }
  }

  xHR.matrix = xHRmatrix;
  xHR.col = width;
  xHR.row = width;
  xHR.max_gray = 255;

  deallocate_dynamic_matrix(counter, width);
  return xHR;
}

//一个test patch 和 一群image 的对应patch 的距离
float* calcDistance(Patch* testLRPatched, Patch* trainLRPatched) {
  float* dist;
  dist = malloc(sizeof(float)*trainLRPatched->row);

  int n, i;
  double tmp;
  for(n = 0; n<trainLRPatched->row; n++){
    tmp = 0.0;
    for(i = 0; i<trainLRPatched->col; i++){
      tmp += (testLRPatched->matrix[0][i] - trainLRPatched->matrix[n][i])*
	(testLRPatched->matrix[0][i] - trainLRPatched->matrix[n][i]);
    }
    dist[n] = sqrt(tmp);
  }

  return dist;
}

// IF A>B, RETURN 1; ELSE RETURN 0;
static int compareDist (const void *a, const void *b)
{
  int aa = *((int *) a), bb = *((int *) b);
  if (baseDistArray[aa] < baseDistArray[bb])
    return -1;
  if (baseDistArray[aa] == baseDistArray[bb])
    return 0;
  if (baseDistArray[aa] > baseDistArray[bb])
    return 1;
}


int *sortDistIndex(float* dist, int numElements) {
  int *idx, i;

  idx = malloc(sizeof(int)*numElements);

  // initialise initial index permutation
  for(i=0; i<numElements; i++){
    idx[i] = i;
  }

  // assign address of original dist array to the static global pointer
  // used by the compare function
  baseDistArray = dist;

  qsort(idx, numElements, sizeof(int), compareDist);

  return idx;
}

void getInitialWeight(float* w, float* w_image, int patch_num, int KNN){
  int n;
  int num_patch = (int)ceil((double)(LRWIDTH-LRO)/(double)(LRPW-LRO));
  if(patch_num == 0){
    for(n=0; n<MM; n++){
      w[n] = w_image[patch_num];
    }

  }else if(patch_num < num_patch){
    for(n=0; n<MM; n++){
      w[n] = w_image[patch_num-1];
    }
  
  }else if(patch_num%num_patch == 0){
    for(n=0; n<MM; n++){
      w[n] = w_image[patch_num-num_patch];   
    }

  }else if(KNN == 1){
    for(n=0; n<MM; n++){
      w[n] = (w_image[patch_num-1] + w_image[patch_num-num_patch]) / 2;
    }
  }else{
    for(n=0; n<MM; n++){
      w[n] = (w_image[patch_num-1] + w_image[patch_num-num_patch] + w_image[patch_num-num_patch-1] + w_image[patch_num-num_patch+1]) / 4;
    }
  }
  w_image[patch_num] = w[0]; //Update the weight for the current patch

}


