#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "load_pgm.h"
#include "util.h"
#include "time.h"
#include <sys/time.h>
#include <assert.h>
#include "cl_setup.h"
#define INITIAL_WEIGHT 0.00278
#define KNN 1
#define TIME_OFFSET 1000000  //Offset used in order to acquire the time in correct form
#define AOCL_ALIGNMENT 64

/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1)
{
  long int diff = (t2->tv_usec + TIME_OFFSET * t2->tv_sec) - (t1->tv_usec + TIME_OFFSET * t1->tv_sec);
  result->tv_sec = diff / TIME_OFFSET;
  result->tv_usec = diff % TIME_OFFSET;

  return (diff<0);
}

int main(int argc, char *argv[]){

  printf("----Iterative Method----\n");
  static Parameters* p;
  p = setParameters();

  //----------------- read training images ------------------------------------------
  int i;
  char istr[3];
  PGMData trainSetLR[NUMTRAINIMAGES];
  PGMData trainSetHR[NUMTRAINIMAGES];

  printf("reading training set\n");
  fflush(stdout);

  for (i=0;i<NUMTRAINIMAGES;i++){
    char fnLR[80];
    char fnHR[80];
    

    sprintf(istr, "%d", i+1);     // convert i with offset to string
    
    // get file name
    
    //strcpy(fnHR, "bin/trainHR/FaceHR_");
    strcpy(fnHR, argv[1]);
    strcat(fnHR, istr);
    strcat(fnHR, ".pgm\0");
      
    //strcpy(fnLR, "bin/trainLR/Face_");
    strcpy(fnLR, argv[2]);
    strcat(fnLR, istr);
    strcat(fnLR, ".pgm\0");

    readPGM(fnHR, &(trainSetHR[i]));
    readPGM(fnLR, &(trainSetLR[i]));
    
  }

  //------------------ Training images: divide to patches ----------------------------
  Patch *trainLRPatched;
  Patch *trainHRPatched;
  trainLRPatched = divideToPatches(trainSetLR, NUMTRAINIMAGES, p, 'l');   //Divide into patches
  trainHRPatched = divideToPatches(trainSetHR, NUMTRAINIMAGES, p, 'h');

  printf("done dividing patch\n");
  fflush(stdout);

  double PatchingTime = 0.0;
  double CombineTime = 0.0;
  double clPrepTime = 0.0;
  cl_ulong  KernelTime = 0.0;
  double RunTime = 0.0;

  clock_t check_start;
  clock_t check_end;

  
  struct timeval tvBegin, tvEnd, tvDiff, tvPatch, tvGS, tvCombine; //Recoding time starts from now 
 
  gettimeofday(&tvBegin, NULL);

  cl_platform_id platform = setOpenCLPlatform();
  cl_context context = createOpenCLContext(platform);
  cl_device_id device = getOpenCLDevices(context);

  cl_program program = createOpenCLProgram(context, device, argv[5]);

  cl_int ret;
  
  //To check the maxComputeUnit we have
  size_t maxComputeUnits;
  ret = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, NULL);
  assert(ret == CL_SUCCESS);
  printf("maxComputeUnits = %d\n", maxComputeUnits);

  // create opencl kernel
  cl_kernel GSeidel = clCreateKernel(program, "GSeidel", &ret);   //Gauss Seidel Method
  assert(ret==CL_SUCCESS);

  // create opencl command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &ret);
  assert(ret==CL_SUCCESS);

  cl_ulong time_start, time_end;

  cl_event kernel_event;
  cl_event read_events[1];
  cl_event write_events[4];
    
  //sizes in 3 dimension
  size_t local_size[3] = {30, 1, 1};        //Specify local work-group size
  size_t global_size[3] = {MM, MM, 10};     //Specify global work item size

  //size_t global_size = 1;
  //size_t local_size = 1;

  //   size_t local_size[2] = {30, 1};
  //  size_t global_size[2] = {MM, 10};

  // create opencl memory objects
  int byte_size = sizeof(float)* MM* MM;
  cl_mem G_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
  assert(ret==CL_SUCCESS);
  cl_mem w_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
  assert(ret==CL_SUCCESS);
  /*
  cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
  assert(ret==CL_SUCCESS);
  cl_mem w_prev_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, byte_size, NULL, &ret);
  assert(ret==CL_SUCCESS);  
  */

  //  end = clock();
  gettimeofday(&tvEnd, NULL);
  
  timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
  clPrepTime = ((double)(tvDiff.tv_sec + tvDiff.tv_usec) / TIME_OFFSET);   //Opencl Preparation time
  int numPatch = trainLRPatched->numPatchX * trainLRPatched->numPatchX; //Number of Patches = number of patches * number of patches
  printf("numpatch = %d\n", numPatch);
  int patch;
  Patch* testHRPatched = malloc(sizeof(Patch)*numPatch);;
  for(patch=0; patch < numPatch; patch++) {

    testHRPatched[patch].matrix = allocate_dynamic_matrix_float(1, p->hrPatchWidth * p->hrPatchWidth);
    testHRPatched[patch].col = trainHRPatched->col;
    testHRPatched[patch].row = 1;
    testHRPatched[patch].numPatchX = trainLRPatched->numPatchX;
  }

  gettimeofday(&tvBegin, NULL);
  
  //------------------- LcR -----------------------------------------------------------
  for(i=0; i<NUMTESTFACE;i++) {

    //------------------- read test image -------------------------------------------
    // printf("reading test image\n");
    fflush(stdout);
   
    char fnTestLR[80];
    char fnTestHROut[80];
   
   
    // get file name
    sprintf(istr, "%d", i+1);     // convert i to string
    strcpy(fnTestLR, argv[3]);
    //strcpy(fnTestLR, "bin/testLR/FaceTest_");
    strcat(fnTestLR, istr);
    strcat(fnTestLR, ".pgm\0");

    char mm[3];                         //mm
    char n[5];                          //n

    sprintf(mm, "%dx", MM);
    sprintf(n, "%do%d", N, LRO);

    //Specify the output name and directory
    strcpy(fnTestHROut, argv[4]);
    //strcpy(fnTestHROut, "bin/Result/");
    strcat(fnTestHROut, mm);
    strcat(fnTestHROut, n);
    strcat(fnTestHROut, "_");
    strcat(fnTestHROut, istr);
    strcat(fnTestHROut, ".pgm\0");
   
    PGMData testLR;
    readPGM(fnTestLR, &(testLR));
    Patch *testLRPatched;
    
    gettimeofday(&tvPatch, NULL);
    testLRPatched = divideToPatches(&testLR, 1, p, 'l');                //Something Write to testLRPatched
    gettimeofday(&tvEnd, NULL);
    timeval_subtract(&tvDiff, &tvEnd, &tvPatch);
    PatchingTime += ((double)(tvDiff.tv_sec + tvDiff.tv_usec) / TIME_OFFSET);   //Patching Time

    //------------------------ reconstruction ----------------------------------------

    assert(p->numTrainImagesToUse<=p->numTrainImages);
    int hrPW = p->hrPatchWidth;

    float **C, *G, *w_image;   
    C = allocate_dynamic_matrix_float(N, sizeof(float)*MM*MM);
    posix_memalign((void**)&G, AOCL_ALIGNMENT, sizeof(float)*MM*MM);
    w_image = malloc(sizeof(float)*numPatch);
    
    int m;
    for(m=0; m<numPatch; m++){
      w_image[m] = INITIAL_WEIGHT;
    }

    for(patch=0; patch < numPatch; patch++){

      // get distance between xLR(patch) and all yLR(patch)
      float* dist;        // size all_Y
      dist = calcDistance(&(testLRPatched[patch]), &(trainLRPatched[patch]));

      // sort dist and get sort index (an index array that contains the order of sorting)
      int* idx;
      idx = sortDistIndex(dist, trainLRPatched->row);

      // get weights
      int i,j,k;

      float* w;
  
      posix_memalign((void**)&w, AOCL_ALIGNMENT, sizeof(float)*MM*MM);
      
      int n;
      for(n=0; n<MM*MM; n++){
	w[n] = 0;
      }

      getInitialWeight(w, w_image, patch, KNN);

      // loop over the closest `numTrainImagesToUse' trainLR
      for(i=0; i<testLRPatched->col; i++) {
	for(j=0; j<p->numTrainImagesToUse; j++){
	  C[i][j] = testLRPatched[patch].matrix[0][i] - trainLRPatched[patch].matrix[idx[j]][i];  //The difference between a pixel from a patch of test image
	                                                                            //and its respective pixel from the first numTrainImageToUse train images 
	}
      }


      for(i=0; i<p->numTrainImagesToUse; i++) {
	for(j =0; j <p->numTrainImagesToUse; j++) {
	  G[i * p->numTrainImagesToUse + j] = 0;
	  for (k = 0; k < testLRPatched->col; k++) {
	    G[i * p->numTrainImagesToUse + j] += C[k][i] * C[k][j];
	  }
	  if(j == i) {
	    G[i * p->numTrainImagesToUse + j] += p->tau * dist[idx[i]] * dist[idx[i]];
	  }
	}
      }

      //---------- Gauss Seidel Method ------------------------------------------------------------
      // write to buffer
     
      ret = clEnqueueWriteBuffer(command_queue, G_buffer, CL_TRUE, 0, byte_size, G, 0, NULL, &write_events[0]);
      assert(ret==CL_SUCCESS);
      ret = clEnqueueWriteBuffer(command_queue, w_buffer, CL_TRUE, 0, byte_size, w, 0, NULL, &write_events[1]);
      assert(ret==CL_SUCCESS);
  
        // set kernel arguments
      ret = clSetKernelArg(GSeidel, 0, sizeof(cl_mem), &w_buffer);
      assert(ret==CL_SUCCESS);
      ret = clSetKernelArg(GSeidel, 1, sizeof(cl_mem), &G_buffer);
      assert(ret==CL_SUCCESS);
      /*
      ret = clSetKernelArg(GSeidel, 2, sizeof(cl_mem), &b_buffer);
      assert(ret==CL_SUCCESS);
      ret = clSetKernelArg(GSeidel, 3, sizeof(cl_mem), &w_prev_buffer);
      assert(ret==CL_SUCCESS);
      */

      // run kernel
      // Run a 3 dimensional kernel
      ret = clEnqueueNDRangeKernel(command_queue, GSeidel, (cl_uint) 3,
      				   NULL, global_size, local_size, 2, write_events, &kernel_event);
     
      //  ret = clEnqueueNDRangeKernel(command_queue, GSeidel, (cl_uint) 2,
      //				   NULL, global_size, local_size, 2, write_events, &kernel_event[0]);
      
    
      //ret = clEnqueueNDRangeKernel(command_queue, GSeidel, (cl_uint) 1, // one dimension
      // 				   NULL, &global_size, &local_size, 2, write_events, &kernel_event);
     
      //      printf("ret = %d\n", ret);
      assert(ret==CL_SUCCESS);
 
      ret = clWaitForEvents(1, &kernel_event);
      assert(ret == CL_SUCCESS);
      

      clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
      clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
      KernelTime = KernelTime + (time_end - time_start);
    // Read weight
      ret = clEnqueueReadBuffer(command_queue, w_buffer, CL_TRUE, 0, sizeof(float)*MM, w, 0, NULL, &read_events[0]);
      assert(ret==CL_SUCCESS);
      ret = clWaitForEvents(1, &read_events[0]);
      assert(ret==CL_SUCCESS);

      //------------------- construct HR patch ----------------------------------------------
      double sum_x = 0.0;
      for(i=0; i<p->numTrainImagesToUse; i++){
	sum_x += w[i];
      }
            
      for(i=0;i<testHRPatched->col;i++){
	double tmp = 0.0;
	for(j=0;j<p->numTrainImagesToUse;j++){
	  tmp += (w[j]/sum_x)*trainHRPatched[patch].matrix[idx[j]][i];       //Math thing...
	}
	testHRPatched[patch].matrix[0][i] = tmp;
      }

      // free allocated memory
      free(dist);
      free(idx);
      free(w);
    }
    
    deallocate_dynamic_matrix_float(C,N);
    free(G);
    free(w_image);
    
         

    //------------------------ combine patches ---------------------------------------
    PGMData testRecombinedImage;
    gettimeofday(&tvCombine, NULL);
    testRecombinedImage = combinePatches(testHRPatched, p);
    gettimeofday(&tvEnd, NULL);
    timeval_subtract(&tvDiff, &tvEnd, &tvCombine);
    CombineTime += ((double) (tvDiff.tv_sec + tvDiff.tv_usec) / TIME_OFFSET);  //CombineTime
   
    //Save Result
    writePGM(fnTestHROut, &(testRecombinedImage));
 

    for(patch=0; patch < numPatch; patch++)
      deallocate_dynamic_matrix_float(testLRPatched[patch].matrix, testLRPatched[patch].row);

    free(testLRPatched);
  }
    
  printf("----Timings:----\n");
  printf("-- Timing for %d testing images -- \n", NUMTESTFACE);
  
  gettimeofday(&tvEnd, NULL);
  timeval_subtract(&tvDiff, &tvEnd, &tvBegin);
  RunTime = ((double)(tvDiff.tv_sec + tvDiff.tv_usec) / TIME_OFFSET);
  printf("everything(time for opencl_setup, LcR, Reconstruction and combination and save for HR pates): %ld.%06ld s\n", tvDiff.tv_sec, tvDiff.tv_usec);
 
  printf("patchingTime: %fs\ncombinePatchTime: %fs\n", PatchingTime, CombineTime);
  printf("Total time taken to process one test image: %fs \n", (PatchingTime+RunTime)/NUMTESTFACE);
  printf("clPrepTime: %fs \n", clPrepTime);
  printf("Kernel Time: %luns \n", KernelTime);


  fflush(stdout);


  for(patch=0; patch < numPatch; patch++){
    deallocate_dynamic_matrix_float(testHRPatched[patch].matrix, testHRPatched[patch].row);
  }
  free(testHRPatched);
  free(p);


  // Clean up OpenCL stuff
  ret = clReleaseKernel(GSeidel);
  assert(ret==CL_SUCCESS);
  ret = clReleaseProgram(program);
  assert(ret==CL_SUCCESS);
  ret = clReleaseCommandQueue(command_queue);
  assert(ret==CL_SUCCESS);
  ret = clReleaseContext(context);
  assert(ret==CL_SUCCESS);
  ret = clReleaseMemObject(G_buffer);
  assert(ret==CL_SUCCESS);
  ret = clReleaseMemObject(w_buffer);
  assert(ret==CL_SUCCESS);
  /*
  ret = clReleaseMemObject(b_buffer);
  assert(ret==CL_SUCCESS);
  ret = clReleaseMemObject(w_prev_buffer);
  assert(ret==CL_SUCCESS);  
  */
  free(trainLRPatched);
  free(trainHRPatched);


  return 0;
}
