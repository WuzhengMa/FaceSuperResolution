//Author: Wuzheng Ma
//Email: oliverm0919@gmail.com
//Date: 08.25.2015

//This file includes all versions of GS method, the first one is the unoptimized code,
//the last one is the final code I used

//Base on the website https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
//There are two ways to implement GS method, one uses both the x from current iteration and x from the previous
//iteration, the other one uses the x from current iteration only.
//From my experements, the one uses both current and previous iterations provides a slightly slower performance
//than the one uses the current iteration only.



//Unoptimized GS method
#define NUM 30
#define NUM_ITER 10

__kernel void GSeidel(global float* restrict x, global float* restrict A, global float* restrict b,
                      global float* restrict  x_prev){
    
    int row, column, iter;
    float temp_sum, temp_x; //max_error, temp_diff
    int m, converge = 0;
    
    for(m=0; m<NUM; m++){
        b[m] = 1; //Initialize b to all 1s
    }
    
    for(iter=0; iter<NUM_ITER; iter++){
        for(m=0; m<NUM; m++){
            x_prev[m] = x[m];
        }
        for(row=0; row<NUM; row++){
            temp_sum = 0;
            for(colu1mn=0; column<NUM; column++){
                if(column != row){
                    if(column < row){
                        temp_sum = temp_sum + A[row*NUM + column] * x[column]; //Lower side of Matrix
                    }else{
                        temp_sum = temp_sum + A[row*NUM + column] * x_prev[column]; //Upper side
                    }
                }
            }
            x[row] = (b[row] - temp_sum) / A[row*NUM + row]; //Update x
        }
    }
}


//Unroll the out-most loop, use both x_prev and x
#define N 30
#define NUM_ITER 10

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void GSeidel(global float* restrict x, global float* restrict A, global float* restrict b,
                      global float* restrict  x_prev){
    
    int row, column;
    float temp_sum, temp_x; //max_error, temp_diff
    int m, converge = 0;
    int iter = get_global_id(0);
    
    for(m=0; m<N; m++){
        b[m] = 1; //Initialize b to all 1s
    //    x[m] = 0; //Take a initial guess of 0 for x[m]
    //    x_prev[m] = 0;
    }
    
    if(iter < NUM_ITER){
        x_prev = x;    //This is not correct
        for(row=0; row<N; row++){
            temp_sum = 0;
            for(column=0; column<N; column++){
                if(column != row){
                    if(column < row){
                        //                      printf("iter%d, column%d, row%d\n", iter, column, row);
                        temp_sum = temp_sum + A[row*N + column] * x[column]; //Lower side of Matrix
                    }else{
                        //                      printf("iter%d, column%d, row%d\n", iter, column, row);
                        temp_sum = temp_sum + A[row*N + column] * x_prev[column]; //Upper side
                    }
                }
            }
            x[row] = (b[row] - temp_sum) / A[row*N + row]; //Update x
        }
    }
}




//Unroll two outter loops, use x only
#define NUM 30
#define ERROR 0.001
#define NUM_ITER 10

__attribute__((reqd_work_group_size(30, 10, 1)))
__kernel void GSeidel(global float* restrict x, global float* restrict A, global float* restrict b){
    
    float temp_sum = 0;
    int column = 0;
    int m = get_global_id(0);
    int row = get_global_id(0);
    int iter = get_global_id(1);
    
    //    event_t copy_event_from = async_work_group_copy(x_local, x, NUM, 0);
    //    wait_group_events(1, &copy_event_from);
    
    b[m] = 1; //Initialize b to all 1s
    
    
    if(iter < NUM_ITER){
        
        for(column=0; column<NUM; column++){
            if(column != row){
                if(column < row){
                    //                                            printf("Before: iter%d, column%d, row%d temp_sum = %f, x[column] = %f\n", iter, column, row, temp_sum, x[column]);
                    temp_sum = temp_sum + A[row*NUM + column] * x[column]; //Lower side of Matrix
                }else{
                    //                                            printf("Before: iter%d, column%d, row%d, temp_sum = %f, x[column] = %f\n", iter, column, row, temp_sum, x[column]);
                    temp_sum = temp_sum + A[row*NUM + column] * x[column]; //Upper side
                }
            }
        }
        x[row] = (b[row] - temp_sum) / A[row*NUM + row]; //Update x
        //                  printf("update: x[row] = %f\n", x[row]);
    }
    
}



//loops fully unrolled, slow performance used x and x_prev
#define NUM 30
#define ERROR 0.001
#define NUM_ITER 10

__attribute__((reqd_work_group_size(30, 30, 10)))
__kernel void GSeidel(__global float* restrict x, __global float* restrict A, __global float* restrict b,
                      __local float* restrict x_local, __local float* restrict x_prev){
    
    __local float temp_sum;
    __local int check_row;
    __local int check_iter;
    int m = get_global_id(0);
    int iter = get_global_id(2);
    int row = get_global_id(1);
    int column = get_global_id(0);
    
    event_t copy_event_from = async_work_group_copy(x_local, x, NUM, 0);
    wait_group_events(1, &copy_event_from);
    
    b[m] = 1;   //Initialize b to all 1s
    printf("m = %d\n", m);
    
    if(iter < NUM_ITER){
        
        
        if((iter == check_iter) && (row == 0)){         //Update the value from last iteration
            x_prev[column] = x_local[column];
            if(column == 29){check_iter++;}
            if(check_iter == 10){check_iter = 0;}
            printf("enter here onec per iteration and obtain x_prev[column] = x_local[column] = %f\n", x_local[column]);
        }
        
        if(row == check_row){           //Reset the temp_sum for every row
            temp_sum = 0;
            check_row++;
            if(check_row == 30){check_row = 0;}
            printf("enter here once per row and reset temp_sum\n");
        }
        
        
        if(column != row){
            if(column < row){
                printf("iter%d, column%d, row%d, before cal temp_sum = %f and x[column] = %f\n", iter, column, row, temp_sum, x_local[column]);
                temp_sum = temp_sum + A[row*NUM + column] * x_local[column];  //Lower side of Matrix
                printf("after cal temp_sum = %f\n", temp_sum);
            }else{
                printf("iter%d, column%d, row%d, before cal temp_sum = %f and x_prev[column] = %f\n", iter, column, row, temp_sum, x_prev[column]);
                temp_sum = temp_sum + A[row*NUM + column] * x_prev[column];  //Upper side
                printf("after cal temp_sum = %f\n", temp_sum);
            }
        }
        
        x_local[row] = (b[row] - temp_sum) / A[row*NUM + row]; //Update x
        printf("Result: x[row] = %f\n", x_local[row]);
        printf("end of an iteration \n\n\n");
    }
    event_t copy_event_to = async_work_group_copy(x, x_local, NUM, 0);
    wait_group_events(1, &copy_event_to);
}


//loops fully unrolled, use x only
#define NUM 30
#define ERROR 0.001
#define NUM_ITER 10

__attribute__((reqd_work_group_size(30, 6, 1)))
__kernel void GSeidel(__global float* restrict x, __global float* restrict A, __global float* restrict b,
                      __global float* restrict temp_sum, __global int* restrict check_row){
    
    //    __local float temp_sum;
    //    __local int check_row;
    int m = get_global_id(0);
    int iter = get_global_id(2);
    int row = get_global_id(1);
    int column = get_global_id(0);
    
    //    event_t copy_event_from = async_work_group_copy(x_local, x, NUM, 0);
    //    wait_group_events(1, &copy_event_from);
    
    b[m] = 1;   //Initialize b to all 1s
    
    if(iter < NUM_ITER){
        
        if(row == *check_row){           //Reset the temp_sum for every row
            *temp_sum = 0;
            *check_row = *check_row + 1;
            if(*check_row == 30){*check_row = 0;}
            //                    printf("enter here once per row and reset temp_sum %d\n", *check_row);
        }
        
        
        if(column != row){
            if(column < row){
                //            printf("iter%d, column%d, row%d, before cal temp_sum = %f and x[column] = %f\n", iter, column, row, *temp_sum, x[column]);
                *temp_sum = *temp_sum + A[row*NUM + column] * x[column];  //Lower side of Matrix
                //            printf("after cal temp_sum = %f\n", temp_sum);
            }else{
                //            printf("iter%d, column%d, row%d, before cal temp_sum = %f and x[column] = %f\n", iter, column, row, *temp_sum, x[column]);
                *temp_sum = *temp_sum + A[row*NUM + column] * x[column];  //Upper side
                //            printf("after cal temp_sum = %f\n", temp_sum);
            }
        }
        
        x[row] = (b[row] - *temp_sum) / A[row*NUM + row]; //Update x
        //    printf("Update x[row] = %f\n", x[row]);
        //    printf("end of an iteration \n\n\n");
    }
    
    //    event_t copy_event_to = async_work_group_copy(x, x_local, NUM, 0);
    //    wait_group_events(1, &copy_event_to);
}



//Final version
//loops fully unrolled, use x only
#define NUM 30
#define NUM_ITER 10

__attribute__((reqd_work_group_size(30, 1, 1)))
__kernel void GSeidel(__global float* restrict x, __global float* restrict A){
    
    int m = get_global_id(0);
    int iter = get_global_id(2);
    int row = get_global_id(1);
    int column = get_global_id(0);
    local float temp_sum;
    
    if(iter < NUM_ITER){
        if(column != row){
            temp_sum = temp_sum + A[row*NUM + column] * x[column];
        }
        
        x[row] = (1 - temp_sum) / A[row*NUM + row]; //Update x, "1" in the equation stands for b which contains a array\
        of ones
        if(column == 29){temp_sum = 0;}
    }
    
}







