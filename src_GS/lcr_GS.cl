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

