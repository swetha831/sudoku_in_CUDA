// Simple SUDOKU probram in CUDA
// cmpe297_hw3_easysudoku.cu

#include<stdio.h>
#include<string.h>
#include <cuda_runtime.h>

const int big_2x[9][9] = {{1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1},
			  {1, 1, 1, 1, 1, 1, 1, 1, 1}};

// input 9x9 sudoku : 
// - 1~9 : valid values 
// - 0 : no value is decided
const int input_sdk[9][9] =  {{0, 7, 0, 0, 6, 5, 0, 8, 0},
			  {6, 0, 0, 0, 3, 0, 4, 0, 0},
			  {0, 2, 0, 0, 4, 0, 7, 0, 0},
			  {8, 6, 0, 0, 0, 2, 5, 7, 0},
			  {0, 0, 7, 4, 0, 6, 1, 0, 0},
			  {0, 5, 2, 3, 0, 0, 0, 6, 4},
			  {0, 0, 8, 0, 2, 0, 0, 3, 0},
			  {0, 0, 5, 0, 8, 0, 0, 0, 1},
			  {0, 4, 0, 7, 1, 0, 0, 5, 0}};
typedef struct {
	int val[9][9]; // values that each entry can get
	int num_options[9][9]; // number of values that each entry can get
	int not_in_cell[9][9];	// values not in each 3x3 cell
	int not_in_row[9][9];	// values not in each row
	int not_in_col[9][9];	// values not in each column
} stContext;
stContext context;

void initialize_all();
void print_all();

#define WIDTH	9

#define IS_OPTION(row, col, k) \
			((l_context->not_in_row[row][k] == 1) && \
			(l_context->not_in_col[col][k] == 1) && \
			(l_context->not_in_cell[row/3+(col/3)*3][k] == 1))? 1 : 0;
// rule: numbers should be unique in each sub-array, each row, and each column


__global__ void k_Sudoku(stContext *l_context)
{
	const unsigned int col = threadIdx.x;
	const unsigned int row = threadIdx.y;
	// TODO: Insert Your Code Here
		while(l_context->num_options[row][col] > 1)
				{
					// Find values that are not in the row, col, and the 
					// 3x3 cell that (row, col) is belonged to.			
					int value = 0, temp;
					l_context->num_options[row][col] = 0;

					for(int k = 0; k < 9; k++)
					{
						temp = IS_OPTION(row, col, k);
						if(temp == 1)
						{
							l_context->num_options[row][col]++;
							value = k;

						}
					}

					// If the above loop found only one value, 
					// set the value to (row, col)
					if(l_context->num_options[row][col] == 1)
					{
						l_context->not_in_row[row][value] = 0;
						l_context->not_in_col[col][value] = 0;
						l_context->not_in_cell[(row)/3+((col)/3)*3][value] = 0;
						//C[row][col]= value +1;
						l_context->val[row][col] = value+1;
					}
				}

}

int main(int argc, char **argv)
{
    cudaError_t err;

    initialize_all();
    print_all();

    stContext *k_context;

    err =cudaMalloc((void**)&k_context, sizeof(stContext));  // TODO: Allocate matrix in GPU device memory 
    if (err != cudaSuccess)
    {
	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    err =cudaMemcpy(k_context,&context,sizeof(stContext),cudaMemcpyHostToDevice); // TODO: Copy the input matrix to GPU
    if (err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy data from host to device (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }

    // Assign as many threads as the matrix size so that
    // each thread can deal with one entry of the matrix
    dim3 dimBlock(WIDTH, WIDTH, 1);
    dim3 dimGrid(1, 1, 1);

    // TODO: Call the kernel function 
    float elapsedTime;
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    k_Sudoku<<<dimGrid,dimBlock>>> (k_context);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
	fprintf(stderr, "Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

    err =cudaMemcpy(&context, k_context, sizeof(stContext), cudaMemcpyDeviceToHost);  // TODO: Copy the result matrix from the GPU device memory
    if (err != cudaSuccess)
    {
	fprintf(stderr, "Failed to copy data from device to host (error code %s)!\n", cudaGetErrorString(err));
	exit(EXIT_FAILURE);
    }

    cudaThreadSynchronize();
	
    // Print the result
    print_all();
    printf("GPU Time elapsed: %f millisecond",elapsedTime);
    // Free the device memory
    err = cudaFree(k_context);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free gpu data (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    getchar();
	
    return 0;
}

void initialize_all()
{
    int i, j;

    memcpy(context.not_in_cell,big_2x, sizeof(big_2x));	
    memcpy(context.not_in_row,big_2x, sizeof(big_2x));	
    memcpy(context.not_in_col,big_2x, sizeof(big_2x));	
		
    for(i=0; i<9; i++){
    	for(j=0; j<9; j++){
            if(input_sdk[i][j] == 0)
            {
                context.val[i][j] = 0;
                context.num_options[i][j]=9;
            }
            else
            {
                context.val[i][j] = input_sdk[i][j];
                context.num_options[i][j] = 1;
                context.not_in_cell[i/3+(j/3)*3][input_sdk[i][j]-1] = 0;
                context.not_in_col[j][input_sdk[i][j]-1] = 0;
                context.not_in_row[i][input_sdk[i][j]-1] = 0;
            }
        }
    }
}


void print_all()
{
    int i, j, k;

    for(i=0; i<9; i++){
        for(j=0; j<9; j++){
            if(context.val[i][j] == 0)
                fprintf(stdout, "  %1d   ", context.val[i][j]);  
            else
                fprintf(stdout, " *%1d*  ", context.val[i][j]);  
            if((j==2)||(j==5)){
                fprintf(stdout, "| ");	
            }
        }
        fprintf(stdout, "\n");	
        if((i==2)||(i==5)){
            for(k=0; k<69; k++){
                fprintf(stdout, "-");	
            }
            fprintf(stdout, "\n");	
        }
    }
    fprintf(stdout, "\n");
}

