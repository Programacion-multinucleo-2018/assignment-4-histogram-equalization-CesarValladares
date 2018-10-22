#include <iostream>
#include <cstdio>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>

#include "common.h"
#include <cuda_runtime.h>

#include <chrono>

using namespace std;
using namespace cv;

__global__ void histog_kernel(unsigned char* input, int width, int height, int grayWidthStep, long totalSize, int * global){

    //2D Index of current thread
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

    unsigned int nxy = threadIdx.x + threadIdx.y * blockDim.x;

    //Location of gray pixel in output
    const int gray_tid  = iy * grayWidthStep + ix;

    __shared__ int hist[256];

    if(nxy < 256){
        hist[nxy] = 0;
    }

    __syncthreads();
    
    //Only valid threads perform memory I/O
	if((ix<width) && (iy<height))
	{   
        int Index = input[gray_tid];
        atomicAdd(&hist[Index], 1);
    }
    
    __syncthreads();

    if (nxy < 256){

        atomicAdd(&global[nxy], hist[nxy]);

    }

}

__global__ void Normal(int * hist , int * hist_s, float totalSize){

    
    unsigned int nxy = threadIdx.x + threadIdx.y * blockDim.x;

    if (nxy < 256 && blockIdx.x == 0 && blockIdx.y == 0){

        for(int i = 0; i < nxy; i++){

            hist_s[nxy]+=hist[i];

        }

        hist_s[nxy] = hist_s[nxy]*(255/totalSize);

    }

}

__global__ void Create_Image(unsigned char* input, unsigned char* output, int width, int height, int grayWidthStep, int *hist_s)
{
	//2D Index of current thread
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;

	//Location of gray pixel in output
    const int gray_tid  = iy * grayWidthStep + ix;

	if( ix < width && iy < height)
	{
        int Index = input[gray_tid];
		output[gray_tid] = hist_s[Index];
	}
}

void histog(const Mat& input, Mat& output)
{
	//Calcu late total number of bytes, input and output image are both gray scale
    size_t grayBytes = input.step * input.rows;
    float totalSize = output.rows * output.cols;

    int * hist_s = {};
    int * hist = {};

    float average = 0;

	unsigned char *d_input, *d_output;

	//Allocate device memory
	SAFE_CALL(cudaMalloc(&d_input,grayBytes),"CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_output,grayBytes),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&hist_s,256*sizeof(int)),"CUDA Malloc Failed");
    SAFE_CALL(cudaMalloc(&hist,256*sizeof(int)),"CUDA Malloc Failed");

	//Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), grayBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
    SAFE_CALL(cudaMemcpy(d_output, output.ptr(), grayBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	//Specify a reasonable block size
	const dim3 block(16,16);

	//Calculate grid size to cover the whole image
    const dim3 grid((input.cols)/block.x, (input.rows)/block.y);

    auto start_cpu =  chrono::high_resolution_clock::now();

	//Launch the color conversion kernel
	histog_kernel<<<grid,block>>>(d_input,input.cols,input.rows, input.step, totalSize, hist);
    Normal<<<grid,block>>>(hist, hist_s, totalSize);
    Create_Image<<<grid,block>>>(d_input, d_output, output.cols, output.rows, output.step, hist_s);

    auto end_cpu =  chrono::high_resolution_clock::now();

    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
    average = duration_ms.count();

    //Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");

    printf("histog_kernel <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, average);

	//Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(),d_output,grayBytes,cudaMemcpyDeviceToHost),"CUDA Memcpy Host To Device Failed");

	//Free the device memory
	SAFE_CALL(cudaFree(d_input),"CUDA Free Failed");
    SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
    SAFE_CALL(cudaFree(hist_s), "CUDA Free Failed");
    SAFE_CALL(cudaFree(hist), "CUDA Free Failed");
}

int main (int argc, char** argv){


    if (argc < 2){
         
        cout << "No hay argumentos suficientes" << endl;
    }
    else {
        
        Mat image;
        Mat grayImage(image.rows,image.cols,CV_8UC1);
        Mat output(image.rows,image.cols,CV_8UC1);

        image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

        cvtColor(image, grayImage, CV_BGR2GRAY);

        output = grayImage.clone();

        cout << "Image size Step: "<< grayImage.step << " Rows: " << image.rows << " Cols: " << image.cols << endl;
        
        
        histog(grayImage, output);      

        //Allow the windows to resize
        namedWindow("Input", cv::WINDOW_NORMAL);
        resizeWindow("Input", 800, 600);
        namedWindow("Output", cv::WINDOW_NORMAL);
        resizeWindow("Output", 800, 600);

        //Show the input and output
        imshow("Input", grayImage);
        imshow("Output", output);  

        imwrite( "Images/Gray_Image.jpg", output);
            
    }   

    waitKey(0);
    return 0;

}