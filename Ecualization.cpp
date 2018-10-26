#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include "omp.h"

using namespace std;
using namespace cv;

void histog(Mat &image, Mat &output){

    int x = image.cols;
    int y = image.rows;

    long totalSize = x*y;

    long hist[256] ={};

    cout << "Calculando histograma" << endl;
    // Calculando histograma
    for (int i = 0; i < y; i++){

        for (int j = 0; j < x; j++){

            unsigned int index = (int)image.at<uchar>(i,j);

            hist[index]++;
        }
    }

    cout << "Normalizando" << endl;
    // Normalizando 
    long hist_s[256]= {};

    for (int i = 0; i < 256; i++){
        
        for(int j = 0; j <= i; j++){
            
            hist_s[i] += hist[j];
        }

        unsigned int aux  = (hist_s[i]*255) /totalSize;

        hist_s[i] = aux;
    }

    cout << "Imagen final" << endl;
    // Rellenando la imagen final 

    for (int i = 0; i < y; i++){
        
        for(int j = 0; j < x; j++){

            unsigned int index = (int)image.at<uchar>(i,j);

            output.at<uchar>(i,j) = hist_s[index];
        }
    }
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

        cout << "Image size Step: "<< image.step << " Rows: " << image.rows << " Cols: " << image.cols << endl;

        cvtColor(image, grayImage, CV_BGR2GRAY);

        output = grayImage.clone();

        auto start_cpu =  chrono::high_resolution_clock::now();
        histog(grayImage, output);
        auto end_cpu =  chrono::high_resolution_clock::now();

        chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
        

        cout << "Time using CPU: " << duration_ms.count() << " ms" << endl;
        
        namedWindow("Input", cv::WINDOW_NORMAL);
        resizeWindow("Input", 800, 600);
        namedWindow("Output", cv::WINDOW_NORMAL);
        resizeWindow("Output", 800, 600);

        imshow("Input", grayImage);
        imshow("Output", output);
    }   

    waitKey(0);
    return 0;

}