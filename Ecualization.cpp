#include <iostream>
#include <cstdio>
#include <stdlib.h>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include "omp.h"


int main (int argc, char** argv){

    if (argc < 2){
         
        cout << "No hay argumentos suficientes" << endl;
    }
    else {
        
        Mat image;
        Mat grayImage;

        image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

        cvtColor(image, grayImage, CV_BGR2GRAY);

        namedWindow("Original", cv::WINDOW_NORMAL);
        namedWindow("Output", cv::WINDOW_NORMAL);

        imshow("Original", image);
        imshow("Output", grayImage);

    }

}