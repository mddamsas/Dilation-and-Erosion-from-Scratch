#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

int main()
{
    // Create a demo image with some white blobs
    Mat demoImage = Mat::zeros(Size(10,10),CV_8U);
    demoImage.at<uchar>(0,0) = 1;
    demoImage.at<uchar>(0,9) = 1;
    demoImage.at<uchar>(9,0) = 1;
    demoImage.at<uchar>(9,9) = 1;
    demoImage(Range(3,5),Range(3,5)).setTo(1);
    demoImage(Range(5,8),Range(5,8)).setTo(1);

    // Save demo image
    Mat inputImage;
    resize(demoImage*255,inputImage,Size(500,500),0,0,INTER_NEAREST);
    cvtColor(inputImage,inputImage,COLOR_GRAY2BGR);
    imwrite("inputImage.png",inputImage);

    // Create an Ellipse Structuring Element
    Mat element = getStructuringElement(MORPH_CROSS, Size(3,3));
    int ksize = element.size().height;

    // Dilation from scratch
    int height, width;
    height = demoImage.size().height;
    width  = demoImage.size().width;
    int border = ksize/2;
    Mat paddedDemoImage = Mat::zeros(Size(height + border*2, width + border*2),CV_8UC1);
    copyMakeBorder(demoImage,paddedDemoImage,border,border,border,border,BORDER_CONSTANT,0);
    Mat paddedDilatedImage = paddedDemoImage.clone();
    Mat mask;
    Mat resizedFrame;
    double minVal, maxVal;

    VideoWriter dilation("dilationScratch.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(500,500));
    dilation.write(inputImage);
    
    for (int h_i = border; h_i < height + border; h_i++)
    {
        for (int w_i = border; w_i < width + border; w_i++)
        {
        
            bitwise_and(paddedDemoImage(Range(h_i-border,h_i+border+1),
                                        Range(w_i-border,w_i+border+1)), element, mask);
            minMaxIdx(mask,0, &maxVal,0,0,element);
            paddedDilatedImage.at<uchar>(h_i,w_i) = maxVal;
            resize(paddedDilatedImage(Range(border,height+border),Range(border,width+border))*255,resizedFrame,Size(500,500),0,0,INTER_NEAREST);
            cvtColor(resizedFrame,resizedFrame,COLOR_GRAY2BGR);
            dilation.write(resizedFrame);
        
        }
    }

    dilation.release();

    // Erosion from scratch
    paddedDemoImage = Mat::zeros(Size(height + border*2, width + border*2),CV_8UC1);
    copyMakeBorder(demoImage,paddedDemoImage,border,border,border,border,BORDER_CONSTANT,0);
    Mat paddedErodedImage = paddedDemoImage.clone();
    
    VideoWriter erosion("erosionScratch.avi", VideoWriter::fourcc('M','J','P','G'), 10, Size(500,500));
    erosion.write(inputImage);

    for (int h_i = border; h_i < height + border; h_i++)
    {
        for (int w_i = border; w_i < width + border; w_i++)
        {
        
            bitwise_and(paddedDemoImage(Range(h_i-border,h_i+border+1),
                                        Range(w_i-border,w_i+border+1)), element, mask);
        
        
            minMaxIdx(mask, &minVal,0,0,0,element);
            paddedErodedImage.at<uchar>(h_i,w_i) = minVal;
            resize(paddedErodedImage(Range(border,height+border),Range(border,width+border))*255,resizedFrame,Size(500,500),0,0,INTER_NEAREST);
            cvtColor(resizedFrame,resizedFrame,COLOR_GRAY2BGR);
            erosion.write(resizedFrame);

        }
    }

    erosion.release();

    return 0;

}
