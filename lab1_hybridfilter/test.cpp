/*
The program is the implement of hybrid with gray image.
*/

#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<iostream>
using namespace cv;
using namespace std;

int main()
{
    // 1. Read image
    double t = (double)getTickCount();
    Mat cat = imread("data\\cat.bmp");
    Mat dog = imread("data\\dog.bmp");

    if(!cat.data || !dog.data){
        cout<<"Can't find the image"<<std::endl;
        return -1;
    }

    // 2. Convert images to grayscale
    cv::Mat grayImage1, grayImage2;
    cv::cvtColor(cat, grayImage1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(dog, grayImage2, cv::COLOR_BGR2GRAY);

    // 3. Apply Gaussian blur to both images for low-pass filtering
    cv::Mat lowPassImage1, lowPassImage2;
    cv::GaussianBlur(grayImage1, lowPassImage1, cv::Size(21, 21), 50, 50);
    cv::GaussianBlur(grayImage2, lowPassImage2, cv::Size(21, 21), 50, 50);

    // 4. Apply Laplace blur to both images for high-pass filtering
    cv::Mat highPassImage1, highPassImage2;
    cv::Laplacian(grayImage1, highPassImage1, CV_8U, 1);
    cv::Laplacian(grayImage2, highPassImage2, CV_8U, 3);

    highPassImage1.convertTo(highPassImage1, lowPassImage2.type());   
    // 5. Combine the high-pass filtered image1 with the low-pass filtered image2 to create the hybrid image
    cv::Mat hybridImage;
    hybridImage = highPassImage1 + lowPassImage2;

    // 6. Normalize the hybrid image
    cv::normalize(hybridImage, hybridImage, 0, 255, cv::NORM_MINMAX);

    // calculate time
    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Hand written function time passed in seconds: " << t << endl;
    // 7. Save and display
    cv::imwrite("./hybrid_image.jpg", hybridImage);
    
    cv::imshow("lowPassImage2", lowPassImage2);
    cv::imshow("highPassImage1", highPassImage1);
    cv::imshow("Hybrid Image2", hybridImage);
    cv::namedWindow("Hybrid Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Hybrid Image", hybridImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
