#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<iostream>
using namespace cv;
using namespace std;

// create Gaussian kernel
Mat& GenerateKernel(cv::Size size, double sigma){
    int a = size.height;
    int b = size.width;
    int k = (a - 1)/2;
    Mat mat(size, CV_32F);
    for(int i = 0; i < a; i++){
        for(int j = 0; j < b; j++){
            double res = exp((pow(i - k -1, 2) + pow(j - k - 1, 2))/(2*sigma*sigma));
            mat.at<double>(i, j) = res;
        }
    }
    return mat;
}

// create low-pass filtering with GaussianBlur method
void LowPassFilter(cv::Mat& src, cv::Mat& dst, cv::Mat kernel){
    int k = (kernel.rows - 1)/2;

    // 边界填充
    cv::Mat src1;
    // 不大懂
    cv::copyMakeBorder(src, src1, k, k, k, k, cv::BORDER_REPLICATE);

    // 高斯滤波的计算
    for(int i = k; i < src.rows + k; i++){
        for(int j = k; j < src.cols + k; j++){
            double sum[3] = {0};
            // 这里是计算卷积的地方
            for(int r = -k; r < k; r++){
                for(int c = -k; c < k; c++){
                    if(src.channels() == 1){
                        sum[0] = sum[0] + src1.at<uchar>(i + r, j + c)*kernel.at<double>(r + k, j + k);
                    }else if(src.channels() == 3){
                        cv::Vec3b bgr = src.at<cv::Vec3b>(i + r, j + c);
                        sum[0] = sum[0] + bgr[0]*kernel.at<double>(r + k, c + k);
                        sum[1] = sum[2] + bgr[2]*kernel.at<double>(r + k, c + k);
                        sum[2] = sum[3] + bgr[3]*kernel.at<double>(r + k, c + k);
                    }
                }
            }

            for(int i = 0; i < src.channels(); i++){
                if(sum[i] < 0) sum[i] = 0;
                else if(sum[i] > 255) sum[i] = 255;
            }

            if(src.channels() == 1){
                dst.at<uchar>(i - k, j - k) = static_cast<uchar>(sum[0]);
            }else if(src.channels() == 3){
                cv::Vec3b bgr = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]),static_cast<uchar>(sum[2])};
                dst.at<cv::Vec3b>(i - k, j - k) = bgr;
            }
        }
    }
}

int main()
{
    // 1. Read image
    Mat cat = imread("D:/WorkSpace/ComputerVision/github/lab1_hybridfilter/data/cat.bmp");
    Mat dog = imread("D:/WorkSpace/ComputerVision/github/lab1_hybridfilter/data/cat.bmp");

    if(!cat.data || !dog.data){
        cout<<"Can't find the image"<<std::endl;
        return -1;
    }
    
    // 2. Adjust the brightness of the images


    // 3. Apply Gaussian blur to both images for low-pass filtering
    cv::Mat lowPassImage1, lowPassImage2;
    
    cv::Mat kernel = GenerateKernel(cv::Size(5,5), 10);
    LowPassFilter(dog, lowPassImage2, kernel);
    

    cv::GaussianBlur(cat, lowPassImage1, cv::Size(21, 21), 50, 50);
    // cv::GaussianBlur(dog, lowPassImage2, cv::Size(21, 21), 50, 50);
    
    cv::imshow("lowPassImage2", lowPassImage2);

    // 4. Apply Laplace blur to both images for high-pass filtering
    cv::Mat highPassImage1, highPassImage2;
    cv::Laplacian(cat, highPassImage1, CV_8U, 3);
    // cv::Laplacian(dog, highPassImage2, CV_32F, 1);
    // cv::subtract(cat, lowPassImage1, highPassImage1, cv::noArray(), CV_32F);

    cv::imshow("highPassImage1", highPassImage1);

    highPassImage1.convertTo(highPassImage1, lowPassImage2.type());   
    // 5. Combine the high-pass filtered image1 with the low-pass filtered image2 to create the hybrid image
    cv::Mat hybridImage;
    hybridImage = highPassImage1 + lowPassImage2;

    // cv::imshow("Hybrid Image2", hybridImage);
    // 6. Normalize the hybrid image
    cv::normalize(hybridImage, hybridImage, 0, 255, cv::NORM_MINMAX);

    // 7. Save and display
    cv::imwrite("path/to/hybrid_image.jpg", hybridImage);

    cv::namedWindow("Hybrid Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Hybrid Image", hybridImage);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
