/*Before trying to construct hybrid images, it is suggested that you
implement myFilter() and then debug it using proj1_test_filtering.cpp */


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat myFilter(Mat, Mat, int);
Mat hybrid_image_visualize(Mat);
Mat DFT_Spectrum(Mat);

// find the nearest point to x, y inside the image
double get_val_with_replicate(Mat im, int x, int y, int c){
	int rx = (x < 0) ? 0 : im.cols - 1;
	int ry = (y < 0) ? 0 : im.rows - 1;
	return im.at<Vec3d>(ry, rx)[c];
}

// find the mirror position of x, y inside the image
double get_val_with_reflect(Mat im, int x, int y, int c){
	int rx = (x < 0) ? -x : (im.cols - 1) * 2 - x;
	int ry = (y < 0) ? -y : (im.rows - 1) * 2 - y;	
	return im.at<Vec3d>(ry, rx)[c];
}


enum border { Border_Replicate, Border_Reflect, Border_Constant };

Mat myFilter(Mat im, Mat filter, int borderType = Border_Replicate)
{
	/*This function is intended to behave like the built in function filter2D()
	Your function should work for color images. Simply filter each color
	channel independently.
	Your function should work for filters of any width and height
	combination, as long as the width and height are odd(e.g. 1, 7, 9).This
	restriction makes it unambigious which pixel in the filter is the center
	pixel.
	Boundary handling can be tricky.The filter can't be centered on pixels
	at the image boundary without parts of the filter being out of bounds.
	There are several options to deal with boundaries. -- pad the input image with zeros, and
	return a filtered image which matches the input resolution. A better
	approach is to mirror the image content over the boundaries for padding.*/

	Mat outI = im.clone();
	int channels = im.channels();
	Vec3d acc;
	double val;

	// iterate over input matrix (cloned as outI)
	for (int y= 0; y < outI.rows; y++){
		for (int x= 0; x < outI.cols; x++){
			// reset color accumulator
			acc = Vec3d(0, 0, 0);
			// iterate over channels
			for (int c=0; c < channels; c++){
				// iterate over filter
				for (int j= 0; j < filter.rows; j++){
					for (int i= 0; i < filter.cols; i++){
						int target_x = (j - filter.rows / 2) + x;
						int target_y = (i - filter.cols / 2) + y;
						
						// if boundary, handle it according to border type
						if (target_x < 0 || target_y < 0 || target_x >= im.cols || target_y >= im.rows){
							switch(borderType){
								case Border_Constant:
									val = 0.0; break;
								case Border_Replicate:
									val = get_val_with_replicate(im, target_x, target_y, c); break;
								case Border_Reflect:
									val = get_val_with_reflect(im, target_x, target_y, c); break;
								default:
									cout << "ERROR! borderType is not valid." << std::endl;
							}
						} else {
							// directly get the pixel from input image
							val = im.at<Vec3d>(target_y, target_x)[c];
						}

						// append product of pixel and filter scalar to accumulator
						acc[c] += val * filter.at<double>(j, i);
					}
				}
				// update output image with accumulated color
				outI.at<Vec3d>(y, x)[c] = acc[c];
			}
		}
	}
	return outI;
}


Mat hybrid_image_visualize(Mat hybrid_image)
{
	//visualize a hybrid image by progressively downsampling the image and
	//concatenating all of the images together.		
	int scales = 5; //how many downsampled versions to create		
	double scale_factor = 0.5; //how much to downsample each time		
	int padding = 5; //how many pixels to pad.
	int original_height = hybrid_image.rows; // height of the image
	int num_colors = hybrid_image.channels(); //counting how many color channels the input has
	Mat output = hybrid_image;
	Mat cur_image = hybrid_image;

	for (int i = 2; i <= scales; i++)
	{
		//add padding
		hconcat(output, Mat::ones(original_height, padding, CV_8UC3), output);

		//dowsample image;
		resize(cur_image, cur_image, Size(0, 0), scale_factor, scale_factor, INTER_LINEAR);

		//pad the top and append to the output
		Mat tmp;
		vconcat(Mat::ones(original_height - cur_image.rows, cur_image.cols, CV_8UC3), cur_image, tmp);
		hconcat(output, tmp, output);
	}

	return output;
}

Mat DFT_Spectrum(Mat img)
{
	/*
	This function is intended to return the spectrum of an image in a displayable form. Displayable form
	means that once the complex DFT is calculated, the log magnitude needs to be determined from the real 
	and imaginary parts. Furthermore the center of the resultant image needs to correspond to the origin of the spectrum.
	*/

	vector<Mat> im_channels(3);
	split(img, im_channels);
	img = im_channels[0];

	/////////////////////////////////////////////////////////////////////
	//STEP 1: pad the input image to optimal size using getOptimalDFTSize()
	Mat padded_img;
	// calculate frame size
	int border_y = getOptimalDFTSize(img.rows) - img.rows;
	int border_x = getOptimalDFTSize(img.cols) - img.cols;
	copyMakeBorder(img, padded_img, 0, border_y, 0, border_x, BORDER_CONSTANT, Scalar(0));

	
	///////////////////////////////////////////////////////////////////
	//STEP 2:  Determine complex DFT of the image. 
	// Use the function dft(src, dst, DFT_COMPLEX_OUTPUT) to return a complex Mat variable.
	// The first dimension represents the real part and second dimesion represents the complex part of the DFT 
	Mat dft_img;
	// get dft
	dft(padded_img, dft_img, DFT_COMPLEX_OUTPUT);
	// split DFT results into real and complex
	vector<Mat> parts(2);
	split(dft_img, parts);
	Mat real = parts[0];
	Mat complex = parts[1];
	////////////////////////////////////////////////////////////////////
	//Step 3: compute the magnitude and switch to logarithmic scale
	//=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	
	Mat magI;
	// calculate magnitude
	magnitude(real, complex, magI);
	// add 1
	magI += Scalar(1);
	// apply log on matrix
	log(magI, magI);

	///////////////////////////////////////////////////////////////////
	// Step 4: 
	/* For visualization purposes the quadrants of the spectrum are rearranged so that the 
	   origin (zero, zero) corresponds to the image center. To achieve this swap the top left
	   quadrant with bottom right quadrant, and swap the top right quadrant with bottom left quadrant
	*/

	//crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// find centers
	int center_y = magI.rows / 2;
	int center_x = magI.cols / 2;

	// find quadrants
	Mat top_left = magI(Rect(0, 0, center_x, center_y));
	Mat bottom_right = magI(Rect(center_x, center_y, center_x, center_y));
	Mat bottom_left = magI(Rect(0, center_y, center_x, center_y));
	Mat top_right = magI(Rect(center_x, 0, center_x, center_y));

	Mat above_mat;
	Mat below_mat;
	Mat final_mat;

	// horizontally concat bottom right and bottom left
	hconcat(bottom_right, bottom_left, above_mat);
	// horizontally concat top right and top left
	hconcat(top_right, top_left, below_mat);
	// vertically concat horizontally generated regions
	vconcat(above_mat, below_mat, final_mat);

	magI = final_mat.clone();

	// Transform the matrix with float values into a viewable image form (float between values 0 and 1).
	normalize(magI, magI, 0, 1, NOMINMAX);
	return magI;
}

int main()
{
	//Read images
	Mat image1 = imread("data/dog.bmp");
	if (!image1.data)                              // Check for invalid image
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	Mat image2 = imread("data/cat.bmp");
	if (!image2.data)                              // Check for invalid image
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	image1.convertTo(image1, CV_64FC3);
	image2.convertTo(image2, CV_64FC3);


	/*Several additional test cases are provided for you, but feel free to make
	your own(you'll need to align the images in a photo editor such as
	Photoshop).The hybrid images will differ depending on which image you
	assign as image1(which will provide the low frequencies) and which image
	you asign as image2(which will provide the high frequencies) */


	//========================================================================
	//							   PART 1 
	//========================================================================

	// IMPLEMENT THE FUNCTION myFilter(Mat,Mat,int) 
	// THIS FUNCTION TAKES THREE ARGUMENTS. FIRST ARGUMENT IS THE MAT IMAGE, 
	// SECOND ARGUMENT IS THE MAT FILTER AND THE THIRD ARGUMENT SPECIFIES THE
	// PADDING TYPE



	//========================================================================
	//							   PART2
	//========================================================================
	////  FILTERING AND HYBRID IMAGE CONSTRUCTION  ////

	int cutoff_frequency = 7; // for dog and cat
	// int cutoff_frequency = 5; // for bird and plane
	// int cutoff_frequency = 5; // for marilyn and einstein
	// int cutoff_frequency = 6; // for fish and submarine
	// int cutoff_frequency = 6; // for motorcycle and bicycle

	/*This is the standard deviation, in pixels, of the
	Gaussian blur that will remove the high frequencies from one image and
	remove the low frequencies from another image (by subtracting a blurred
	version from the original version). You will want to tune this for every
	image pair to get the best results.*/

	Mat filter = getGaussianKernel(cutoff_frequency * 4 + 1, cutoff_frequency, CV_64F);
	filter = filter*filter.t();

	// YOUR CODE BELOW. 
	// Use myFilter() to create low_frequencies of image 1. The easiest
	// way to create high frequencies of image 2 is to subtract a blurred
	// version of image2 from the original version of image2. Combine the
	// low frequencies and high frequencies to create 'hybrid_image'


	Mat low_freq_img = myFilter(image1, filter);
	Mat high_freq_img = image2 - myFilter(image2, filter);
	Mat hybrid_image = low_freq_img + high_freq_img;


	////  Visualize and save outputs  ////	
	//add a scalar to high frequency image because it is centered around zero and is mostly black	
	high_freq_img = high_freq_img + Scalar(0.5, 0.5, 0.5) * 255;
	//Convert the resulting images type to the 8 bit unsigned integer matrix with 3 channels
	high_freq_img.convertTo(high_freq_img, CV_8UC3);
	low_freq_img.convertTo(low_freq_img, CV_8UC3);
	hybrid_image.convertTo(hybrid_image, CV_8UC3);

	Mat vis = hybrid_image_visualize(hybrid_image);

	imshow("Low frequencies", low_freq_img); waitKey(0);
	imshow("High frequencies", high_freq_img);	waitKey(0);
	imshow("Hybrid image", vis); waitKey(0);


	imwrite("low_frequencies.jpg", low_freq_img);
	imwrite("high_frequencies.jpg", high_freq_img);
	imwrite("hybrid_image.jpg", hybrid_image);
	imwrite("hybrid_image_scales.jpg", vis);

	/*Several additional test cases are provided for you, but feel free to make
	your own(you'll need to align the images in a photo editor such as
	Photoshop).The hybrid images will differ depending on which image you
	assign as image1(which will provide the low frequencies) and which image
	you asign as image2(which will provide the high frequencies) */
	
	//============================================================================
	//							PART 3
	//============================================================================
	//In this part determine the DFT of just one channel of image1 and image2, as well 
	// as the DFT of the low frequency image and high frequency image.

	//Complete the code for DFT_Spectrum() method

	Mat img1_DFT = DFT_Spectrum(image1);
	imshow("Image 1 DFT", img1_DFT); waitKey(0);
	imwrite("Image1_DFT.jpg", img1_DFT * 255);

	low_freq_img.convertTo(low_freq_img, CV_64FC3);
	Mat low_freq_DFT = DFT_Spectrum(low_freq_img);
	imshow("Low Frequencies DFT", low_freq_DFT); waitKey(0);
	imwrite("Low_Freq_DFT.jpg", low_freq_DFT * 255);

	Mat img2_DFT = DFT_Spectrum(image2);
	imshow("Image 2 DFT", img2_DFT); waitKey(0);
	imwrite("Image2_DFT.jpg", img2_DFT * 255);

	high_freq_img.convertTo(high_freq_img, CV_64FC3);
	Mat high_freq_DFT = DFT_Spectrum(high_freq_img);
	imshow("High Frequencies DFT", high_freq_DFT); waitKey(0);
	imwrite("High_Freq_DFT.jpg", high_freq_DFT * 255);

}