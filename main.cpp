// Basic Bilateral Filter.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "BilateralFilter.h"
#include "DomainTransformBilateralFilter.h"
#include <conio.h>
#include <omp.h>

#include "fstream"
#include "iostream"

using namespace cv;
using namespace std;

void displayImage(char * winName,IplImage* image){
	cvNamedWindow(winName,0);
	cvShowImage(winName,image);
}

int wmain(int argc, _TCHAR* argv[])
{
	//IplImage *oimage = cvLoadImage(".\\images\\cat.png");
	IplImage *oimage = cvLoadImage(".\\images\\statue.jpg");
	IplImage *oimage2 = cvLoadImage(".\\images\\red.png");
	IplImage *oimage3 = cvLoadImage(".\\images\\green.png");
	IplImage *oimage4 = cvLoadImage(".\\images\\blue.png");
	
	Mat image(oimage);

#ifdef _DEBUG 
	printf("=============================\n");
	printf("=== Debug! ==================\n");
	printf("=============================\n");
	//DebugCTFileOut();
#endif

	DomainTransformBilateralFilter *NC_filter 
		= new DomainTransformBilateralFilter(image);
	DomainTransformBilateralFilter *NCMP_filter 
		= new DomainTransformBilateralFilter(image);
	DomainTransformBilateralFilter *IC_filter 
		= new DomainTransformBilateralFilter(image);
	DomainTransformBilateralFilter *ICMP_filter 
		= new DomainTransformBilateralFilter(image);
	DomainTransformBilateralFilter *RF_filter 
		= new DomainTransformBilateralFilter(image);
	DomainTransformBilateralFilter *RFMP_filter 
		= new DomainTransformBilateralFilter(image);
	
	//NC
	int64 pretime = getTickCount();
	for(int i = 0; i < 5 ; i++)
	{
		NC_filter->ApplyNC(60, 0.4, 3, false);
	}
	printf("Non NC : %lf\n", (getTickCount() - pretime) / 5 / getTickFrequency());
	
	//NC OpenMP
	pretime = getTickCount();
	for(int i = 0; i < 5 ; i++)
	{
		NCMP_filter->ApplyNC(60, 0.4, 3, true);
	}
	printf("MP  NC : %lf\n", (getTickCount() - pretime) / 5 / getTickFrequency());

	// IC
	pretime = getTickCount();
	for(int i = 0; i < 5 ; i++)
	{
		IC_filter->ApplyIC(60, 0.4, 3, false);
	}
	printf("Non IC : %f\n", (getTickCount() - pretime) / 5 / getTickFrequency());

	// IC OpenMP
	pretime = getTickCount();
	for(int i = 0; i < 5 ; i++)
	{
		ICMP_filter->ApplyIC(60, 0.4, 3, true);
	}
	printf("MP  IC : %f\n", (getTickCount() - pretime) / 5 / getTickFrequency());
	// RF
	pretime = getTickCount();
	for(int i = 0; i < 1 ; i++)
	{
		RF_filter->ApplyRF(60, 0.4, 3, false);
	}
	printf("Non RF : %f\n", (getTickCount() - pretime) / getTickFrequency());

	// RF OpenMP
	pretime = getTickCount();
	for(int i = 0; i < 1 ; i++)
	{
		RFMP_filter->ApplyRF(60, 0.4, 3, true);
	}
	printf("MP  RF : %f\n", (getTickCount() - pretime) / getTickFrequency());
	
	imshow("1. Normalized Convolution", NC_filter->output_image_);
	imshow("1. Normalized Convolution(MP)", NCMP_filter->output_image_);
	imshow("2. Normalized Convolution", IC_filter->output_image_);
	imshow("2. Normalized Convolution(MP)", ICMP_filter->output_image_);
	imshow("3. Recursive Filtering", RF_filter->output_image_);
	imshow("3. Recursive Filtering(MP)", RFMP_filter->output_image_);
	
	cvWaitKey(0);
	return 0;
}
