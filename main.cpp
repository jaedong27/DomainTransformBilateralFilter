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
	NC_filter->Apply(NORMALIZED_CONVOLUTION, NONE, 60, 0.4, 3);
	printf("Non NC : %lf\n", (getTickCount() - pretime) / getTickFrequency());
	
	//NC OpenMP
	pretime = getTickCount();
	NCMP_filter->Apply(NORMALIZED_CONVOLUTION, OPEN_MP, 60, 0.4, 3);
	printf("MP  NC : %lf\n", (getTickCount() - pretime) / getTickFrequency());

	// IC
	pretime = getTickCount();
	IC_filter->Apply(INTERPOLATED_CONVOLUTION, NONE, 60, 0.4, 3);
	printf("Non IC : %f\n", (getTickCount() - pretime) / getTickFrequency());

	// IC OpenMP
	pretime = getTickCount();
	ICMP_filter->Apply(INTERPOLATED_CONVOLUTION, OPEN_MP, 60, 0.4, 3);
	printf("MP  IC : %f\n", (getTickCount() - pretime) / getTickFrequency());
	
	// RF
	pretime = getTickCount();
	RF_filter->Apply(RECURSIVE_FILTERING, NONE, 60, 0.4, 3);
	printf("Non RF : %f\n", (getTickCount() - pretime) / getTickFrequency());

	// RF OpenMP
	pretime = getTickCount();
	RFMP_filter->Apply(RECURSIVE_FILTERING, OPEN_MP, 60, 0.4, 3);
	printf("MP  RF : %f\n", (getTickCount() - pretime) / getTickFrequency());
	
	imshow("1. Normalized Convolution", NC_filter->image_);
	imshow("1. Normalized Convolution(MP)", NCMP_filter->image_);
	imshow("2. Normalized Convolution", IC_filter->image_);
	imshow("2. Normalized Convolution(MP)", ICMP_filter->image_);
	imshow("3. Recursive Filtering", RF_filter->image_);
	imshow("3. Recursive Filtering(MP)", RFMP_filter->image_);
	
	cvWaitKey(0);
	return 0;
}
