#include "stdafx.h"
#include "BilateralFilter.h"
#include <math.h>

int max(int num1, int num2){
	if(num1>num2)
		return num1;
	else
		return num2;
}

int getValue(IplImage * image,int i,int j)
	{
		CvScalar pixel = cvGet2D(image,i,j);
		int pixelValue = pixel.val[0];
		return pixelValue;
	}


BilateralFilter::BilateralFilter(IplImage *image,double sigmaD, double sigmaR){

	    this->image = cvCloneImage(image);
        int sigmaMax = max(sigmaD, sigmaR);
        this->kernelRadius = ceil((double)2 * sigmaMax);

        
        double twoSigmaRSquared = 2 * sigmaR * sigmaR;

        int kernelSize = this->kernelRadius * 2 + 1;

       kernelD = new double*[kernelSize];
	   for(int i=0; i<kernelSize;i++){
			kernelD[i] = new double[kernelSize];
		}

       int center = (kernelSize - 1) / 2;

		for (int x = -center; x < -center + kernelSize; x++) {
			for (int y = -center; y < -center + kernelSize; y++) {
				kernelD[x + center][y + center] = this->gauss(sigmaD, x, y);
			}
		}


		gaussSimilarity = new double[256];
		for (int i = 0; i < 256; i++) {
			gaussSimilarity[i] = exp((double)-((i) / twoSigmaRSquared));
		}

		rimage = cvCloneImage(image);
}
IplImage * BilateralFilter::runFilter(){

            
	 for(int i=0;i<rimage->height;i++){
		for (int j=0;j<rimage->width;j++){
			
				apply(i,j);
			
			}
	 }
    
     return rimage;

}

double BilateralFilter::getSpatialWeight(int m, int n,int i,int j){
	return kernelD[(int)(i-m + kernelRadius)][(int)(j-n + kernelRadius)];
}


void BilateralFilter::apply(int i, int j) {// ~i=y j=x

	if(i>0 && j>0 && i<image->height && j< image->width){
		double sum = 0;
		double totalWeight = 0;
		int intensityCenter = getValue(image,i,j);


		int mMax = i + kernelRadius;
		int nMax = j + kernelRadius;
		double weight;

		for (int m = i-kernelRadius; m < mMax; m++) {
			for (int n = j-kernelRadius; n < nMax; n++) {

				if (this->isInsideBoundaries(m, n)) {
					int intensityKernelPos = getValue(image,m,n);
					weight = getSpatialWeight(m,n,i,j) * similarity(intensityKernelPos,intensityCenter);
					totalWeight += weight;
					sum += (weight * intensityKernelPos);
				}
			}
		}
		int newvalue=(int)floor(sum / totalWeight);
		
		CvScalar pixel;
		pixel.val[0]=newvalue;
		pixel.val[1]=newvalue;
		pixel.val[2]=newvalue;
		cvSet2D(rimage,i,j,pixel);
	}

}

double BilateralFilter::similarity(int p, int s) {
		// this equals: Math.exp(-(( Math.abs(p-s)) /  2 * this->sigmaR * this->sigmaR));
		// but is precomputed to improve performance
		return this->gaussSimilarity[abs(p-s)];
	}


double BilateralFilter::gauss (double sigma, int x, int y) {
		return exp(-((x * x + y * y) / (2 * sigma * sigma)));
	}

	
bool BilateralFilter::isInsideBoundaries(int m,int n){
        if(m>-1 && n>-1 && m<image->height && n <image->width)
            return true;
        else 
            return false;
}
