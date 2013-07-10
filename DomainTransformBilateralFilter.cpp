#include "StdAfx.h"
#include "DomainTransformBilateralFilter.h"
#include "fstream"
#include "iostream"

#include <omp.h>

#include <cmath>

using namespace cv;
using namespace std;

DomainTransformBilateralFilter::DomainTransformBilateralFilter(void)
{

}

DomainTransformBilateralFilter::~DomainTransformBilateralFilter(void)
{

}

DomainTransformBilateralFilter::DomainTransformBilateralFilter(Mat input_image)
{
	Init(input_image);
}

bool DomainTransformBilateralFilter::Init()
{
	sigma_s_ = 60;
	sigma_r_ = 0.4;
	return true;
}

bool DomainTransformBilateralFilter::Init(Mat input_image)
{
	original_image_= input_image.clone();
	width_ = original_image_.size().width;
	height_ = original_image_.size().height;
	Init();
	return true;
}

bool DomainTransformBilateralFilter::ImageLoad(Mat input_image)
{
	Init(input_image);
	return true;
}

bool DomainTransformBilateralFilter::ApplyNC(double sigma_s, double sigma_r, int iteration_number, bool open_mp_flag)
{
	double r, sigma_H_i;
	double temp1, temp2;
	int i;

#ifdef _DEBUG 
	printf("Calculate CT\n");
#endif

	CalculateCTFunction();

#ifdef _DEBUG 
	printf("Complete CT\n");
#endif
	Mat image = original_image_;
	//DebugCTFileOut();
	for ( i = 0 ; i < iteration_number ; i++)
	{
		#ifdef _DEBUG 
			printf("Start iteration %d\n",i);
		#endif

		r = 3 * sigma_s * pow((double)2,(double)(iteration_number - (i + 1))) 
			/ ( pow( pow((double)4,(double)(iteration_number)) - 1.0 , 0.5) );

		if(open_mp_flag == false)
		{
			image = IterationNCFunction(image, ctH_, r);
			image = Transpose(image);

			image = IterationNCFunction(image, ctV_, r);
			image = Transpose(image);
		}
		else
		{
			//Use OpenMP
			image = IterationNCFunctionOpenMP(image, ctH_, r);
			image = Transpose(image);

			image = IterationNCFunctionOpenMP(image, ctV_, r);
			image = Transpose(image);
		}
		
		#ifdef _DEBUG 
			printf("Complete iteration %d\n",i);
		#endif
	}

	output_image_ = image;
	return true;
}

bool DomainTransformBilateralFilter::ApplyIC(double sigma_s, double sigma_r, int iteration_number, bool open_mp_flag)
{
	double r, sigma_H_i;
	double temp1, temp2;
	int i;

#ifdef _DEBUG 
	printf("ApplyIC Calculate CT\n");
#endif

	CalculateCTFunction();

#ifdef _DEBUG 
	printf("Complete CT\n");
	printf("Debug!\n");
//	DebugCTFileOut();
#endif

	Mat image = original_image_;

	for ( i = 0 ; i < iteration_number ; i++)
	{
		#ifdef _DEBUG 
			printf("Start iteration %d\n",i);
		#endif

		r = 3 * sigma_s * pow((double)2,(double)(iteration_number - (i + 1))) 
			/ ( pow( pow((double)4,(double)(iteration_number)) - 1.0 , 0.5) );
		if(open_mp_flag == false)
		{
			image = IterationICFunction(image, ctH_, r);
			image = Transpose(image);

			image = IterationICFunction(image, ctV_, r);
			image = Transpose(image);
		}
		else
		{
			//Use OpenMP
			image = IterationICFunctionOpenMP(image, ctH_, r);
			image = Transpose(image);

			image = IterationICFunctionOpenMP(image, ctV_, r);
			image = Transpose(image);
		}
		

		#ifdef _DEBUG 
			printf("Complete iteration %d\n",i);
		#endif
	}

	output_image_ = image;
	return true;
}

bool DomainTransformBilateralFilter::ApplyRF(double sigma_s, double sigma_r, int iteration_number, bool open_mp_flag)
{
	double sigma, sigma_H_i;
	double temp1, temp2;
	int i;

#ifdef _DEBUG 
	printf("Recursive Filtering\n");
	printf("Calculate CT\n");
#endif

	CalculateCTFunction();

#ifdef _DEBUG 
	printf("Complete CT\n");
	printf("Debug!\n");
	//	DebugdHdxFileOut();
#endif

	Mat image = original_image_;

	for ( i = 0 ; i < iteration_number ; i++)
	{
#ifdef _DEBUG 
		printf("Start iteration %d\n",i);
#endif
		sigma = sqrt(3.0f) * sigma_s * pow((double)2,(double)(iteration_number - (i + 1))) 
			/ ( pow( pow((double)4,(double)(iteration_number)) - 1.0 , 0.5) );

		if( open_mp_flag == false)
		{
			image = IterationRFFunction(image, dHdx_, sigma);
			image = Transpose(image);
			image = IterationRFFunction(image, dVdy_, sigma);
			image = Transpose(image);
		}
		else
		{
			// Use OpenMP
			image = IterationRFFunctionOpenMP(image, dHdx_, sigma);
			image = Transpose(image);
			image = IterationRFFunctionOpenMP(image, dVdy_, sigma);
			image = Transpose(image);
		}

#ifdef _DEBUG 
		printf("Complete iteration %d\n",i);
#endif
	}

	output_image_ = image;
	return true;
}

Mat DomainTransformBilateralFilter::Transpose(Mat input_image)
{
	int intput_image_width = input_image.size().width;
	int intput_image_height = input_image.size().height;
	Mat output_image(intput_image_width, intput_image_height, input_image.type());
	int c;
	for( c = 0 ; c < 3 ; c++)
	{
		#pragma omp parallel for schedule(dynamic)
		for(int i = 0 ; i < intput_image_width ; i++)
		{
			for(int j = 0 ; j < intput_image_height ; j++)
			{
				output_image.at<Vec3b>(i,j)[c] = input_image.at<Vec3b>(j,i)[c];
			}
		}
	}
	return output_image;
}

// ====> +X
// =
// =
// =
// +Y

#define Diff(_x, _y)	 ((double) (abs ( (unsigned char)_x - (unsigned char)_y) ) / 255.0f)
// 두방향으로 모두 구하는 변수
// Transpose를 하기 위한 방법
bool DomainTransformBilateralFilter::CalculateCTFunction()
{
	ctH_  = new double[width_ * height_];
	ctV_  = new double[height_ * width_];

	dHdx_ = new double[width_ * height_];
	dVdy_ = new double[height_ * width_];

	int x, y;

	for( y = 0; y < height_ ; y++ )
	{
		//ctH_[PointToIndex(0, y)] = 1.0f;
		ctH_[width_ * y]  = 1.0f;
		dHdx_[width_ * y] = 1.0f;
	}

	for( x = 0; x < width_ ; x++ )
	{
		//ctV_[PointToIndex(x, 0)] = 1.0f;
		ctV_[height_ * x]  = 1.0f;
		dVdy_[height_ * x] = 1.0f;
	}

	double redIdx, blueIdx, greenIdx, Idx, ctdx, ctdy;

	// Get ctH()
	for( y = 0; y < height_ ; y++ )
	{
		for (x = 1; x < width_ ; x++ )
		{
			//Add diff Red
			redIdx = Diff( original_image_.at<Vec3b>(y,x)[2] , original_image_.at<Vec3b>(y,x-1)[2] );
			//Add diff Green
			greenIdx = Diff( original_image_.at<Vec3b>(y,x)[1] , original_image_.at<Vec3b>(y,x-1)[1] );
			//Add diff Blue
			blueIdx = Diff( original_image_.at<Vec3b>(y,x)[0] , original_image_.at<Vec3b>(y,x-1)[0] );

			Idx = redIdx + greenIdx + blueIdx;
			ctdx =  1 + sigma_s_/sigma_r_ * Idx;
			ctH_[width_*y + x]  = ctH_[width_*y + x - 1] + ctdx;
			dHdx_[width_*y + x] = ctdx;
		}
	}

	// Get ctV()
	for( x = 0; x < width_ ; x++ )
	{
		for (y = 1; y < height_ ; y++ )
		{
			//Add diff Red
			redIdx = Diff( original_image_.at<Vec3b>(y,x)[2] , original_image_.at<Vec3b>(y-1,x)[2]);
			//Add diff Green
			greenIdx = Diff( original_image_.at<Vec3b>(y,x)[1] , original_image_.at<Vec3b>(y-1,x)[1]);
			//Add diff Blue
			blueIdx = Diff( original_image_.at<Vec3b>(y,x)[0] , original_image_.at<Vec3b>(y-1,x)[0]);

			Idx = redIdx + greenIdx + blueIdx;
			ctdx =  1 + sigma_s_/sigma_r_ * Idx;
			ctV_[y + height_ * x] = (double)ctV_[y-1 + height_ * x] + (double)ctdx;
			dVdy_[y + height_ * x] = ctdx;
		}
	}
	return true;
}

bool DomainTransformBilateralFilter::DebugCTFileOut()
{
	int x, y;
	ofstream fout("CTH.csv",ios_base::out);

	for( y = 0; y < height_ ; y++ )
	{
		for (x = 0; x < width_ ; x++ )
		{
			fout<<ctH_[width_*y + x]<<",";
		}
		fout<<"\n";
	}
	fout.close();

	fout.open("CTV.csv",ios_base::out);

	for( y = 0; y < height_ ; y++ )
	{
		for (x = 0; x < width_ ; x++ )
		{
			fout<<ctV_[width_*y + x]<<",";
		}
		fout<<"\n";
	}
	fout.close();
	return true;
}

bool DomainTransformBilateralFilter::DebugdHdxFileOut()
{
	int x, y;
	ofstream fout("dHdx.csv",ios_base::out);

	for( y = 0; y < height_ ; y++ )
	{
		for (x = 0; x < width_ ; x++ )
		{
			fout<<dHdx_[width_*y + x]<<",";
		}
		fout<<"\n";
	}
	fout.close();

	fout.open("dVdy.csv",ios_base::out);

	for( y = 0; y < height_ ; y++ )
	{
		for (x = 0; x < width_ ; x++ )
		{
			fout<<dVdy_[width_*y + x]<<",";
		}
		fout<<"\n";
	}
	fout.close();
	return true;
}

Mat DomainTransformBilateralFilter::IterationNCFunction(Mat input_image, double *ctH, double r)
{
	int width = input_image.size().width;
	int height = input_image.size().height;
	
	Mat output_image(height, width, input_image.type());

	int c = input_image.channels();
	double Kp;
	double *accumulate_data;	
	double *ctH_column;
	int lower_index, upper_index;
	int x;

	for( c = 0 ; c < 3 ; c++)
	{
		// Calculate Horizon()
		int y;

		for( y = 0; y < height ; y++ )
		{
			//printf("%d %d\n", omp_get_thread_num(), y);

			accumulate_data = new double[width + 1];

			lower_index = 0;
			upper_index = 0;
			ctH_column = ctH + width * y;

			//Accumulate Data
			accumulate_data[0]   = 0;
			for (x = 0; x < width ; x++ )
			{
				accumulate_data[x+1]   = accumulate_data[x] + input_image.at<Vec3b>(y,x)[c];
			}

			for (x = 0; x < width ; x++ )
			{
				for( ; ( ctH_column[x] - ctH_column[lower_index] ) >= r                         ; lower_index++ ) {}
				for( ; (upper_index < width) && (ctH_column[upper_index] - ctH_column[x] ) <= r ; upper_index++ ) {}
				//lower_index는 시작 index;
				//upper_index는 upper_index-1이 마지막 인덱스

				Kp = upper_index - lower_index;
				output_image.at<Vec3b>(y,x)[c] = (accumulate_data[upper_index] -  accumulate_data[lower_index]) / Kp;
			}
		}
	}
	
	return output_image;
}


Mat DomainTransformBilateralFilter::IterationNCFunctionOpenMP(Mat input_image, double *ctH, double r)
{
	int width = input_image.size().width;
	int height = input_image.size().height;

	Mat output_image(height, width, input_image.type());

	int c = input_image.channels();
	
	for( c = 0 ; c < 3 ; c++)
	{
		// Calculate Horizon()
		
		#pragma omp parallel for schedule(dynamic)
		for(int y = 0; y < height ; y++ )
		{
			//printf("%d %d\n", omp_get_thread_num(), y);
			
			double *accumulate_data = new double[width + 1];

			int lower_index = 0;
			int upper_index = 0;
			double *ctH_column = ctH + width * y;

			//Accumulate Data
			accumulate_data[0]   = 0;
			int x;
			for ( x = 0; x < width ; x++ )
			{
				accumulate_data[x+1]   = accumulate_data[x] + input_image.at<Vec3b>(y,x)[c];
			}

			for (x = 0; x < width ; x++ )
			{
				for( ; ( ctH_column[x] - ctH_column[lower_index] ) >= r                         ; lower_index++ ) {}
				for( ; (upper_index < width) && (ctH_column[upper_index] - ctH_column[x] ) <= r ; upper_index++ ) {}
				//lower_index는 시작 index;
				//upper_index는 upper_index-1이 마지막 인덱스

				double Kp = upper_index - lower_index;
				//Kp = ;
				output_image.at<Vec3b>(y,x)[c] = (accumulate_data[upper_index] -  accumulate_data[lower_index]) / Kp;
			}
		}
	}

	return output_image;
}

Mat DomainTransformBilateralFilter::IterationICFunction(Mat input_image, double *ctH, double r)
{
	int width = input_image.size().width;
	int height = input_image.size().height;

	Mat output_image(height, width, input_image.type());
		
	double Kp, sum_color;
	int lower_index, upper_index;
	double *ctH_column, *ctV_row;
	double *accumulate_area;

	int x, y, c, i, j;
	double rect_width, rect_left_length, rect_right_height, alpha;
	double xa, xb, ya, yb, xm, ym;

	accumulate_area   = new double[width];

	for( c = 0 ; c < 3 ; c++ )
	{
		// Calculate Horizon()
		for( y = 0; y < height ; y++ )
		{
			lower_index = 0;
			upper_index = 0;
			ctH_column = ctH + width * y;

			//Accumulate Data
			accumulate_area[0] = 0;
			for (x = 0; x < width - 1 ; x++ )
			{
					accumulate_area[x + 1] = accumulate_area[x] + ( 0.5 * (input_image.at<Vec3b>(y,x + 1)[c] + input_image.at<Vec3b>(y,x)[c])
															  * (ctH_column[x + 1] - ctH_column[x]) );
			}

			for (x = 0; x < width ; x++ )
			{
				for( ; ( ctH_column[x] - ctH_column[lower_index] ) >= r                         ; lower_index++ ) {}
				for( ; (upper_index < width) && (ctH_column[upper_index] - ctH_column[x] ) <= r ; upper_index++ ) {}
			
				// Get Center Area
				sum_color = accumulate_area[upper_index - 1] - accumulate_area[lower_index];

				//Get left Rectangle
				if(lower_index == 0) 
				{ 
					xa = ctH_column[lower_index] - (1.2 * r); 
					ya = input_image.at<Vec3b>(y,lower_index)[c];
				}                           
				else
				{
					xa = ctH_column[lower_index - 1] ;
					ya = input_image.at<Vec3b>(y,lower_index-1)[c];
				}
				xb = ctH_column[lower_index];
				yb = input_image.at<Vec3b>(y,lower_index)[c];

				xm = ctH_column[x] - r;
				alpha = (yb - ya) / (xb - xa);
				ym = ya + alpha * (xm - xa);
				sum_color += 0.5 * (xb - xm) * (ym + yb);
				

				//Get right Rectangle
				xa = ctH_column[upper_index - 1];
				ya = input_image.at<Vec3b>(y,upper_index - 1)[c];
				if(upper_index == width) 
				{ 
					xb = ctH_column[upper_index - 1] + (1.2 * r); 
					yb = input_image.at<Vec3b>(y,upper_index - 1)[c];
				}                           
				else
				{
					xb = ctH_column[upper_index] ;
					yb = input_image.at<Vec3b>(y,upper_index)[c];
				}

				xm = ctH_column[x] + r;
				alpha = (yb - ya) / (xb - xa);
				ym = ya + alpha * (xm - xa);
				sum_color += 0.5 * (xm - xa) * (ym + ya);
				
				output_image.at<Vec3b>(y,x)[c] = (sum_color / (2*r));
			}	
		}
	}

	return output_image;
}

Mat DomainTransformBilateralFilter::IterationICFunctionOpenMP(Mat input_image, double *ctH, double r)
{
	int width = input_image.size().width;
	int height = input_image.size().height;

	Mat output_image(height, width, input_image.type());

	int c;
	
	for( c = 0 ; c < 3 ; c++ )
	{
		// Calculate Horizon()
		#pragma omp parallel  for schedule(dynamic)
		for(int y = 0; y < height ; y++ )
		{
			int lower_index = 0;
			int upper_index = 0;
			double *ctH_column = ctH + width * y;

			//Accumulate Data
			double *accumulate_area   = new double[width];
			accumulate_area[0] = 0;
			int x;
			for (x = 0; x < width - 1 ; x++ )
			{
				accumulate_area[x + 1] = accumulate_area[x] + ( 0.5 * (input_image.at<Vec3b>(y,x + 1)[c] + input_image.at<Vec3b>(y,x)[c])
					* (ctH_column[x + 1] - ctH_column[x]) );
			}

			for (x = 0; x < width ; x++ )
			{
				for( ; ( ctH_column[x] - ctH_column[lower_index] ) >= r                         ; lower_index++ ) {}
				for( ; (upper_index < width) && (ctH_column[upper_index] - ctH_column[x] ) <= r ; upper_index++ ) {}

				// Get Center Area
				double sum_color = accumulate_area[upper_index - 1] - accumulate_area[lower_index];

				double rect_width, rect_left_length, rect_right_height, alpha;
				double xa, xb, ya, yb, xm, ym;

				//Get left Rectangle
				if(lower_index == 0) 
				{ 
					xa = ctH_column[lower_index] - (1.2 * r); 
					ya = input_image.at<Vec3b>(y,lower_index)[c];
				}                           
				else
				{
					xa = ctH_column[lower_index - 1] ;
					ya = input_image.at<Vec3b>(y,lower_index-1)[c];
				}
				xb = ctH_column[lower_index];
				yb = input_image.at<Vec3b>(y,lower_index)[c];

				xm = ctH_column[x] - r;
				alpha = (yb - ya) / (xb - xa);
				ym = ya + alpha * (xm - xa);
				sum_color += 0.5 * (xb - xm) * (ym + yb);


				//Get right Rectangle
				xa = ctH_column[upper_index - 1];
				ya = input_image.at<Vec3b>(y,upper_index - 1)[c];
				if(upper_index == width) 
				{ 
					xb = ctH_column[upper_index - 1] + (1.2 * r); 
					yb = input_image.at<Vec3b>(y,upper_index - 1)[c];
				}                           
				else
				{
					xb = ctH_column[upper_index] ;
					yb = input_image.at<Vec3b>(y,upper_index)[c];
				}

				xm = ctH_column[x] + r;
				alpha = (yb - ya) / (xb - xa);
				ym = ya + alpha * (xm - xa);
				sum_color += 0.5 * (xm - xa) * (ym + ya);

				output_image.at<Vec3b>(y,x)[c] = (sum_color / (2*r));
			}	
		}
	}

	return output_image;
}

bool DomainTransformBilateralFilter::DebugRedAreaSizeFileOut()
{
	int x, y;
	ofstream fout("Area.csv",ios_base::out);

	double sum_color[3];
	double* ctH_column;
	for( y = 0; y < height_ ; y++ )
	{
		ctH_column = ctH_+ width_ * y;
		for (x = 0; x < (width_ - 1) ; x++ )
		{
			sum_color[0] = 0.5 * (original_image_.at<Vec3b>(y,x+1)[0] + original_image_.at<Vec3b>(y,x)[0])
								* (ctH_column[x+1] - ctH_column[x]);
			fout<<sum_color[0]<<",";
		}
		fout<<"\n";
	}
	fout.close();
	return true;
}

bool DomainTransformBilateralFilter::DebugCenterCumsumAreaSizeFileOut()
{
	int x, y;
	ofstream fout("C.csv",ios_base::out);

	for( y = 0; y < height_ ; y++ )
	{
	}
	fout.close();
	return true;
}

Mat DomainTransformBilateralFilter::IterationRFFunction(Mat input_image,double *dHdx, double sigma)
{
	double Kp, sum_color[3];
	int lower_index, upper_index;
	double *dHdx_column, *dVdy_row;

	int height = input_image.size().height;
	int width = input_image.size().width;
	//Mat output_image(height, width, input_image.type());

	int x, y, c;
	double diff_color;
	double a = exp(-sqrt(2.0f) / sigma);
	double temp, V = 1;
	double I1, I2;

	//double *result_picture = new double[width_ * height_];
	for ( c = 0 ; c < 3 ; c++)
	{
		for( y = 0; y < height ; y++ )
		{
			// Calculate Horizon() Left -> Right
			double *row_array = new double[width];
			row_array[0] = input_image.at<Vec3b>(y,0)[c];
			dHdx_column = dHdx + width * y;
			for (x = 1 ; x < width ; x++ )
			{
				I1 = row_array[x-1];// output_image_.at<Vec3b>(y,x+1)[c];
				I2 = input_image.at<Vec3b>(y,x)[c];
				diff_color = I1 - I2;
				V = pow((float)a, (float)dHdx_column[x]);
				row_array[x] = I2 + V * diff_color;
			}

			// Calculate Horizon() Right -> Left
			input_image.at<Vec3b>(y,width-1)[c] = row_array[width];
			for (x = width - 2 ; x >= 0 ; x-- )
			{
				I1 = row_array[x+1];
				I2 = row_array[x];
				diff_color = I1 - I2;
				V = pow((float)a, (float)dHdx_column[x + 1]);
				row_array[x] = I2 + V * diff_color;
				input_image.at<Vec3b>(y,x)[c] = row_array[x];
			}
			
		}
	}

	return input_image;
}

Mat DomainTransformBilateralFilter::IterationRFFunctionOpenMP(Mat input_image,double *dHdx, double sigma)
{
	int height = input_image.size().height;
	int width = input_image.size().width;
	//Mat output_image(height, width, input_image.type());

	int c;
	
	double a = exp(-sqrt(2.0f) / sigma);
	double V;
	
	//double *result_picture = new double[width_ * height_];
	for ( c = 0 ; c < 3 ; c++)
	{
		#pragma omp parallel  for schedule(dynamic)
		for(int y = 0; y < height ; y++ )
		{
			double diff_color, I1, I2;
			double *dHdx_column;
			// Calculate Horizon() Left -> Right
			double *row_array = new double[width];
			double temp;
			row_array[0] = input_image.at<Vec3b>(y,0)[c];
			dHdx_column = dHdx + width * y;
			for (int x = 1 ; x < width ; x++ )
			{
				I1 = row_array[x-1];// output_image_.at<Vec3b>(y,x+1)[c];
				I2 = input_image.at<Vec3b>(y,x)[c];
				diff_color = I1 - I2;
				V = pow((float)a, (float)dHdx_column[x]);
				row_array[x] = I2 + V * diff_color;
			}

			// Calculate Horizon() Right -> Left
			input_image.at<Vec3b>(y,width-1)[c] = row_array[width];
			for (int x = width - 2 ; x >= 0 ; x-- )
			{
				I1 = row_array[x+1];
				I2 = row_array[x];
				diff_color = I1 - I2;
				
				V = pow((float)a, (float)dHdx_column[x+1]);
				row_array[x] = I2 + V * diff_color;
				input_image.at<Vec3b>(y,x)[c] = row_array[x];
			}
		}
	}

	return input_image;
}
