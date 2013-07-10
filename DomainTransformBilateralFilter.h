
#include "cv.h"
#include "cxcore.h"

using namespace cv;

class DomainTransformBilateralFilter
{
public:
	DomainTransformBilateralFilter(void);
	DomainTransformBilateralFilter(Mat input_image);
	~DomainTransformBilateralFilter(void);

	Mat original_image_;
	Mat output_image_;

	bool Init();
	bool Init(Mat input_image);
	bool ImageLoad(Mat input_image);
	bool ApplyNC(double sigma_s, double sigma_r, int iteration_number, bool open_mp_flag);
	bool ApplyIC(double sigma_s, double sigma_r, int iteration_number, bool open_mp_flag);
	bool ApplyRF(double sigma_s, double sigma_r, int iteration_number, bool open_mp_flag);
	
	Mat Transpose(Mat input_image);

private:
	double sigma_s_, sigma_r_;
	double *ctH_, *ctV_;
	double *dHdx_, *dVdy_; 
	int width_, height_;
	bool DebugCTFileOut();
	bool DebugdHdxFileOut();
	bool CalculateCTFunction();
	Mat IterationNCFunction(Mat input_image, double *ctH, double r);
	Mat IterationNCFunctionOpenMP(Mat input_image, double *ctH, double r);
	Mat IterationICFunction(Mat input_image, double *ctH, double r); 
	Mat IterationICFunctionOpenMP(Mat input_image, double *ctH, double r); 
	Mat IterationRFFunction(Mat input_image,double *dHdx, double sigma);
	Mat IterationRFFunctionOpenMP(Mat input_image,double *dHdx, double sigma);
	
	bool DebugRedAreaSizeFileOut();
	bool DebugCenterCumsumAreaSizeFileOut();
};
