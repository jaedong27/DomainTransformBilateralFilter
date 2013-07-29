
#include "cv.h"
#include "cxcore.h"

using namespace cv;

enum{NORMALIZED_CONVOLUTION, INTERPOLATED_CONVOLUTION, RECURSIVE_FILTERING};
enum{NONE, OPEN_MP};

class DomainTransformBilateralFilter
{
public:
	
	Mat image_;
	
	DomainTransformBilateralFilter(void);
	DomainTransformBilateralFilter(Mat input_image);
	~DomainTransformBilateralFilter(void);

	bool Init();
	bool Init(Mat input_image);
	bool ImageLoad(Mat input_image);
	bool ApplyNC(double sigma_s, double sigma_r, int iteration_number, bool open_mp_flag);
	bool ApplyIC(double sigma_s, double sigma_r, int iteration_number, bool open_mp_flag);
	bool ApplyRF(double sigma_s, double sigma_r, int iteration_number, bool open_mp_flag);
	bool Apply(int filter_type, bool open_mp_flag, double sigma_s, double sigma_r, int iteration_number);
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
