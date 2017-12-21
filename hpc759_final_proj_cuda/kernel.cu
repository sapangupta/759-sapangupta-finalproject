
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <stdio.h>
#include <math.h>

using namespace cv;
using namespace std;

// device constant memory arrays for convolution kernels
__constant__ int c_sobel_x[3][3];
__constant__ int c_sobel_y[3][3];
__constant__ int c_gaussian[5][5];


// convolution kernels
int sobel_x[3][3] = {
	{ -1, 0, 1 },
	{ -2, 0, 2 },
	{ -1, 0, 1 }
};
int sobel_y[3][3] = {
	{ -1, -2, -1 },
	{ 0, 0, 0 },
	{ 1, 2, 1 }
};
int gaussian[5][5] = {
	{ 2, 4, 5, 4, 2 },
	{ 4, 9, 12, 9, 4 },
	{ 5,  12,  15, 12, 5 },
	{ 4, 9, 12, 9, 4 },
	{ 2, 4, 5, 4, 2 },
};


int loadVideo();
unsigned char* invokeKernel(unsigned char *frame, int rows, int cols);


__global__ void applyFiltersGaussian(uchar* d_input, const size_t width, const size_t height, const int kernel_width, uchar* d_output)
{
	extern __shared__ uchar s_input2[];

	//2D Index of current thread
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((x<width) && (y<height))
	{
		const int crtShareIndex = threadIdx.y * blockDim.x + threadIdx.x;
		const int crtGlobalIndex = y * width + x;
		s_input2[crtShareIndex] = d_input[crtGlobalIndex];
		__syncthreads();

		const int r = (kernel_width - 1) / 2;
		int sum = 0;

		for (int i = -r; i <= r; ++i)
		{
			int crtY = threadIdx.y + i; //clamp the neighbor pixel, prevent overflow
			if (crtY < 0)						crtY = 0;
			else if (crtY >= blockDim.y)   		crtY = blockDim.y - 1;

			for (int j = -r; j <= r; ++j)
			{
				int crtX = threadIdx.x + j;
				if (crtX < 0) 					crtX = 0;
				else if (crtX >= blockDim.x)	crtX = blockDim.x - 1;

				const float inputPix = (float)(s_input2[crtY * blockDim.x + crtX]);	
				sum += inputPix * c_gaussian[r + j][r + i] / 159;
			}
		}
		d_output[y * width + x] = (uchar)sum;
	}
}


__global__ void applyFiltersSobel(uchar* d_input, const size_t cols, const size_t rows, const int kernel_width,
	int* d_output_x, int* d_output_y, uchar* d_output_grad)
{
	extern __shared__ uchar s_input2[];

	//2D Index of current thread
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((x<cols) && (y<rows))
	{
		const int crtShareIndex = threadIdx.y * blockDim.x + threadIdx.x;
		const int crtGlobalIndex = y * cols + x;
		s_input2[crtShareIndex] = d_input[crtGlobalIndex];
		__syncthreads();

		const int r = (kernel_width - 1) / 2;
		int sum_x = 0;
		int sum_y = 0;
		int sum = 0;

		for (int i = -r; i <= r; ++i)
		{
			int crtY = threadIdx.y + i; //clamp the neighbor pixel, prevent overflow
			if (crtY < 0)						crtY = 0;
			else if (crtY >= blockDim.y)   		crtY = blockDim.y - 1;

			for (int j = -r; j <= r; ++j)
			{
				int crtX = threadIdx.x + j;
				if (crtX < 0) 					crtX = 0;
				else if (crtX >= blockDim.x)	crtX = blockDim.x - 1;
				const float inputPix = (float)(s_input2[crtY * blockDim.x + crtX]);
				sum_x += inputPix * c_sobel_x[r + j][r + i];
				sum_y += inputPix * c_sobel_y[r + j][r + i];
			}
		}
		d_output_x[y * cols + x] = sum_x;
		d_output_y[y * cols + x] = sum_y;
		d_output_grad[y * cols + x] = sqrt(pow((double)sum_x, (double)2) + pow((double)sum_y, (double)2));
	}
}


__global__ void cuSuppressNonMax(uchar *mag, int *deltaX, int *deltaY, uchar *nms, int rows, int cols)
{
	const int SUPPRESSED = 0;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= 0 && idx < rows * cols)
	{
		float alpha;
		float mag1, mag2;
		// put zero all boundaries of image
		// TOP edge line of the image
		if ((idx >= 0) && (idx <cols))
			nms[idx] = 0;

		// BOTTOM edge line of image
		else if ((idx >= (rows - 1)*cols) && (idx < (cols * rows)))
			nms[idx] = 0;

		// LEFT & RIGHT edge line
		else if (((idx % cols) == 0) || ((idx % cols) == (cols - 1)))
		{
			nms[idx] = 0;
		}

		else // not the boundaries
		{
			// if magnitude = 0, no edge
			if (mag[idx] == 0)
				nms[idx] = (uchar)SUPPRESSED;
			else {
				if (deltaX[idx] >= 0)
				{
					if (deltaY[idx] >= 0)  // dx >= 0, dy >= 0
					{
						if ((deltaX[idx] - deltaY[idx]) >= 0)       // direction 1 (SEE, South-East-East)
						{
							alpha = (float)deltaY[idx] / deltaX[idx];
							mag1 = (1 - alpha)*mag[idx + 1] + alpha*mag[idx + cols + 1];
							mag2 = (1 - alpha)*mag[idx - 1] + alpha*mag[idx - cols - 1];
						}
						else                                // direction 2 (SSE)
						{
							alpha = (float)deltaX[idx] / deltaY[idx];
							mag1 = (1 - alpha)*mag[idx + cols] + alpha*mag[idx + cols + 1];
							mag2 = (1 - alpha)*mag[idx - cols] + alpha*mag[idx - cols - 1];
						}
					}
					else  // dx >= 0, dy < 0
					{
						if ((deltaX[idx] + deltaY[idx]) >= 0)    // direction 8 (NEE)
						{
							alpha = (float)-deltaY[idx] / deltaX[idx];
							mag1 = (1 - alpha)*mag[idx + 1] + alpha*mag[idx - cols + 1];
							mag2 = (1 - alpha)*mag[idx - 1] + alpha*mag[idx + cols - 1];
						}
						else                                // direction 7 (NNE)
						{
							alpha = (float)deltaX[idx] / -deltaY[idx];
							mag1 = (1 - alpha)*mag[idx + cols] + alpha*mag[idx + cols - 1];
							mag2 = (1 - alpha)*mag[idx - cols] + alpha*mag[idx - cols + 1];
						}
					}
				}

				else
				{
					if (deltaY[idx] >= 0) // dx < 0, dy >= 0
					{
						if ((deltaX[idx] + deltaY[idx]) >= 0)    // direction 3 (SSW)
						{
							alpha = (float)-deltaX[idx] / deltaY[idx];
							mag1 = (1 - alpha)*mag[idx + cols] + alpha*mag[idx + cols - 1];
							mag2 = (1 - alpha)*mag[idx - cols] + alpha*mag[idx - cols + 1];
						}
						else                                // direction 4 (SWW)
						{
							alpha = (float)deltaY[idx] / -deltaX[idx];
							mag1 = (1 - alpha)*mag[idx - 1] + alpha*mag[idx + cols - 1];
							mag2 = (1 - alpha)*mag[idx + 1] + alpha*mag[idx - cols + 1];
						}
					}

					else // dx < 0, dy < 0
					{
						if ((-deltaX[idx] + deltaY[idx]) >= 0)   // direction 5 (NWW)
						{
							alpha = (float)deltaY[idx] / deltaX[idx];
							mag1 = (1 - alpha)*mag[idx - 1] + alpha*mag[idx - cols - 1];
							mag2 = (1 - alpha)*mag[idx + 1] + alpha*mag[idx + cols + 1];
						}
						else                                // direction 6 (NNW)
						{
							alpha = (float)deltaX[idx] / deltaY[idx];
							mag1 = (1 - alpha)*mag[idx - cols] + alpha*mag[idx - cols - 1];
							mag2 = (1 - alpha)*mag[idx + cols] + alpha*mag[idx + cols + 1];
						}
					}
				}

				// non-maximal suppression
				// compare mag1, mag2 and mag[t]
				// if mag[t] is smaller than one of the neighbours then suppress it

				if ((mag[idx] < mag1) || (mag[idx] < mag2))
					nms[idx] = (uchar)SUPPRESSED;
				else
				{
					nms[idx] = (uchar)mag[idx];
				}
			}
		}
	}
}


__global__ void cuHysteresisHigh(uchar *o_frame, uchar *i_frame, int *strong_edge_mask, int t_high, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < (rows * cols)) {
		/* apply high threshold */
		if (i_frame[idx] > t_high) {
			strong_edge_mask[idx] = 1;
			o_frame[idx] = 255;
		}
		else {
			strong_edge_mask[idx] = 0;
			o_frame[idx] = 0;
		}
	}
}


__device__ void traceNeighbors(uchar *out_pixels, uchar *in_pixels, int idx, int t_low, int cols)
{
	unsigned n, s, e, w;
	unsigned nw, ne, sw, se;

	/* get indices */
	n = idx - cols;
	nw = n - 1;
	ne = n + 1;
	s = idx + cols;
	sw = s - 1;
	se = s + 1;
	w = idx - 1;
	e = idx + 1;

	if (in_pixels[nw] >= t_low) {
		out_pixels[nw] = 255;
	}
	if (in_pixels[n] >= t_low) {
		out_pixels[n] = 255;
	}
	if (in_pixels[ne] >= t_low) {
		out_pixels[ne] = 255;
	}
	if (in_pixels[w] >= t_low) {
		out_pixels[w] = 255;
	}
	if (in_pixels[e] >= t_low) {
		out_pixels[e] = 255;
	}
	if (in_pixels[sw] >= t_low) {
		out_pixels[sw] = 255;
	}
	if (in_pixels[s] >= t_low) {
		out_pixels[s] = 255;
	}
	if (in_pixels[se] >= t_low) {
		out_pixels[se] = 255;
	}
}


__global__ void cuHysteresisLow(uchar *out_pixels, uchar *in_pixels, int *strong_edge_mask, int t_low, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if ((idx > cols)                               
		&& (idx < (rows * cols) - cols) 
		&& ((idx % cols) < (cols - 1))        
		&& ((idx % cols) > (0)))                  
	{
		if (1 == strong_edge_mask[idx]) { 
			traceNeighbors(out_pixels, in_pixels, idx, t_low, cols);
		}
	}
}


void setConvolutionKernelsInDeviceConstantMem() {
	cudaMemcpyToSymbol(c_sobel_x, sobel_x, sizeof(int) * 9);
	cudaMemcpyToSymbol(c_sobel_y, sobel_y, sizeof(int) * 9);
	cudaMemcpyToSymbol(c_gaussian, gaussian, sizeof(int) * 25);
}


int main(int argc, char** argv)
{	
	setConvolutionKernelsInDeviceConstantMem();
	loadVideo();
	return 0;
}


int loadVideo() {
	string path = "C:/Users/sapan/Downloads/The Secret Life Of Walter Mitty.mp4";

	VideoCapture capVideo(path);
	if (!capVideo.isOpened()) {
		cout << "Cannot open the video" << endl;
		waitKey(0);
		return -1;
	}

	Size S = Size((int)capVideo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)capVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
	int ex = static_cast<int>(capVideo.get(CV_CAP_PROP_FOURCC));

	VideoWriter bVideo, gVideo, rVideo, mVideo;
	gVideo.open("grey.avi", -1, capVideo.get(CV_CAP_PROP_FPS), S, true);
	mVideo.open("modified.avi", -1, capVideo.get(CV_CAP_PROP_FPS), S, true);

	if (!gVideo.isOpened() || !mVideo.isOpened())
	{
		cout << "Could not open the output video for write." << endl;
		waitKey(0);
		return -1;
	}

	double fps = capVideo.get(CV_CAP_PROP_FPS);
	double startTime = 70000;
	double currentTime = startTime;
	capVideo.set(CV_CAP_PROP_POS_MSEC, startTime);
	cout << "Frames per second: " << fps << endl;
	namedWindow("ThisVideo", CV_WINDOW_NORMAL);

	while (1) {
		Mat colorframe;
		capVideo >> colorframe;
		if (colorframe.empty()) {
			break;
		}

		Mat frame;
		cvtColor(colorframe, frame, COLOR_BGR2GRAY);

		/*cout << "Frame " << i << ": " << "flags: " << frame.flags << ", dims: " << frame.dims << ", rows-cols: "
			<< frame.rows << "-" << frame.cols << ", isContinuous: " << frame.isContinuous() << ", type: " << frame.type()
			<< ", channels: " << frame.channels() << ", total: " << frame.total() << ", elemSize: " << frame.elemSize() << ", step: "
			<< frame.step << endl;*/

		//cout << endl;
	
		//gVideo << frame;

		vector<unsigned char> frameArray(frame.rows*frame.cols);
		if (frame.isContinuous()) {
			frameArray.assign(frame.datastart, frame.dataend);
		}


		unsigned char *new_frame =  invokeKernel(&frameArray[0], frame.rows, frame.cols);
		Mat modified_frame(frame.rows, frame.cols, frame.type(), new_frame, frame.step);


		//mVideo << modified_frame;

		/*Mat canvas = Mat::zeros(frame.rows, frame.cols * 2 + 10, frame.type());*/
		Mat grayBGR;
		cvtColor(modified_frame, grayBGR, COLOR_GRAY2BGR);
		Mat frames[2] = { colorframe, grayBGR };
		Mat canvas;
		hconcat(frames, 2, canvas);


		/*colorframe.copyTo(canvas(Range::all(), Range(0, frame.cols)));
		modified_frame.copyTo(canvas(Range::all(), Range(frame.cols+10, frame.cols *2 + 10)));*/

		//resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));

		imshow("ThisVideo", canvas);

		int keyPressed = waitKey(3);
		if (keyPressed == 27) {
			cout << "ESC pressed" << endl;
			break;
		}
		else if (keyPressed != -1) {
			currentTime = capVideo.get(CV_CAP_PROP_POS_MSEC) + startTime;
			capVideo.set(CV_CAP_PROP_POS_MSEC, currentTime);
			cout << "Pressed=" << keyPressed << " : Forwarding the video by 60s" << endl;
		}
	}

	waitKey(0); // Wait for any keystroke in the window
	destroyWindow("ThisVideo"); //destroy the created window
	cout << "ENDING THIS PROCESS !!";
	return 0;
}


unsigned char* invokeKernel(unsigned char *frame, int rows, int cols) {
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		waitKey(0);
		exit(1);
	}

	unsigned char *frame_out = new unsigned char[rows*cols];
	/*int sobel_x[5][5] = { 
		{2, 1, 0, -1, 2}, 
		{3, 2, 0, -2, -3},
		{4, 3, 0, -3, -4},
		{3, 2, 0, -2, -3},
		{2, 1, 0, -1, -2} 
	};
	int sobel_y[5][5] = { 
		{ 2 ,3 ,4, 3, 2 },
		{ 1, 2, 3, 2 ,1 },
		{ 0, 0, 0, 0, 0 },
		{ -1, -2, -3, -2, -1 },
		{ -2, -3, -4, -3, -2}
	};*/
		

	// Allocate GPU buffers for two vectors (one input, one output)
	unsigned char *d_frame, *d_out_gaussian, *d_out_suppress, *d_out_sobel_grad, *d_out_hys_high, *d_out_hys_low;
	int *d_out_sobel_x, *d_out_sobel_y, *d_strong_edge_mask;

	cudaStatus = cudaMalloc((void**)&d_frame, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&d_out_gaussian, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&d_out_sobel_grad, rows*cols * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&d_out_sobel_x, rows*cols * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&d_out_sobel_y, rows*cols * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&d_out_suppress, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&d_out_hys_high, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&d_out_hys_low, rows*cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&d_strong_edge_mask, rows*cols * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		waitKey(0);
		exit(1);
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_frame, frame, rows*cols * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		waitKey(0);
		exit(1);
	}

	// cuda kernel execution configurations
	const int threadsPerBlock = 1024;
	const int blocksPerGrid = rows*cols * 1023 / 1024;
	const dim3 blockDimHist(16, 16, 1);
	const dim3 gridDimHist(ceil((float)cols / blockDimHist.x), ceil((float)rows / blockDimHist.y), 1);

	// Launch a kernel on the GPU for sobel filtering
	applyFiltersGaussian << <gridDimHist, blockDimHist, blockDimHist.x * blockDimHist.y * sizeof(uchar) >> >(d_frame, cols, rows, 5, d_out_gaussian);
	applyFiltersSobel << <gridDimHist, blockDimHist, blockDimHist.x * blockDimHist.y * sizeof(uchar) >> >(d_out_gaussian, cols, rows, 3, 
																										  d_out_sobel_x, d_out_sobel_y, d_out_sobel_grad);
	cuSuppressNonMax << <blocksPerGrid, threadsPerBlock>> > (d_out_sobel_grad, d_out_sobel_x, d_out_sobel_y,
   														   d_out_suppress, rows, cols);
	cuHysteresisHigh << <blocksPerGrid, threadsPerBlock >> > (d_out_hys_high, d_out_suppress, d_strong_edge_mask, 50, rows, cols);
	cuHysteresisLow << <blocksPerGrid, threadsPerBlock >> > (d_out_hys_low, d_out_hys_high, d_strong_edge_mask, 1, rows, cols);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		waitKey(0);
		exit(1);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(frame_out, d_out_hys_low, rows*cols * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		waitKey(0);
		exit(1);
	}

	cudaFree(d_frame);			cudaFree(d_out_gaussian);
	cudaFree(d_out_sobel_x);	cudaFree(d_out_sobel_y);	cudaFree(d_out_sobel_grad);
	cudaFree(d_out_suppress);
	cudaFree(d_out_hys_high);	cudaFree(d_strong_edge_mask);
	cudaFree(d_out_hys_low);

	return frame_out;
}
