
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <stdio.h>

using namespace cv;
using namespace std;


int loadVideo();
uchar* invokeKernel(uchar *frame, int rows, int cols);


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


__global__ void testKernel(uchar *i_frame, uchar* o_frame, int rows, int cols)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	o_frame[i] = i_frame[i] - 127;
}


int main(int argc, char** argv)
{
	loadVideo();
	return 0;
}


int loadVideo() {
	String path = "C:/Users/sapan/Downloads/The Secret Life Of Walter Mitty.mp4";

	VideoCapture capVideo(path);
	if (!capVideo.isOpened()) {
		cout << "Cannot open the video" << endl;
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
		return -1;
	}

	double fps = capVideo.get(CV_CAP_PROP_FPS);
	double startTime = 70000;
	double currentTime = startTime;
	capVideo.set(CV_CAP_PROP_POS_MSEC, startTime);
	cout << "Frames per second: " << fps << endl;
	namedWindow("ThisVideo", CV_WINDOW_NORMAL);

	int i = 1;
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

		vector<uchar> frameArray(frame.rows*frame.cols);
		if (frame.isContinuous()) {
			frameArray.assign(frame.datastart, frame.dataend);
		}

		uchar *new_frame =  invokeKernel(&frameArray[0], frame.rows, frame.cols);
		Mat modified_frame(frame.rows, frame.cols, frame.type(), new_frame, frame.step);

		//mVideo << modified_frame;

		Mat canvas = Mat::zeros(frame.rows, frame.cols * 2 + 10, frame.type());

		frame.copyTo(canvas(Range::all(), Range(0, frame.cols)));
		modified_frame.copyTo(canvas(Range::all(), Range(frame.cols+10, frame.cols *2 + 10)));

		resize(canvas, canvas, Size(canvas.cols / 2, canvas.rows / 2));

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


uchar* invokeKernel(uchar *frame, int rows, int cols) {
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(1);
	}

	uchar *frame_out = new uchar[rows*cols]();

	// Allocate GPU buffers for two vectors (one input, one output)
	uchar *d_frame, *d_out;
	cudaStatus = cudaMalloc((void**)&d_frame, rows*cols * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}

	cudaStatus = cudaMalloc((void**)&d_out, rows*cols * sizeof(uchar));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_frame, frame, rows*cols * sizeof(uchar), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(1);
	}

	const int threadsPerBlock = 1024;
	const int blocksPerGrid = rows*cols / threadsPerBlock + 1;

	// Launch a kernel on the GPU with one thread for each element.
	testKernel << <blocksPerGrid, threadsPerBlock>> >(d_frame, d_out, rows, cols);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		exit(1);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		exit(1);
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(frame_out, d_out, rows*cols * sizeof(uchar), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		exit(1);
	}

	return frame_out;
}
