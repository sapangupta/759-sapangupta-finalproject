// hpc759_final_proj.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <string>

using namespace cv;
using namespace std;


int loadImage();
int loadVideo();

int main(int argc, char** argv)
{
	//loadImage();
	loadVideo();
	return 0;
}


int loadImage() {
	// Read the image file
	Mat image = imread("C:/Users/sapan/Pictures/sapan.jpg");

	if (image.empty()) // Check for failure
	{
		cout << "Could not open or find the image" << endl;
		system("pause"); //wait for any key press
		return -1;
	}

	String windowName = "My HelloWorld Window"; //Name of the window

	namedWindow(windowName); // Create a window

	imshow(windowName, image); // Show our image inside the created window.

	waitKey(0); // Wait for any keystroke in the window

	destroyWindow(windowName); //destroy the created window
	return 0;
}


int loadVideo() {
	String path = "C:/Users/sapan/Downloads/movies/492503111.mp4";

	VideoCapture capVideo(path);
	if (!capVideo.isOpened()) {
		cout << "Cannot open the video" << endl;
		return -1;
	}

	Size S = Size((int)capVideo.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
		(int)capVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
	int ex = static_cast<int>(capVideo.get(CV_CAP_PROP_FOURCC));

	VideoWriter bVideo, gVideo, rVideo, mVideo;
	bVideo.open("blue.avi", -1, capVideo.get(CV_CAP_PROP_FPS), S, true);
	gVideo.open("green.avi", -1, capVideo.get(CV_CAP_PROP_FPS), S, true);
	rVideo.open("red.avi", -1, capVideo.get(CV_CAP_PROP_FPS), S, true);
	mVideo.open("merge.avi", -1, capVideo.get(CV_CAP_PROP_FPS), S, true);

	if (!bVideo.isOpened() || !gVideo.isOpened() || !rVideo.isOpened() || !mVideo.isOpened())
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
		Mat frame;
		capVideo >> frame;
		if (frame.empty()) {
			break;
		}

		cout << "Frame " << i << ": " << "flags: " << frame.flags << ", dims: " << frame.dims << ", rows-cols: " 
			<< frame.rows << "-" << frame.cols << ", isContinuous: " << frame.isContinuous() << ", type: " << frame.type()
			<< ", channels: " << frame.channels() << ", total: " << frame.total() << ", elemSize: " << frame.elemSize()
			<< endl;

		vector<Mat> bgr(3);
		split(frame, bgr);

		cout << endl;
		for (int ix = 0; ix < 3; ix++) {
			cout << i << ": " << "flags: " << bgr[ix].flags << ", dims: " << bgr[ix].dims << ", rows-cols: " <<
				bgr[ix].rows << "-" << bgr[ix].cols << ", isContinuous: " << bgr[ix].isContinuous() << ", type: " << bgr[ix].type()
				<< ", channels: " << bgr[ix].channels() << ", total: " << bgr[ix].total() << ", elemSize: " << bgr[ix].elemSize()
				<< endl;
			switch (ix) {
			case 0:
				cout << "Blue:" << endl;
				bVideo << bgr[ix];
				break;
			case 1:
				cout << "Green:" << endl;
				gVideo << bgr[ix];
				break;
			case 2:
				cout << "Red:" << endl;
				rVideo << bgr[ix];
				break;
			}
			/*vector<uchar> frameArray(bgr[ix].rows*bgr[ix].cols);
			if (frame.isContinuous()) {
				frameArray.assign(bgr[ix].datastart, bgr[ix].dataend);
			}

			cout << "frameArray: " << endl;
			for (vector<uchar>::const_iterator iter = frameArray.begin(); iter != frameArray.end(); iter++) {
				cout << (int)*iter << ",";
			}
			cout << endl;*/
		}
		i++;

		Mat merged;
		merge(bgr, merged);
		mVideo << merged;

		imshow("ThisVideo", frame);

		int keyPressed = waitKey(30);
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
	return 0;
}