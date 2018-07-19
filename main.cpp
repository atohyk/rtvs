/*
Atohyk: Edited the feature tracking to spread out the points in the image and to remove
the pairs of points that are too long beyond a certain amount.

Code originally by Nghia Ho and edited by Chen Jia.

Originally taken from: http://nghiaho.com/uploads/videostabKalman.cpp
Thanks Nghia Ho for his excellent code.
And,I modified the smooth step using a simple kalman filter .
So,It can processes live video streaming.
modified by chen jia.
email:chenjia2013@foxmail.com
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;

// This video stablisation smooths the global trajectory using a sliding average window

//const int SMOOTHING_RADIUS = 15; // In frames. The larger the more stable the video, but less reactive to sudden panning
const int HORIZONTAL_BORDER_CROP = 200; // In pixels. Crops the border to reduce the black borders from stabilisation being too noticeable.

// 1. Get previous to current frame transformation (dx, dy, da) for all frames
// 2. Accumulate the transformations to get the image trajectory
// 3. Smooth out the trajectory using an averaging window
// 4. Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
// 5. Apply the new transformation to the video

struct TransformParam
{
    TransformParam() {}
    TransformParam(double _dx, double _dy, double _da) {
        dx = _dx;
        dy = _dy;
        da = _da;
    }

    double dx;
    double dy;
    double da; // angle
};

struct Trajectory
{
    Trajectory() {}
    Trajectory(double _x, double _y, double _a) {
        x = _x;
        y = _y;
        a = _a;
    }
	// "+"
	friend Trajectory operator+(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x+c2.x,c1.y+c2.y,c1.a+c2.a);
	}
	//"-"
	friend Trajectory operator-(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x-c2.x,c1.y-c2.y,c1.a-c2.a);
	}
	//"*"
	friend Trajectory operator*(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x*c2.x,c1.y*c2.y,c1.a*c2.a);
	}
	//"/"
	friend Trajectory operator/(const Trajectory &c1,const Trajectory  &c2){
		return Trajectory(c1.x/c2.x,c1.y/c2.y,c1.a/c2.a);
	}
	//"="
	Trajectory operator =(const Trajectory &rx){
		x = rx.x;
		y = rx.y;
		a = rx.a;
		return Trajectory(x,y,a);
	}

    double x;
    double y;
    double a; // angle
};
//
int main(int argc, char **argv)
{
	cout<<CV_VERSION<<endl;
	if(argc < 2) {
		cout << "./VideoStab [video.avi]" << endl;
		return 0;
	}
	// For further analysis
	//ofstream out_transform("prev_to_cur_transformation.txt");
	//ofstream out_trajectory("trajectory.txt");
	//ofstream out_smoothed_trajectory("smoothed_trajectory.txt");
	//ofstream out_new_transform("new_prev_to_cur_transformation.txt");

	VideoCapture cap(argv[1]);
	assert(cap.isOpened());

	Mat cur, cur_grey;
	Mat prev, prev_grey;

	cap >> prev;//get the first frame.ch
	cvtColor(prev, prev_grey, COLOR_BGR2GRAY);
	
	// Step 1 - Get previous to current frame transformation (dx, dy, da) for all frames
	vector <TransformParam> prev_to_cur_transform; // previous to current
	// Accumulated frame to frame transform
	double a = 0;
	double x = 0;
	double y = 0;
	// Step 2 - Accumulate the transformations to get the image trajectory
	vector <Trajectory> trajectory; // trajectory at all frames
	//
	// Step 3 - Smooth out the trajectory using an averaging window
	vector <Trajectory> smoothed_trajectory; // trajectory at all frames
	Trajectory X;//posteriori state estimate
	Trajectory	X_;//priori estimate
	Trajectory P;// posteriori estimate error covariance
	Trajectory P_;// priori estimate error covariance
	Trajectory K;//gain
	Trajectory	z;//actual measurement
	// double pstd = 4e-3;//can be changed
	// double cstd = 0.25;//can be changed
	double pstd = 2e-4;//can be changed, lower is more damped
	double cstd = 2.0;//can be changed, higher is more damped 
	Trajectory Q(pstd,pstd,pstd);// process noise covariance
	Trajectory R(cstd,cstd,cstd);// measurement noise covariance 
	// Step 4 - Generate new set of previous to current transform, such that the trajectory ends up being the same as the smoothed trajectory
	vector <TransformParam> new_prev_to_cur_transform;
	//
	// Step 5 - Apply the new transformation to the video
	//cap.set(CV_CAP_PROP_POS_FRAMES, 0);
	Mat canvas = Mat::zeros(prev.rows, prev.cols*2+10, prev.type());
	Mat T(2,3,CV_64F);
	int vert_border = HORIZONTAL_BORDER_CROP * prev.rows / prev.cols; // get the aspect ratio correct
	VideoWriter outputVideo; 

	//preparing output filename
	string outputfilename = string(argv[1]);
	size_t slashindex = outputfilename.rfind("/");
	if(slashindex == string::npos)
	{
		slashindex = 0;
	}
	outputfilename = outputfilename.substr(slashindex+1, outputfilename.size()-slashindex);
	size_t dotindex = outputfilename.rfind(".");
	if(dotindex == string::npos)
	{
		dotindex = outputfilename.size();
	}
	outputfilename = outputfilename.substr(0,dotindex);
	outputfilename = "./output/"+outputfilename + "_out.avi";
	//done preparing output filename

	outputVideo.open(outputfilename , CV_FOURCC('X','V','I','D'), 10 , canvas.size());  
	cout<<"size "<<prev.rows<<" "<<prev.cols*2<<endl;
	int k=1;
	int max_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Mat last_T;
	Mat prev_grey_,cur_grey_;
	Mat compare;
	
	int framenum = 0;
	 
	while(true) {

		cap >> cur;
		if(cur.data == NULL) {
			break;
		}

		cvtColor(cur, cur_grey, COLOR_BGR2GRAY);

		// vector from prev to cur
		vector <Point2f> prev_corner, cur_corner;
		vector <Point2f> prev_corner2, cur_corner2;
		vector <Point2f> prev_corner3, cur_corner3;
		vector <uchar> status;
		vector <float> err;
		//goodFeaturesToTrack(prev_grey, prev_corner, 200, 0.01, 30);
		//goodFeaturestoTrack(input, output, maxCorners, minQuality, minDist, mask, blocksize, harris, k)
		//increasing the distance causes the points to be evenly spread
		goodFeaturesToTrack(prev_grey, prev_corner, 500, 0.01, 40, noArray(), 10);
		calcOpticalFlowPyrLK(prev_grey, cur_grey, prev_corner, cur_corner, status, err);
		// weed out bad matches

		compare = 0.5*cur+0.5*prev;
		vector <double> dist;
		for(size_t i=0; i < status.size(); i++) {
			if(status[i]) {
				//further filtering of points
				Point2f diff = prev_corner[i]-cur_corner[i];
				dist.push_back(sqrt(diff.x*diff.x+diff.y*diff.y));
				prev_corner2.push_back(prev_corner[i]);
				cur_corner2.push_back(cur_corner[i]);
				//draw circles
				circle(compare,cur_corner[i],5,Scalar(0,255,0),2);
				circle(compare,prev_corner[i],5,Scalar(255,0,0),2);
				line(compare, cur_corner[i], prev_corner[i], Scalar(0,0,255),2);
			}
		}
		// for(size_t i =0;i<dist.size();i++)
		// {
		// 	cout<<"dist "<<i<<" "<<dist[i]<<endl;
		// }
		vector <double> largestdist;
		//find the largest n items in the vector
		float ratio=0.3;
		int numlargest = int(ratio*status.size());
		//init the vect with the first n items of the dist
		for(size_t i = 0;i<numlargest;i++)
		{
			largestdist.push_back(dist[i]);
		}
		//sort the vect
		sort(largestdist.begin(),largestdist.end());
		// for(size_t i =0;i<largestdist.size();i++)
		// {
		// 	cout<<"largedists "<<i<<" "<<largestdist[i]<<endl;
		// }

		for(int i = largestdist.size()-1;i<dist.size()-largestdist.size();i++)
		{//for each remaining distance
			for(int j = largestdist.size()-1;j>=0;j--)
			{//for each element in largestdist
				if(dist[i] > largestdist[j])
				{//insert distance into sorted array and remove lowest(first) element
					auto it =  largestdist.begin()+j+1;
					largestdist.insert(it, dist[i]);
					largestdist.erase(largestdist.begin());
					break;
				}
			}
		}
		// for(size_t i =0;i<largestdist.size();i++)
		// {
		// 	cout<<"largedistf "<<i<<" "<<largestdist[i]<<endl;
		// }
		double minlargest = largestdist[0];
		if(minlargest < 0.05)
		{//threshold the min largest, if 0 will remove all points.
			minlargest = 0.05;
		}
		//cout<<"minlargest "<<minlargest<<endl;
		for(size_t i=0;i<dist.size();i++)
		{
			if(dist[i]<minlargest)
			{
				prev_corner3.push_back(prev_corner2[i]);
				cur_corner3.push_back(cur_corner2[i]);
				line(compare, cur_corner2[i], prev_corner2[i], Scalar(0,255,255),4);
			}
		}
		//cout<<"curcorner3"<<cur_corner3.size()<<endl;
		if(framenum > 0)
		{
			resize(compare, compare, Size(compare.cols*0.7,compare.rows*0.7));
			//imshow("compare",compare);
			//waitKey(0);
		}

		// translation + rotation only
		//Mat T = estimateRigidTransform(prev_corner2, cur_corner2, false); // false = rigid transform, no scaling/shearing
		
		//TODO: make sure enough corners for good rigid transform
		Mat T = estimateRigidTransform(prev_corner3, cur_corner3, false);
		// in rare cases no transform is found. We'll just use the last known good transform.
		if(T.data == NULL) {
			last_T.copyTo(T);
		}

		T.copyTo(last_T);

		// decompose T
		double dx = T.at<double>(0,2);
		double dy = T.at<double>(1,2);
		double da = atan2(T.at<double>(1,0), T.at<double>(0,0));
		//
		//prev_to_cur_transform.push_back(TransformParam(dx, dy, da));

		//out_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		// Accumulated frame to frame transform
		x += dx;
		y += dy;
		a += da;
		//trajectory.push_back(Trajectory(x,y,a));
		//
		//out_trajectory << k << " " << x << " " << y << " " << a << endl;
		//
		z = Trajectory(x,y,a);
		//
		if(k==1){
			// intial guesses
			X = Trajectory(0,0,0); //Initial estimate,  set 0
			P =Trajectory(1,1,1); //set error variance,set 1
		}
		else
		{
			//time update£¨prediction£©
			X_ = X; //X_(k) = X(k-1);
			P_ = P+Q; //P_(k) = P(k-1)+Q;
			// measurement update£¨correction£©
			K = P_/( P_+R ); //gain;K(k) = P_(k)/( P_(k)+R );
			X = X_+K*(z-X_); //z-X_ is residual,X(k) = X_(k)+K(k)*(z(k)-X_(k)); 
			P = (Trajectory(1,1,1)-K)*P_; //P(k) = (1-K(k))*P_(k);
		}
		//smoothed_trajectory.push_back(X);
		//out_smoothed_trajectory << k << " " << X.x << " " << X.y << " " << X.a << endl;
		//-
		// target - current
		double diff_x = X.x - x;//
		double diff_y = X.y - y;
		double diff_a = X.a - a;

		dx = dx + diff_x;
		dy = dy + diff_y;
		da = da + diff_a;

		//new_prev_to_cur_transform.push_back(TransformParam(dx, dy, da));
		//
		//out_new_transform << k << " " << dx << " " << dy << " " << da << endl;
		//
		T.at<double>(0,0) = cos(da);
		T.at<double>(0,1) = -sin(da);
		T.at<double>(1,0) = sin(da);
		T.at<double>(1,1) = cos(da);

		T.at<double>(0,2) = dx;
		T.at<double>(1,2) = dy;

		Mat cur2;
		
		warpAffine(prev, cur2, T, cur.size());

		cur2 = cur2(Range(vert_border, cur2.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, cur2.cols-HORIZONTAL_BORDER_CROP));

		// Resize cur2 back to cur size, for better side by side comparison
		resize(cur2, cur2, cur.size());

		prev.copyTo(canvas(Range::all(), Range(0, cur2.cols)));
		cur2.copyTo(canvas(Range::all(), Range(cur2.cols+10, cur2.cols*2+10)));

		//cout<<"canvas"<<canvas.rows<<" "<<canvas.cols;
		outputVideo<<canvas;
		//outputVideo<<cur2;
		// If too big to fit on the screen, then scale it down by 2, hopefully it'll fit :)
		// if(canvas.cols > 1920) {
		// 	resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
		// }
		Mat img;
		resize(canvas, img, Size((compare.cols*2+10)*0.4,compare.rows*0.4));

		imshow("before and after", img);

		waitKey(1);
		//
		prev = cur.clone();//cur.copyTo(prev);
		cur_grey.copyTo(prev_grey);

		cout << "Frame: " << k << "/" << max_frames << " - good optical flow: " << prev_corner2.size() << endl;
		k++;
		framenum++;
	}
	cap.release();
	outputVideo.release();
	return 0;
}
