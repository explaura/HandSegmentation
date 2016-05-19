package com.thoughtworks.hand_segmentation;

import android.content.pm.ActivityInfo;
import android.os.Environment;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.List;

public class MainActivity extends AppCompatActivity implements View.OnTouchListener,
        CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "ColourBlobDetection";
    private boolean mIsColourSelected = false;
    private Mat mRgba;
    private Mat mMask;
    private Scalar mBlobColourRgba;
    private Scalar mBlobColourHsv;
    private ColourBlobDetector mDetector;
    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {

            switch (status) {
                // OpenCV initialized successfully
                case LoaderCallbackInterface.SUCCESS:
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate() called");
        //Placed above call to super otherwise onCreate() is called twice
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

//        FloatingActionButton floatingActionButton = (FloatingActionButton) findViewById(R.id.fab);
//        floatingActionButton.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                Snackbar.make(v, "Button selected", Snackbar.LENGTH_SHORT).show();
//            }
//        });

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.surfaceView);
        mOpenCvCameraView.setCvCameraViewListener(this);

        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
            Toast.makeText(this.getApplicationContext(), ":(", Toast.LENGTH_SHORT).show();
        } else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            Toast.makeText(this.getApplicationContext(), "OpenCV Initialized", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        //CvType - Bit depth/number of channels
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        //mMask = Mat.zeros(mRgba.size(), CvType.CV_8UC4);
        mMask = new Mat(mRgba.size(), CvType.CV_8U, Scalar.all(0));
        mDetector = new ColourBlobDetector();
        mBlobColourHsv = new Scalar(255);
        mBlobColourRgba = new Scalar(255);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mMask.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();

        //If user has selected colour by touch
        if (mIsColourSelected) {
            mDetector.process(mRgba);
            List<MatOfPoint> contours = mDetector.getContours();

            Mat clonedMat = mRgba.clone(); //Colour space changes to BGR
            Imgproc.cvtColor(clonedMat, clonedMat, Imgproc.COLOR_BGR2RGBA);
            Mat mask_image = new Mat(clonedMat.size(), CvType.CV_8UC4, Scalar.all(0));
            Imgproc.drawContours(mask_image, contours, -1, Scalar.all(255), -1);
            //Invert mask
            //Core.bitwise_not(mask_image, mask_image);

            mRgba.copyTo(mMask, mask_image);
            //saveMatToFile(mMask, "mMask.png");
        }

        return mRgba;
    }

    private boolean saveMatToFile(Mat matImage, String filename) {
        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        if (path.isDirectory()) {
            File file = new File(path, filename);
            boolean res = Imgcodecs.imwrite(file.toString(), matImage);
            return res;
        }
        return false;
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        int cols = mRgba.cols();
        int rows = mRgba.rows();

        //View/Touch offset
        int xOffset = (mOpenCvCameraView.getWidth() - cols) / 2;
        int yOffset = (mOpenCvCameraView.getHeight() - rows) / 2;

        int x = (int)event.getX() - xOffset;
        int y = (int)event.getY() - yOffset;

        if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

        Rect touchedRect = new Rect();

        touchedRect.x = (x>4) ? x-4 : 0;
        touchedRect.y = (y>4) ? y-4 : 0;

        touchedRect.width = (x+4 < cols) ? x + 4 - touchedRect.x : cols - touchedRect.x;
        touchedRect.height = (y+4 < rows) ? y + 4 - touchedRect.y : rows - touchedRect.y;

        Mat touchedRegionRgba = mRgba.submat(touchedRect);

        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL);

        // Calculate average color of touched region
        mBlobColourHsv = Core.sumElems(touchedRegionHsv);
        int pointCount = touchedRect.width*touchedRect.height;
        for (int i = 0; i < mBlobColourHsv.val.length; i++)
            mBlobColourHsv.val[i] /= pointCount;

        mBlobColourRgba = convertScalarHsv2Rgba(mBlobColourHsv);

        Log.i(TAG, "Touched rgba color: (" + mBlobColourRgba.val[0] + ", " + mBlobColourRgba.val[1] +
                ", " + mBlobColourRgba.val[2] + ", " + mBlobColourRgba.val[3] + ")");

        mDetector.setHsvColor(mBlobColourHsv);

        mIsColourSelected = true;

        touchedRegionRgba.release();
        touchedRegionHsv.release();

        return false; // don't need subsequent touch events
    }

    private Scalar convertScalarHsv2Rgba(Scalar hsvColour) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColour);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }
}
