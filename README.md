# Udacity Self-Driving Car Engineer Nanodegree Program
## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/1-training-data.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/experiments.PNG
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/scale1.png
[image6]: ./output_images/scale2.png
[image7]: ./output_images/scale3.png
[image8]: ./output_images/scale4.png
[image9]: ./output_images/multi-scale.png
[image10]: ./output_images/heatmap.png
[image11]: ./output_images/heatmap_thresh.png
[image12]: ./output_images/labels.png
[image13]: ./output_images/objects.png
[image14]: ./output_images/test_images.png

[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cells from 1 to 5 of the IPython notebook 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `gray` color channel and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters during the lesson and in the project and my choice was based on the best classifier accuracy from the used parameters. Here is the output of the experiments 

![alt text][image3]


I decided to use YCrCb as I understood from searching that it has a good performance with the HOG 
Increasing the orientations has a very few effect on the accuracy
I noticed also increasing the pixels per cell could give better accuracy and using all the channels of will lead to better accuracy than one channel. I reached with the accuracy to more than 98% which I assumed it will be enough


So I settled on the following parameters 
`YCrCb`,`orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM as a first trial because I learned it performs well in this case 
I performed splitting on the training data to extract testing data for cross_validation using `train_test_split()` function from sklearn
The training has been done using the `fit()` function of sklearn.svm 

Then I applied testing for the output model to predict 20 samples of the test_set using `predict()` function 

Finally I calculated the accuracy using `score()` function

###### Note: This part is implemented in the code cell number 8 under the name of "Train the classifier"
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the bottom half of the image with a scale of 1.5 (default search window is 64x64) and 75% overlapping which was suitable for the image I was testing on this case. I used for that the function `find_cars()` which can extract the hog features from the image and run the sliding window technique then where ever the classifier returns a positive detection the position of the window in which the detection was made will be saved 

Here is an output of the implementation

![alt text][image4]

It performs well although there are one false positive and multiple detections on the two cars. and that what shall be fixed afterwards

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features which provided a nice result.  
I also applied an optimization for searching. As long as the car gets further from the car the search width decreases. Here are 4 images for the 4 scales I used for searching :
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

##### and here is the result on a test image:
![alt text][image9]

###### Note: The implementation of this part could be found from code cell 9 to 16

#### 3. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

##### To be able to handle the multiple detections and false positives I needed to apply the heat map technique then apply a threshold below it I reject the detection.
the implementation of this part can be found under the name `Heatmap`

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.

Here is the output of the previous test image 

![alt text][image10]

After threshold of 1 :

![alt text][image11]

Then I used the `scipy.ndimage.measurements.label()` function to collect the two detected objects:

![alt text][image12]

And here is the output on the test image:


![alt text][image13]


##### Then I implemented the whole pipeline `process_image()` which shall be modified afterwards to handle the video  
I run the pipeline on all test images and here is the output:

![alt text][image14]

looks good. Next step is to try it on the video

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

I tried first to run the pipeline on the test video, the result was not bad but It was noisy and unstable. and here's a [link to the first video trial](./test_video_out.mp4)

I then used the information of previous 15 video frames to add more confidence for the current detected objects. so I've done the heatmap on all the previous detections of the object not only on the current detection and a higher threshold also is applied to fit the higher heatmap
here's a [link to the second video trial](./test_video_out2.mp4)


Here's a [link to my final video result](./project_video_out.mp4)






---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I started by trying one feature (Hog) which returned to perform nicely with YCrCb and the test accuracy was acceptable, then I used the sliding window technique to capture the detected cars but the output contained false positives and multiple detections which was handled afterwards using the heatmap. I've done a search optimization which enabled to search in a smaller area and not get distracted by the trees and cars from the other road. Finally the video performance has been improved by adding the detection information from previous frames.

The pipeline shall work well in most cases but for sure it needs some improvements like :

* Dynamically calculate the search area for the sliding window to be more robust specially for sharp curves.
* Try different types of classifiers as I found that such case can be completely handled using Neural Networks like YOLO which needs massive training data but for sure will give much better results
* It would be interesting to use the information collected about objects and the camera information to calculate the position and speed of the car. For sure this will need fusion between different steps and for that we shall need to build uncertainty for the detections and deal with multiple motion models but this shall be very interesting
