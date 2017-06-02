# **Finding Lane Lines on the Road** 

---

In this project, I created a pipeline to find lane on a road. The focus of the project was to make use of dominant signals within the image to bias lane construction. Also the focus intended to identify straight stretches of road with non-occluded areas.

[//]: # (Image References)

[image1]: ./test_images_output/begin_solidYellowCurve2.jpg "begin"
[image2]: ./test_images_output/gaussian_solidYellowCurve2.jpg "grayscale"
[image3]: ./test_images_output/canny_solidYellowCurve2.jpg "canny"
[image4]: ./test_images_output/example_solidYellowCurve2.jpg "hough raw output"
[image5]: ./test_images_output/hough_solidYellowCurve2.jpg "hough filtered output"
[image6]: ./test_images_output/roi_solidYellowCurve2.jpg "roi"
[image7]: ./test_images_output/end_solidYellowCurve2.jpg "end"

---

### Reflection

### 1. The Pipeline

I implemented the pipeline by first defining python module `lane_detect`, and then made available an object wrapper `LaneFilter`. Creating a wrapper class allows me to quick instantiate and apply filter algorithms from the top level. Using [test-lane-detect](https://github.com/hhony/CarND-LaneLines-P1/blob/master/test-lane-detect), I was able to quickly test parameters and isolate portions of the pipeline.

#### Pipeline overview

```python
from lane_detect import LaneFilter

filter = LaneFilter(filename=str(_file))
filter.gaussian_blur()
filter.canny_edges(CANNY_LOWER_BOUND, CANNY_UPPER_BOUND)
filter.hough_lines(rho=HOUGH_RHO, threshold=HOUGH_THRESH,
                   min_line_len=HOUGH_LINE_LEN, max_line_gap=HOUGH_LINE_GAP,
                   with_lines=True)
filter.apply_roi_mask()
filter.weighted_image()
```

#### Steps in the pipeline:

1) Read image (from file or n-dimensional array)
    * Store original image untouched for later overlay
    * Create copies for transform, mask, and grayscale
    * I also created an output storage copy
![alt text][image1]
    
2) Apply gaussian to grayscale image
    * Introducing noise helps pronouce dominant features, and mutes textures
![alt text][image2]

3) Apply Canny edge detection to gaussian grayscale
    * This performs the most work to isolate features
![alt text][image3]

4) Apply Hough transform to Canny edges image
    * This performs the other magic part, by correlating Euclidian space to Hough space
    * The default behavior of the pipeline is to drawing the output
    * The `draw_lines` method could have contained a bunch of math, instead, these became the helper functions
    * I added `land_detect/line_math.py` to maintain all the line/slope/interpolation/filtering
![alt text][image4]
becomes..
![alt text][image5]

5) Apply the Region of Interest (ROI) mask
    * I maintained a pyramid shape, with a flat top as the default roi shape, inside `LaneFilter.__init__`
![alt text][image6]

6) Apply the weighted image using original, mask and transform combination
![alt text][image7]


#### Logic around annotations


**Note:** When running `./test-lane-detect`, you will see real `[WARNING]` and `[ERROR]` messages. The warnings are to let me know that I am excluding points and lines from the final filtered set.. while the errors are real errors. Either I did not compute a lane line or there was a computation error. However, there should not be any computation errors at this point, in the current naive implementation.

The `draw_lines` function could have become an giant mess of logic. Instead of cramming all that logic into one function, I made several helper functions in `line_math.py`. The Hough transform returns several line segments, some which overlap and others that are complete outliers for lane data.

My naive approach is to first generate the ROI from within `draw_lines`. I then convert the `ndarray` to an `array(dtype=bool)`. This allows me to directly use the points from the Hough transform in the first pass of the data, to disqualify any point pairs which do not have at least on point in the ROI.

Also in the first pass over these data, I stored the `slope`, `offset`, `p1`, `p2` which are `m`, `b`, point1 and point2 respectively. Additionally, in the first pass I also find the dominant signal, i.e., the largest `magnitude`. My assumption is that the slope of the dominant signal is roughly the opposite of the other lane. Therefore, I average the absolute value of the minimum and maximum slopes for the dominant signals above `magnitude_thresh`. I also throw out any point whose `x1 == x2` or `y1 == y2`.. these would result in zero-value slope that will cause a crash division later. To triage any divides by zero, I also `assert` divisors are non-zero within `try .. except` blocks.

In the second pass over the data, I separate the negative slopes (or right-side) from the positive (or left-side). This is useful because collection statistical data about both groups seems less meaningful for the test cases.

During the third pass over the data, or the interpolation section, I extend the lines using the ROI `y_min, y_max` as the input to the line equation `y = mx + b` or rather `x = (y - b)/m`. Since in the input has already been filter by ROI, the ouput is simply interpolated value. For some of these values the slope is too steep and the interpolation will result in a line which extends outside the image. These outliner lines are thrown out here.

The only `assert` exception is in the last pass.. all zero divisors should have been filtered before this stage. When finding the mean of the `x1, x2` for left-lane and right-lane points in order to estimate a polygon shape - it is assumed that something would be very wrong to have a zero divisor at this stage.


### 2. Potential shortcomings and failures in the current pipeline


My naive assumption, due to the immediate deadline, was to implement a linear filter based on the line equation: `y = mx + b`. For obvious reasons this will not work on all roads, because roads have turns. A better approach might be to consider all the connections between Hough line segments within the ROI and use tangential relationship to approximate curved lines using `[A * cos(theta), B * sin(theta)]`, where `theta` is the angle relative to the center of the ROI and `A, B` represent the magnitude of vectors from the ROI center to hough segment: `(x1, y1), (y1, y2)` respectively. The derivative of `theta` could be used to create a linear approximation of the lane (especially if curved), and using the relationship `tan = sin/cos` of the intersection I could then approximate the sum of piecewise `[(A1 cos(t1), B1 sin(t1)), (A2 cos(t2), B2 sin(t2)),...(An cos(tn), Bn sin(tn))]` lane lines for left and right sides.

True safety practice might be to apply `try .. except` around ALL `assert` sections, for a real application.

An undiscovered failure in the second pass of the data - I should also determine if the negative slope (right-side) line or the positive slope (left-side) line is actually on the left or right side of the ROI center. These lines could be thrown out pre-interpolation, as they have at least one point who resides in the ROI in the first pass.

Another shortcoming is that the prior needs a dominant signal to bias the mean slope calculation. This will very quickly break down when the car is in a center lane on a highway. There will be broken lines on either side and the dominant signal may be the side of the road. However, in the current implementation this may result in NO lanes detected, as the interpolated lines may extend outside of the ROI - and would be thrown out. I attempted to compensate for this scenario by setting the dominant signal to be fairly conservative magnitude of `40`.

The most interesting failure is when shadows from trees or buildings are introduced onto the road. Although the use of the canny edges attempts to reduce the dependancy of color space filters, a shadow creates a hard transition non-the-less. The false positives can be introduced early in the current pipeline when qualifying dominant magnitudes of line segments, and their slope (if too steep) have potential to bias the mean slope calculation. In turn, the line interpolation may throw out all lines and the end result would be, again, a possiblity of no lanes detected.


### 3. Improvements to the Pipeline


I believe the following improvements can be made:

1) As mentioned above, the use of `sin` and `cos`, rather than naive assumtion that all roads are straight.
2) Correlate Hough line segments relative to ROI center.
3) Implement a confidence filter on top of the pipeline, so that missing lanes may be approximated based on the prior.
4) Add test cases for occulsions (semi-trucks, road signs, etc..), shadows, and other obstructions which may falsely influence the annotation.
5) Iterate, iterate, iterate..
