# **Finding Lane Lines on the Road** 

---

In this project, I created a pipeline to find lane on a road. The focus of the project was to make use of dominant signals within the image to bias lane construction. Also the focus intended to identify straight stretches of road with occluded areas.

[//]: # (Image References)

[image1]: ./test_images_output/begin_solidYellowCurve2.jpg "begin"
[image2]: ./test_images_output/gaussian_solidYellowCurve2.jpg "grayscale"
[image3]: ./test_images_output/canny_solidYellowCurve2.jpg "canny"
[image4]: ./test_images_output/hough_solidYellowCurve2.jpg "hough"
[image5]: ./test_images_output/roi_solidYellowCurve2.jpg "roi"
[image6]: ./test_images_output/end_solidYellowCurve2.jpg "end"

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
    * Introducing noise help pronouce dominate features, and mutes textures
![alt text][image2]

3) Apply Canny edge detection to gaussian grayscale
    * This performs the most work to isolate features
![alt text][image3]

4) Apply Hough transform to Canny edges image
    * This performs the other magic part, by corealating Euclian space to Hough space
    * The default behavior of the pipeline is to drawing the output
    * The `draw_lines` method could have contained a bunch of math, instead, these became the helper functions
    * I added `land_detect/line_math.py` to maintain all the line/slope/interpolation/filtering
![alt text][image4]

5) Apply the Region of Interest (ROI) mask
    * I maintained a pyramid shape, with a flat top as the default roi shape, inside `LaneFilter.__init__`
![alt text][image5]

6) Apply the weighted image using original, mask and transform combination
![alt text][image6]

#### Logic around annotations

The `draw_lines` function could have become an giant mess of logic. Instead of cramming all that logic into one function, I made several helper functions in `line_math.py`. The Hough transform returns several line segments, some which overlap and others that are complete outliers for lane data.

My approach as to first generate the ROI from within `draw_lines`. I then convert the `ndarray` to an `array(dtype=bool)`. This allows me to directly use the points from the Hough transform in the first pass of the data, to disqualify any point pairs which do not have at least on point in the ROI.

Also in the first pass over these data, I stored the `slope`, `offset`, `p1`, `p2` which are `m`, `b`, point1 and point2 respectively. Additionally, in the first pass we also find the dominant signal, i.e., the larges `magnitude`. My assumption is that the slope of the dominant signal is roughly the opposite of the other lane. Therefore, I average the absolute value of the minimum and maximum slopes for the dominant signals above `magnitude_thresh`.

In the second pass over the data, I separate the negative slopes (or right) from the positive. This is usful because collection statistical data about both groups seems less meaningful for the test cases.

During the third pass over the data, or the interpolation section, I extend the lines using the ROI `(y_min, y_max)` as the input to the line equation `y = mx + b` or rather `x = (y - b)/m`. 


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
