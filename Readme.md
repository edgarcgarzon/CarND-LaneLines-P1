# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_image_output/originalImages.jpg
[image2]: ./test_image_output/ImagesHsv.jpg
[image3]: ./test_image_output/WhiteMask.jpg
[image4]: ./test_image_output/YellowMask.jpg
[image5]: ./test_image_output/ImagesColorMasked.jpg
[image6]: ./test_image_output/ImagesInGreyGaussianFilter.jpg
[image7]: ./test_image_output/ImagesColorEdges.jpg
[image8]: ./test_image_output/ImagesMaskedLines.jpg
[image9]: ./test_image_output/LineMaskImages.jpg
[image10]: ./test_image_output/ImagesMaskedExtrapolatedLines.jpg
[image11]: ./test_image_output/LineImages.jpg

---

### Reflection

### 1. Pipeline


 The pipeline consisted of 7 steps:
 
 1. Color Mask
 2. Gaussian filter
 3. Canny function
 4. Evaluation area mask
 5. hough Transform
 6. Extrapolate lines
 7. Extrapolation lines and original images
 
 
 The originial images before the pipeline:
 ![alt text][image1]
 
### i. Color mask
 
The images are converted to HLS color-space to increase the contrants of the white and yellow color. 

 ![alt text][image2]

Apply, separately, a yellow and white mask.

White mask:

 ![alt text][image3]
 
 Yellow mask:

 ![alt text][image4]
 
 Merge both filters and apply it to the original image.
 As it could be observed all the colors beside the yellow and white has been masked from the pictures. 
 Note that there is still some green due range of the white filter.
 
 It could be improve tunning the sensitivity:
 
```python
sensitivity = 55 
WhiteLowMask = np.uint8([ 0, 255-sensitivity, 0])
WhiteHighMask  = np.uint8([255, 255, 255])

sensitivity = 155 
YellowLowMask = np.uint8([ 0, 0, 255-sensitivity])
YellowHighMask = np.uint8([255, 255, 255])
```

 ![alt text][image5]
 
### ii. Gaussian filter
 
 Convert the image to 8-bit (gray scale images) and apply a Gaussian filter with the paramters:
 
```python
kernel_size = 1
```
![alt text][image6]

### iii. Canny function
 
Apply the canny function to identify the changes in the 8-bit image using the following thresholds:

```python
low_threshold = 50
high_threshold = 150

```

![alt text][image7]

### iv. Evaluation area mask

Define the evaluation area using a polygon and mask all the image outside this area.

The polygon is defines as percentual of the immage dimmension in order to be calibrated for a different image size and camera angle.

```python
PointDownLeft  = ( int(imshape[1]*0.10), int(imshape[0]) )
PointUpLeft    = ( int(imshape[1]*0.45),  int(imshape[0]*0.6) )
PointUpRight   = ( int(imshape[1]*0.55),  int(imshape[0]*0.6) )
PointDownRight = ( int(imshape[1]*0.95), int(imshape[0]) )

vertices = np.array([[PointDownLeft,PointUpLeft, PointUpRight, PointDownRight]], dtype=np.int32)

```

![alt text][image8]

### v. Hough transform

Apply the Hough trasform and get the hough lines using the parameters:

```python
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 15 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
```
![alt text][image9]

### vi. Extrapolate lines

Get the parameters for the left and right lines.
m: slope  
b:  offset


```python
def ExtrapolateLines(Lines):
    
    """
    Lines is the output of cv2.HoughLinesP.
    this function returns a tupple with the line parameters for right and left line
    Line parameters: 
        m:slope  
        b:offset
    """

    LeftLine = []
    RightLine = []
    
    #Define the minimum slope value to considere a lane line
    min_pos_slope = 0.3
    min_neg_slope = -0.3
    
    #For each set of points in lines, calculate the line paramters:
    # slope (m), offset (b) and lenght (D)
    for line in Lines:
        for x1, y1, x2, y2 in line:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            D = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
               
            #Append to the list the parameters only if the slope is higher than the minimum value 
            if (m > 0 and m > min_pos_slope):
                LeftLine.append((m, b, D))
            elif (m < 0 and m < min_neg_slope):
                RightLine.append((m, b, D))
            
    #Calulate the weight of all the left lines: Sum of lenghts:
    WeightOfLeftLine = sum(D for m, b, D in LeftLine)
    
    #If the weight is not zero, calculate the weighted average of m and b
    if(WeightOfLeftLine != 0):
        mLeftLine = sum(m * D for m, b, D in LeftLine) / WeightOfLeftLine
        bLeftLine = sum(b * D for m, b, D in LeftLine) / WeightOfLeftLine
    else:
        mLeftLine = 0
        bLeftLine = 0
    
    #Calulate the weight of all the left lines: Sum of lenghts:
    WeightOfRighLine = sum(D for m, b, D in RightLine)
    
    #If the weight is not zero, calculate the weighted average of m and b
    if(WeightOfRighLine != 0):
        mRightLine = sum(m * D for m, b, D in RightLine) / WeightOfRighLine
        bRightLine = sum(b * D for m, b, D in RightLine) / WeightOfRighLine
    else:
        mRightLine = 0
        bRightLine = 0
    
    #return a tupple with the line parameters for left and right lines
    return [(mLeftLine, bLeftLine), (mRightLine, bRightLine)]
    
```

Once the parameters are calulate, calculate 2 points of the line. The points are located in the bottom and top of the polygon use in the step above.

```python
def makePoints(lineParamerter, y2, y1):
    '''
    return a tupple with the (x,y) points from the y points and the line parameters.
    '''
    m, b = lineParamerter
    x2 = int((y2 - b) / m)
    x1 = int((y1 - b) / m)

    return [[(x1, y1, x2, y2)]]

imshape = Images[i].shape
LeftPoints = makePoints(LeftLineParam, int(imshape[0]*0.6), int(imshape[0]))
RightPoints = makePoints(RightLineParam, int(imshape[0]*0.6), int(imshape[0]))
```

Use these point to draw 2 lines in the image correponding to the left and right lane extrapolated.

![alt text][image10]

### vii. Extrapolation lines and original images

Draw the extrapolated lines in the original image.

![alt text][image11]

The all pipeline put together in one function is as follow:

Note that all the parameters are at the begining at the fuction.

---
```python
def process_image(image):
    
    #Define parameter for image processing:
    
    #Paramters for color mask
    #White color mask:
    sensitivity = 55 
    WhiteLowMask = np.uint8([ 0, 255-sensitivity, 0])
    WhiteHighMask  = np.uint8([255, 255, 255])

    sensitivity = 155 
    YellowLowMask = np.uint8([ 0, 0, 255-sensitivity])
    YellowHighMask = np.uint8([255, 255, 255])
    
    #Parameters for Gauss filter
    kernel_size = 5
    
    #Parameters for Canny function
    low_threshold = 50
    high_threshold = 150

    # Define parameters for Hough transform 
    rho = 1              # distance resolution in pixels of the Hough grid
    theta = np.pi/180    # angular resolution in radians of the Hough grid
    threshold = 15       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15 # minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    
    
    #Define the region of evaluation of the image
    imshape = image.shape
    PointDownLeft = (int(imshape[1] * 0.05), int(imshape[0]))
    PointUpLeft = (int(imshape[1] * 0.45), int(imshape[0] * 0.6))
    PointUpRight = (int(imshape[1] * 0.55), int(imshape[0] * 0.6))
    PointDownRight = (int(imshape[1] * 0.95), int(imshape[0]))
    vertices = np.array([[PointDownLeft, PointUpLeft, PointUpRight, PointDownRight]], dtype=np.int32)
    
    #Start the image pipeline
    
    #Color mask 
    #Convert image in HLS color-space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #Apply the white and yellow filter
    WhiteMask = cv2.inRange(hls, WhiteLowMask, WhiteHighMask)
    YellowMask = cv2.inRange(hls, YellowLowMask, YellowHighMask)
    #Combine white and yellow filter masks
    CombineMask = cv2.bitwise_or(WhiteMask,YellowMask)
    #Apply combined filter to the original image
    ImagesColorMasked = cv2.bitwise_and(image,image, mask=CombineMask)
    
    #Convert to grey scale and apply a gaussian filter
    ImageInGrey = grayscale(ImagesColorMasked)
    ImageInGreyGaussianFilter = gaussian_blur(ImageInGrey, kernel_size)
    
    #USe the Canny function over the 8bit filtered image
    ImageEdges = canny(ImageInGreyGaussianFilter, low_threshold, high_threshold)
    
    #Mask the edges with the evaluation region (polygon)
    ImageMasked = region_of_interest(ImageEdges, vertices)
    
    #Get the Hough lines from the masked image
    LinesImage, HoughLines = hough_lines(ImageMasked, rho, theta, threshold, min_line_length, max_line_gap)
    
    #Create a blank image to draw the extrapolation lines
    ImageExtrapolatedLines = np.copy(image) * 0
    
    #Caculated the extrapolated lines parameters
    (LeftLineParam, RightLineParam) = ExtrapolateLines(HoughLines)
    
    #Check if the slope is not zero
    #Calculate 2 points: at the bottom and top of the evaluation region, using the lines parameters
    #Draw the line in the blank image
    #Do it for the left and the right lane line
    if LeftLineParam[0] != 0:
        LeftPoints = makePoints(LeftLineParam, int(imshape[0] * 0.6), int(imshape[0]))
        draw_lines(ImageExtrapolatedLines, LeftPoints, [255, 0, 0], 10)
            
    if RightLineParam[0] != 0:
        RightPoints = makePoints(RightLineParam, int(imshape[0] * 0.6), int(imshape[0]))   
        draw_lines(ImageExtrapolatedLines, RightPoints, [255, 0, 0], 10)
    
    #Combine the original image with the extrapolated lines image
    ImageWithLines = weighted_img(ImageExtrapolatedLines, image)

    return ImageWithLines
```
---

### 2. Identify potential shortcomings with your current pipeline


The great shotcomming identify is when no line has been identified in the road. This could happend when in some part of the road the lines has been vanish or the shade of a tree or any other elements does not allow to identify the colors.

Due the following piece of code, no line will be draw, producing an intermitance in the extrapolated line.

Generally in a straigh lane, the last group of extrapolated linesm could be use could be used to cover this case.

```python
    
    #Caculated the extrapolated lines parameters
    (LeftLineParam, RightLineParam) = ExtrapolateLines(HoughLines)
    
    #Check if the slope is not zero
    #Calculate 2 points: at the bottom and top of the evaluation region, using the lines parameters
    #Draw the line in the blank image
    #Do it for the left and the right lane line
    if LeftLineParam[0] != 0:
        LeftPoints = makePoints(LeftLineParam, int(imshape[0] * 0.6), int(imshape[0]))
        draw_lines(ImageExtrapolatedLines, LeftPoints, [255, 0, 0], 10)
            
    if RightLineParam[0] != 0:
        RightPoints = makePoints(RightLineParam, int(imshape[0] * 0.6), int(imshape[0]))   
        draw_lines(ImageExtrapolatedLines, RightPoints, [255, 0, 0], 10)
    
    #Combine the original image with the extrapolated lines image
    ImageWithLines = weighted_img(ImageExtrapolatedLines, image)

    return ImageWithLines
```


### 3. Suggest possible improvements to your pipeline

During the vide processing, every frame is processing idenpendly to the precedent one. Therefore the extrapolate lines flicker during the video. The flickering is  more evident during the curve due the rapid change of the lines parameters (m, b),

In order to improve this behavior, a filter with memory could be applied to the extrapolated lines before drawing them. 
