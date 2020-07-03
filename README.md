

---

## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration & distortion

in the first step I am worked to remove the distortion in the image , by usig an image for a chess board and detect the corners on the image and using object points which are an array of coordinates. Also I detected Image points using builtin function `cv2.findChessboardCorners(gray, (9,6),None)`

`ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)`


![alt text](https://i.ibb.co/J7tqxrg/1.png)

### Pipeline (single images)



####  1. removing distortion 

here is an example of removing the distortion in one of my tested images  on the road

![alt text](https://i.ibb.co/41prPS4/index3.png)

#### 2. different types of thresholding
here I tried to test different types of thresholding like gradient threshold , magnitude threshold and color threshold.
finally I combined between color threshold and gradient threshold to get better results

```python
def combine(img, s_thresh=(120, 255), sx_thresh=(15, 100)):
    img=np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    sxbinary=sobel(l_channel,sx_thresh)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary
```


![alt text](https://i.ibb.co/Fhdzpd3/index2.png)

#### 3. warping the Image 

Then I warped the image to get the birdeye view , this step will help us to detect curvature on the road 

```python
def warper(img):
    #get the width and the height of the image
    h=img.shape[0]
    w=img.shape[1]
    
    
    src = np.float32([
    (585, 460), # Top-left corner
    (203, 720), # Bottom-left corner
    (1127, 720), # Bottom-right corner
    (695, 460) # Top-right corner
    ])
    
    dst = np.float32([
    (320, 0), # Top-left corner
    (320, 720), # Bottom-left corner
    (960, 720), # Bottom-right corner
    (960, 0) # Top-right corner
     ])
    #calculate perspective transform matrix
    M=cv2.getPerspectiveTransform(src,dst)
    #warp the image
    warped=cv2.warpPerspective(img,M,(w,h))
    return warped,M
```
![alt text](https://i.ibb.co/DWBzPQz/index6.png)

#### 4. detect the pixels that belong to the right nd left lane line

the first step to define the pixels that belong to the lines is defining histogram that describes the activited pixels in the binary warped image  `def histogram(img):
    #normalize the image
    img=img/255
    #work with the bottom half of the warped image
    bottom_half=img[img.shape[0]//2:,:]
    #sum the pixels vertically
    histogram=np.sum(bottom_half,axis=0)
    return histogram`
![alt text](https://i.ibb.co/LJBpytC/index7.png)
#### 5. define where the lines go

After finding the peaks of the histogram and use it as a starting point , I used sliding windows to detect where the lines go
```python
# Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img
```
![alt text](https://i.ibb.co/m5jPRV8/w.png)

#### 6. area of search
after defining the polynomial i will create area of search instead of blind search
![alt text](https://i.ibb.co/MsHdMZ3/screen-shot-2017-01-28-at-12-39-43-pm.png)

#### 7. calculate radius of curvature 
then I calculated the radius of curvature using the polynomial and I converted the values from pixels into meters
 
```
y_eval = np.max(ploty)
    ym_per_pix = 30/720
    xm_per_pix = 3.7/700
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad
 ```
 
#### 7.offset
finally i calculated the position of the vehicle with respect to the center
```
center=(right_fitx[-1]+left_fitx[-1])/2
    xm_per_pix = 3.7/700
    #calculate the offset  of the lane center from the center of the image converted to meters
    center = (center-warped.shape[1]/2)*xm_per_pix
    #distance=distance*xm_per_pix
    return center
```

### finally I projected my calculations into the original image
![alt text](https://i.ibb.co/6FPjFXy/f.png)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](output_images/solidWhiteRight.mp4)

---

### Discussion


Here I used combine thresholding after removing destortion after that I implemented bird-eye view to detect curvature by plotting polynomial using sliding window and area of search techniques , finally I calculated the radius of curvature and the position of the car using the polynomial.
I am facing a problem with working under some condition like changing the style of the rod , different degrees of light
