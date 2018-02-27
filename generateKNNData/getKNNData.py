# GenData.py

import sys
import numpy as np
import cv2
import os

# module level variables ##
#setting minimum contour area to pick
MIN_CONTOUR_AREA = 100


RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def main():
    trainingChars = cv2.imread("trainingChars.png")            # read in training numbers image

    #no success
    if trainingChars is None:
        print "error: image not read from file \n\n"
        os.system("pause")
        return
    # end if

    imgGray = cv2.cvtColor(trainingChars, cv2.COLOR_BGR2GRAY)          # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur - to eliminate noise , to get contours properly

    #Now using the adaptive threshold
    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # Observed that Gaussian works better
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    #Showing the thresholded image
    cv2.imshow("imgThresh", imgThresh)

    #Making a copy as the function of contours make changes to original image
    imgThreshCopy = imgThresh.copy()

    imgContours, allContours, _ = cv2.findContours(imgThreshCopy,        # input image
                                                 cv2.RETR_EXTERNAL,                 # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)           # compress horizontal, vertical, and diagonal segments and leave only their end points

    #Create an empty array to store values later on
    FlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)) #we get a 20 X 30 array

    #For storing classification
    intClassifications = []

    # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
    #We can also make additions to the same
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]
    print len(allContours)
    for Contour in allContours:                          # for each contour
        if cv2.contourArea(Contour) > MIN_CONTOUR_AREA:   # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(Contour)  # Create a bounding rectangle

            # draw rectangle around each contour as user is asked for input
            cv2.rectangle(trainingChars,           # draw rectangle on original training image
                          (intX, intY),                 # upper left corner
                          (intX+intW,intY+intH),        # lower right corner
                          (0, 0, 255),                  # red
                          2)                            # thickness

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage

            cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
            cv2.imshow("training_numbers.png", trainingChars)      # show training numbers image, this will now have red rectangles drawn on it

            intChar = cv2.waitKey(0)  # get key press
            #print(intChar)
            if intChar == 27:                   # if esc key was pressed
                sys.exit()                      # exit program
            elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for

                intClassifications.append(intChar)                                                # append classification char to integer list of chars (we will convert to float later before writing to file)

                FlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # flatten image to 1d numpy array so we can write to file later
                FlattenedImages = np.append(FlattenedImages, FlattenedImage, 0)                    # add current flattened impage numpy array to list of flattened image numpy arrays
            # end if
        # end if
    # end for

    fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats

    Classifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

    print "\n\ntraining complete !!\n"

    np.savetxt("classifications1.txt", Classifications)           # write flattened images to file
    np.savetxt("flattened_images1.txt", FlattenedImages)          #

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if




