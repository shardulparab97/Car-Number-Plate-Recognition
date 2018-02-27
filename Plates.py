import cv2
import numpy as np
import math
import random
import preprocess
import Chars

showOperation = False
############################Module based variables
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

        # constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

        # other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
###################################################

#class to get the countour geometry

class getContourGeometry:

    def __init__(self,_contour):
        self.contour = _contour

        #get bouding Rectangle properties
        self.boundingRect = cv2.boundingRect(self.contour)

        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight

        #find bounding area
        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight

        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2

        #get diagonal size and aspect ratio
        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))

        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)

##########################################################################################

class PossiblePlate:
#The class to initialie Possible Plate properties
    def __init__(self):
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""


##########################################################################################

def getPossiblePlates(originalImage):
    listOfPossiblePlates = []

    height, width, _ = originalImage.shape #getting dimensions

    #creating empty arrays for storing corresponding image operations data
    GrayscaleImage = np.zeros((height,width,1), np.uint(8))
    ThresholdImage = np.zeros((height,width,1), np.uint(8))
    Image_Contours = np.zeros((height,width,3), np.uint(8))

    #show_operation
    if showOperation == True:
        cv2.imshow("Original Image - 0",originalImage)

    GrayscaleImage, ThresholdImage = preprocess.preprocess(originalImage)

    #show_operation
    if showOperation == True:
        cv2.imshow("GrayScale Image - 1a", GrayscaleImage)
        cv2.imshow("Thresholded Image - 1b",ThresholdImage)
        cv2.waitKey()

    listOfPossibleContours = findPossibleContoursInImage(ThresholdImage)

    if showOperation== True: # show steps #######################################################
        print "step 2 - len(listOfPossibleContours) = " + str(len(listOfPossibleContours))         # 131 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleContours:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, (255.0,255.0,255.0)) #show all contours
        cv2.imshow("2b", imgContours)

    megalistOfMatchingContours =  Chars.findMegalistOfMatchingChars(listOfPossibleContours)
    #print (megalistOfMatchingContours)

    if showOperation == True: # show steps #######################################################
        print "step 3 - megaListOfMatchingContours - Count = " + str(len(megalistOfMatchingContours))    # 13 with MCLRNF1 image

        imgContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingContours in megalistOfMatchingContours:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingContour in listOfMatchingContours:
                contours.append(matchingContour.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
        # end for

        cv2.imshow("All matching contours - 3", imgContours)

    #For each group of matching contours find the list of possible possible plates
    for listOfMatchingContours in megalistOfMatchingContours:
        possiblePlate = extractPlate(originalImage, listOfMatchingContours)

        if possiblePlate.imgPlate is not None:
            listOfPossiblePlates.append(possiblePlate)

    print "\n" + str(len(listOfPossiblePlates)) + " possible plates found image"          # 13 with MCLRNF1 image

    if showOperation == True: # show steps #######################################################
        print "\n"
        # cv2.imshow("4a", imgContours)

        #we are using this area to draw all a red box around the obtained plates
        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            #drwaing a red boundary around
            cv2.line(imgContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), (0,0,255.0), 2)
            cv2.line(imgContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), (0,0,255.0), 2)
            cv2.line(imgContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), (0,0,255.0), 2)
            cv2.line(imgContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), (0,0,255.0), 2)

            cv2.imshow("4a - Showing the image with plates around matching contours", imgContours)

            print "All Possible plated detected are shown " + str(i) + ", click on any image and press a key to continue . . ."

            cv2.imshow("All possible plates - 4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print "\nplate detection complete, click on any image and press a key to begin char recognition . . .\n"
        cv2.waitKey(0)
    # end if # show steps #########################################################################

    return listOfPossiblePlates

    # end if # show steps #########################################################################

####################################################################

def findPossibleContoursInImage(ThresholdImage):
    #This function finds the right contours accoring to geometric sizes

    listOfPossibleContours = []

    countOfPossibleContours = 0

    #creating a copy of Threshold image as Threshold image may change
    ThresholdImageCopy = ThresholdImage.copy()

    height, width = ThresholdImage.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    #get all the contours i.e. the closed curves
    _, allContours, _= cv2.findContours(ThresholdImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0,len(allContours)):

        if showOperation == True:
            cv2.drawContours(imgContours, allContours, i, (255.0, 255.0, 255.0) ) #giving white as color

        possibleContour = getContourGeometry(allContours[i]) #store possible contour geometry

        #If it is a possible contour, increment the count and the list of contours
        if checkIfPossibleContour(possibleContour):
            countOfPossibleContours = 1 + countOfPossibleContours
            listOfPossibleContours.append(possibleContour)
        

    if showOperation== True: # show steps #######################################################
        print "\nstep 2 - len(contours) = " + str(len(allContours))                       # 2362 with MCLRNF1 image
        print "step 2 - intCountOfPossibleChars = " + str(countOfPossibleContours)       # 131 with MCLRNF1 image
        cv2.imshow("DRAWING ALL CONTOURS - 2a:", imgContours)

    #returns the entire list of contours passing the first checkpoint
    return listOfPossibleContours


        #print possibleContour

#############################################################################
def checkIfPossibleContour(possibleContour):
#This function is a 'first pass' that does a rough check on whether contour to see if it could be a plate based on geometric properties
    if (possibleContour.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleContour.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleContour.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleContour.fltAspectRatio and possibleContour.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function
##############################################################################

def extractPlate(imgOriginal, listOfMatchingContours):
    possiblePlate = PossiblePlate()           # this will be the return value

    listOfMatchingContours.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position

            # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingContours[0].intCenterX + listOfMatchingContours[len(listOfMatchingContours) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingContours[0].intCenterY + listOfMatchingContours[len(listOfMatchingContours) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

            # calculate plate width and height
    intPlateWidth = int((listOfMatchingContours[len(listOfMatchingContours) - 1].intBoundingRectX + listOfMatchingContours[len(listOfMatchingContours) - 1].intBoundingRectWidth - listOfMatchingContours[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingContours:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingContours)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

            # calculate correction angle of plate region
            #skew correction angle ZL Hong
    fltOpposite = listOfMatchingContours[len(listOfMatchingContours) - 1].intCenterY - listOfMatchingContours[0].intCenterY
    fltHypotenuse = Chars.distanceBetweenChars(listOfMatchingContours[0], listOfMatchingContours[len(listOfMatchingContours) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            # final steps are to perform the actual rotation

            # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possiblePlate.imgPlate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate

    return possiblePlate
# end function
