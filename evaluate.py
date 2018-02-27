#This file runs the code for all the images in a given folder and gives the corresponding predicted registration number

import os
import cv2
import main
import Plates
import Chars
import KNNFile
showOperation = False

directory = './Images'
"""for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        Image=(os.path.join(directory, filename))
        img = cv2.imread(Image)
        cv2.imshow(Image,img)
        cv2.waitKey()
        continue
    else:
        continue"""

val=0
def drawBoxAroundPlate(originalImage,finalPlate):

    getCoordinates = cv2.boxPoints(finalPlate.rrLocationOfPlateInScene)
    RED=(0,0,255.0)
    cv2.line(originalImage, tuple(getCoordinates[0]), tuple(getCoordinates[1]),RED, 2)
    cv2.line(originalImage, tuple(getCoordinates[1]), tuple(getCoordinates[2]),RED, 2)
    cv2.line(originalImage, tuple(getCoordinates[2]), tuple(getCoordinates[3]),RED, 2)
    cv2.line(originalImage, tuple(getCoordinates[3]), tuple(getCoordinates[0]),RED, 2)

for i in range(1,17):
    Image = "./Dataset/LicensePlates/("+str(i)+").png"
    img = cv2.imread(Image)

    KNNData = KNNFile.loadKNNData() #Running training of KNN

    if KNNData!=True:
        print "\n UNABLE TO RUN KNN TRAINING \n"
        exit()

    originalimage  = cv2.imread(Image)

    if originalimage is None:
        print "\n UNABLE TO READ IMAGE PROPERLY, PLEASE TRY AGAIN  "
        continue

    listOfPossiblePlates = Plates.getPossiblePlates(originalimage)

    #now we move towards obtaining proper characters in the plates by using various constraints
    #we are only sending the list of possible plates obtained in the previous function call
    listOfPossibleCharsInPossiblePlates = Chars.getPossibleCharsInPlates(listOfPossiblePlates)

   # cv2.imshow("Original Image", originalimage)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print "\nno license plates were detected\n"             # inform user no plates were found
        with open("Predicted_Values.txt", "a") as myfile:
            myfile.write("\n")
    else:                                                       # else

        #We have to now find the plates with the most number of recognized characters.
        #Hence we sort in descending order and take the plate with most characters
        listOfPossibleCharsInPossiblePlates.sort(key = lambda possibleCharsinPlate: len(possibleCharsinPlate.strChars), reverse = True)


        finalPlate = listOfPossibleCharsInPossiblePlates[0]

       # cv2.imshow("imgPlate", finalPlate.imgPlate)           # show crop of plate and threshold of plate
       # cv2.imshow("imgThresh", finalPlate.imgThresh)
       # cv2.waitKey()
        if len(finalPlate.strChars) == 0:                     # if no chars were found in the plate
            print "\nNo characters were detected\n\n"  
            with open("Predicted_Values.txt", "a") as myfile:
                myfile.write("\n")     # show message
            continue

        drawBoxAroundPlate(originalimage, finalPlate)
        # draw red rectangle around plate
        print "----------------------------------------"
        print "\nThe registration number for "+Image+" is: = " + finalPlate.strChars + "\n"       # write license plate text to std out
        print "----------------------------------------"


        with open("Predicted_Values.txt", "a") as myfile:
            myfile.write(finalPlate.strChars+"\n")


#Looping over all the 100 IMAGES in the dataset and getting the corresponding vehicle plate number
for i in range(1,101):
    Image = "./Dataset/("+str(i)+").jpg"
    img = cv2.imread(Image)

    KNNData = KNNFile.loadKNNData() #Running training of KNN

    if KNNData!=True:
        print "\n UNABLE TO RUN KNN TRAINING \n"
        exit()

    originalimage  = cv2.imread(Image)

    if originalimage is None:
        #print "\n UNABLE TO READ IMAGE PROPERLY, PLEASE TRY AGAIN  "
        continue

    listOfPossiblePlates = Plates.getPossiblePlates(originalimage)

    #now we move towards obtaining proper characters in the plates by using various constraints
    #we are only sending the list of possible plates obtained in the previous function call
    listOfPossibleCharsInPossiblePlates = Chars.getPossibleCharsInPlates(listOfPossiblePlates)

   # cv2.imshow("Original Image", originalimage)            # show scene image

    if len(listOfPossiblePlates) == 0:                          # if no plates were found
        print "\nno license plates were detected\n"  
        with open("Predicted_Values.txt", "a") as myfile:
            myfile.write("\n")           # inform user no plates were found
    else:                                                       # else

        #We have to now find the plates with the most number of recognized characters.
        #Hence we sort in descending order and take the plate with most characters
        listOfPossibleCharsInPossiblePlates.sort(key = lambda possibleCharsinPlate: len(possibleCharsinPlate.strChars), reverse = True)


        finalPlate = listOfPossibleCharsInPossiblePlates[0]

       # cv2.imshow("imgPlate", finalPlate.imgPlate)           # show crop of plate and threshold of plate
       # cv2.imshow("imgThresh", finalPlate.imgThresh)
       # cv2.waitKey()
        if len(finalPlate.strChars) == 0:                     # if no chars were found in the plate
            print "\nNo characters were detected\n\n"
            with open("Predicted_Values.txt", "a") as myfile:
                myfile.write("\n")       # show message
            continue

        drawBoxAroundPlate(originalimage, finalPlate)
        # draw red rectangle around plate
        print "----------------------------------------"
        print "\nThe registration number for "+Image+" is: = " + finalPlate.strChars + "\n"       # write license plate text to std out
        print "----------------------------------------"


    with open("Predicted_Values.txt", "a") as myfile:
        myfile.write(finalPlate.strChars+"\n")



"""with open("Predicted_Values.txt","r") as myfile:
    plateChars = myfile.readlines()

totalChars = 0
for plateChar in plateChars:
    print plateChar
    print "Number of characters = ",len(plateChar)-1
    totalChars+=len(plateChar)-1
    for ch in plateChar:
        if ch == "\n":
            break
        print ch

print "Total Number of Characters detected = ", totalChars
"""

print("##########################FINAL EVALUATION ###########################")
with open ("actual_num.txt", "r") as myfile:
    data=myfile.readlines()
testdata=""
for line in data:
    testdata+=line

with open ("Predicted_Values.txt", "r") as myfile:
    data=myfile.readlines()
preddata=""
for line in data:
    preddata+=line

val+=len(preddata)
print("Total Number of characters to be predicted:",len(testdata))
print("Total number of characters in plates detected by program:",val)
print("#################################################################")


