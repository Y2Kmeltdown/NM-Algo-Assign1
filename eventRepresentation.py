########################### IMPORTS ##########################

from numpy.lib.recfunctions import unstructured_to_structured
import cv2 as cv
import numpy as np
from time import time
import event_stream

##############################################################


windowName = "Event Representation" # Initial Parameters for open cv window display

video_src = 'Pexels Videos 2629.mp4' # Video Relative Source Path

data = cv.VideoCapture(video_src) # open cv read mp4 file

frames = data.get(cv.CAP_PROP_FRAME_COUNT) # Get Video Frame Count
print(frames)

fps = data.get(cv.CAP_PROP_FPS) # Get Video Frame Rate
print(fps)

ret, img = data.read() # Get First Frame Data

frameInterval = 1/fps # Get Frame Period
print(frameInterval)

frameTime = 0 # Initial Frame Time Incrementer


height = 480
width = 640

native_width = data.get(3)
native_height = data.get(4)
ret = data.set(3, width)
ret = data.set(4, height)
ret, img = data.read()


thresh = 50

cv.namedWindow(windowName, 1)

ret, img = data.read()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype('int16')
prevImg = img

last_t = time()

resX = np.shape(img)[0]
resY = np.shape(img)[1]



dtype = np.dtype([('t', '<u8'), ("x", '<u2'), ("y", '<u2'), ("on", '?')])

with open(f"EventOutput\\{video_src[:-4]}.csv", 'a') as csvFile:
        np.savetxt(csvFile, np.zeros(1, dtype=dtype), delimiter=',', fmt=["%d","%d","%d","%d"], header="timestamp,x,y,polarity") # Write header out to file

encoder = event_stream.Encoder(f"EventOutput\\{video_src[:-4]}.es", 'dvs', resX, resY)

while(data.isOpened()):
    ret, img = data.read()

    if type(img) == type(None):
        break
    
    t = time()
    rate = 1/(t-last_t)

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype('int16')

    eventMask = cv.compare(cv.absdiff(img, prevImg), thresh, cv.CMP_GT)

    change = cv.subtract(img, prevImg) # Get light difference

    eventFrame = cv.bitwise_and(change, change, mask = eventMask) # Get Event Data in frame
    
    eventCoords = np.nonzero(eventFrame) # Get coordinates of all nonzeros in frame mask

    eventData = eventFrame[eventFrame != 0] #Get values of all nonzeros in frame mask

    eventSpikes = np.asarray(np.divide(eventData, thresh), dtype=np.int16) # Interpolate number of spikes between frames

    spikeNum = np.abs(eventSpikes) # get total number of interpolated events for each pixel between frames
    
    polarity = np.equal(np.divide(eventSpikes, spikeNum), 1) # get polarity of events for each pixel
    
    currentTmstp = np.full(np.shape(polarity), frameTime*1000000) # make array of timestamps

    Events = np.stack((currentTmstp, eventCoords[0], eventCoords[1], polarity)) # join all arrays together
    Events = np.transpose(Events) #Transpose array
    Events = unstructured_to_structured(Events, dtype=dtype) # Fit array to structured array



##########################Interpolation#####################################
    spikeMax = np.max(spikeNum) # Get maximum number of interpolated spikes found between frames
    
    if frameTime != 0:
        spikeTimeDelta = np.divide(frameInterval, spikeNum) # get spike time delta for each interpolated event between frames

        eventChunk = [] #Initialise Interpolation List

        for i in np.arange(1,spikeMax): # Loop up to maximum spike times found

            timeTemp = np.multiply(spikeTimeDelta*1000000, i) #Create array of interpolated time deltas
            timeTemp = np.asarray(timeTemp[spikeNum > i],dtype=np.float32) # remove interpolated time deltas that don't correspond to events

            polarityTemp = polarity[spikeNum > i] # Create array of polarities of interpolated events
            
            coordXTemp = np.asarray(eventCoords[0][spikeNum > i], dtype=np.uint16) # Create array of interpolated event X coordinates
            coordYTemp = np.asarray(eventCoords[1][spikeNum > i], dtype=np.uint16) # Create array of interpolated event Y coordinates

            event = np.stack((timeTemp, coordXTemp, coordYTemp, polarityTemp)) # Join interpolated information into single array
            eventChunk.append(event) # add chunk of interpolated events to list

        interpolatedEvents = np.concatenate(eventChunk, axis=1) # Concatenate all event chunks from list
        intpTmstp = np.full((1, np.size(interpolatedEvents, 1)), frameTime*1000000) # Create timestamp array the same size as the interpolated events
        interpolatedEvents[0,:] = np.subtract(intpTmstp, interpolatedEvents[0,:]) # Calculate time delta from current timestamp of interpolated events

        interpolatedEvents = np.transpose(interpolatedEvents) # Transpose interpolated event array
        interpolatedEvents = unstructured_to_structured(interpolatedEvents, dtype=dtype) # Fit array to structured array

        interpolatedEvents = np.sort(interpolatedEvents ,order='t', kind='stable') # Sort Structured Interpolated Array by timestamp

        Events = np.concatenate((interpolatedEvents, Events)) # Join Interpolated Events and Regular Events

    

    with open(f"EventOutput\\{video_src[:-4]}.csv", 'a') as csvFile:
        np.savetxt(csvFile, Events, delimiter=',', fmt=["%d","%d","%d","%d"]) # Write array out to file
        encoder.write(Events)
  

##############################DISPLAY DATA####################################
    (_, posEvent) = cv.threshold(cv.subtract(img, prevImg), thresh, 127, cv.THRESH_BINARY_INV)

    (_, negEvent) = cv.threshold(cv.subtract(prevImg, img), thresh, 255, cv.THRESH_BINARY)

    displayFrame = cv.max(negEvent, posEvent)

    last_t = t
    
    frameTime += frameInterval

    disp = np.asarray(displayFrame, dtype= np.uint8)

    cv.putText(disp, "FPS = %f"%(rate), (10,height-10), cv.FONT_HERSHEY_SIMPLEX, .5, 1)
    cv.imshow(windowName, disp)

    prevImg = img

    c = cv.waitKey(1) & 0xFF
    if c == ord("q"):
        break

data.release()
cv.destroyAllWindows()
c= cv.waitKey(1)