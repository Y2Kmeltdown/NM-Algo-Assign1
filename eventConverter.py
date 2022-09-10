########################### IMPORTS ##########################

from numpy.lib.recfunctions import unstructured_to_structured
import cv2 as cv
import numpy as np
import event_stream

##############################################################

video_src = 'Pexels Videos 2629.mp4' # Video Relative Source Path
data = cv.VideoCapture(video_src) # open cv read mp4 file
fps = data.get(cv.CAP_PROP_FPS) # Get Video Frame Rate
frameInterval = 1/fps # Get Frame Period
frameTime = 0 # Initial Frame Time Incrementer
thresh = 30
ret, img = data.read()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype('int16')
prevImg = img



dtype = np.dtype([('t', '<u8'), ("x", '<u2'), ("y", '<u2'), ("on", '?')])
#with open(f"EventOutput\\{video_src[:-4]}.csv", 'a') as csvFile:
        #np.savetxt(csvFile, np.zeros(1, dtype=dtype), delimiter=',', fmt=["%d","%d","%d","%d"], header="timestamp,x,y,polarity") # Write header out to file

resX = np.shape(img)[0]
resY = np.shape(img)[1]
encoder = event_stream.Encoder(f"EventOutput\\{video_src[:-4]}.es", 'dvs', resX, resY)

while(data.isOpened()):
    ret, img = data.read()

    if type(img) == type(None):
        break

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

    encoder.write(Events) # Write array out to ES file

    #with open(f"EventOutput\\{video_src[:-4]}.csv", 'a') as csvFile:
        #np.savetxt(csvFile, Events, delimiter=',', fmt=["%d","%d","%d","%d"]) # Write array out to file
        
  
    frameTime += frameInterval
    prevImg = img

    

data.release()
