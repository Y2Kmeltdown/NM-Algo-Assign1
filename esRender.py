########################### IMPORTS ##########################

import argparse
import pathlib
from numpy.lib.recfunctions import unstructured_to_structured
import cv2 as cv
import numpy as np
from time import time
import event_stream

##############################################################

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

dirname = pathlib.Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description="Generate event stream data from a .mp4 frame based video",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("input", help="input .mp4 file")
parser.add_argument(
    "--output",
    "-o",
    help="output folder path",
)
parser.add_argument(
    "--threshold",
    "-th",
    type=int,
    default=50,
    choices=range(1,256),
    metavar="[1-255]",
    help="set light intensity thresholding",
)
parser.add_argument(
    "--csv",
    action="store",
    dest="CSV",
    type=str,
    choices=['yes', 'no'],
    default="no",
    help="create csv output",
)

args = parser.parse_args()

input = pathlib.Path(args.input).resolve()
if args.output is None:
    outputEs = (
        input.parent
        / f"{input.stem}_threshold={args.threshold}.es"
    )
    if args.CSV == "yes":
        outputCsv = (
            input.parent
            / f"{input.stem}_threshold={args.threshold}.csv"
        )
else:
    outputEs = (
        pathlib.Path(args.output).resolve()
        / f"{input.stem}_threshold={args.threshold}.es"
    )
    if args.CSV == "yes":
        outputCsv = (
            pathlib.Path(args.output).resolve()
            / f"{input.stem}_threshold={args.threshold}.csv"
        )

data = cv.VideoCapture(str(input)) # open cv read mp4 file

frames = data.get(cv.CAP_PROP_FRAME_COUNT) # Get Video Frame Count
fps = data.get(cv.CAP_PROP_FPS) # Get Video Frame Rate
frameInterval = 1/fps # Get Frame Period
totalT = frameInterval*frames


ret, img = data.read()
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype('int16')

thresh = args.threshold
if args.CSV == "yes":
    csvFlag = True
else:
    csvFlag = False

prevImg = img
last_t = time()
frameTime = 0 # Initial Frame Time Incrementer

print(f"Rendering file: {input.stem}")
print(f"Total Frames: {frames}")
print(f"Input Frame Rate: {fps}")
print(f"Event Threshold: {thresh}")

# Initial call to print 0% progress
printProgressBar(0, totalT, prefix = 'Progress:', suffix = f'Complete | Framerate = 0', length = 50)

dtype = np.dtype([('t', '<u8'), ("x", '<u2'), ("y", '<u2'), ("on", '?')])
if csvFlag:
    with open(outputCsv, 'a') as csvFile:
            np.savetxt(csvFile, np.zeros(1, dtype=dtype), delimiter=',', fmt=["%d","%d","%d","%d"], header="timestamp,x,y,polarity") # Write header out to file

resX = np.shape(img)[0]
resY = np.shape(img)[1]
encoder = event_stream.Encoder(outputEs, 'dvs', resX, resY)

while(data.isOpened()):
    ret, img = data.read()

    if type(img) == type(None):
        break
    
    t = time()
    rate = 1/(t-last_t)

    ############################ FRAME INSPECTION ##########################################

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


    ########################## EVENT INTERPOLATION #####################################
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

    ######################### WRITING OUTPUT ###############################################################
    encoder.write(Events)
    if csvFlag:
        with open(outputCsv, 'a') as csvFile:
            np.savetxt(csvFile, Events, delimiter=',', fmt=["%d","%d","%d","%d"]) # Write array out to file
        
  


    last_t = t
    
    frameTime += frameInterval
    printProgressBar(frameTime, totalT, prefix = 'Progress:', suffix = f'Complete | Framerate = {round(rate,2)}', length = 50)

    prevImg = img

data.release()





