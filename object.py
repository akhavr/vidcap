# https://www.pyimagesearch.com/2016/02/29/saving-key-event-video-clips-with-opencv/

# import the necessary packages
from pyimagesearch.keyclipwriter import KeyClipWriter

from imutils.video import VideoStream
from imutils.video import FPS

from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import datetime
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
        help="minimum probability to filter weak detections")
ap.add_argument("-b", "--buffer-size", type=int, default=32,
	help="buffer size of video clip writer")
ap.add_argument("--codec", type=str, default="MJPG",
	help="codec of output video")
ap.add_argument("-f", "--fps", type=int, default=20,
	help="FPS of output video")
ap.add_argument('--headless', dest='headless', action='store_true', help='Run headless')
ap.set_defaults(headless=False)
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()

#initialize the key clip writer and the motionFrames
# and consecFrames to track frames without motion
kcw = KeyClipWriter(bufSize=args["buffer_size"])
consecFrames = 0 #number of frames with no motion
prev_detections = None
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None

def classify_frame(net, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if inputQueue.empty():
            continue
        # grab the frame from the input queue, resize it, and
        # construct a blob from it
        frame = inputQueue.get()
        frame = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(frame, 0.007843,
                                     (300, 300), 127.5)
        # set the blob as input to our deep learning object
        # detector and obtain the detections
        net.setInput(blob)
        detections = net.forward()
        # write the detections to the output queue
        outputQueue.put(detections)


def loop_over_detections(frame, detections, prev_detections):
    detected = False
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the timestamp
            timestamp = datetime.datetime.now()
            cv2.putText(frame, timestamp.strftime("%Y.%m.%d %H:%M:%S"), (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # record this frame
            if prev_detections is None:
                continue

            if np.array_equal(prev_detections, detections[0, 0, :, 1]):
                continue

            detected = True
            # if we are not already recording, start recording
            if not kcw.recording:
                difference = set(detections[0, 0, :, 1]).symmetric_difference(set(prev_detections))
                for o in difference:
                    print('{} appeared'.format(CLASSES[int(o)]))
                p = "{}/{}.avi".format(args["output"],
                                       timestamp.strftime("%Y%m%d-%H%M%S"))
                print(timestamp, 'Start recording',p)
                kcw.start(p, cv2.VideoWriter_fourcc(*args["codec"]),
                          args["fps"])
    return detected

# construct a child process *indepedent* from our main process of
# execution
print("[INFO] starting process...")
p = Process(target=classify_frame, args=(net, inputQueue,
	                                 outputQueue,))
p.daemon = True
p.start()

# loop over the frames from the video stream
try:
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # update the key frame clip buffer
        kcw.update(frame)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]

        # if the input queue *is* empty, give the current frame to
        # classify
        if inputQueue.empty():
            inputQueue.put(frame)
        # if the output queue *is not* empty, grab the detections
        if not outputQueue.empty():
            detections = outputQueue.get()

        if detections is not None:
            if loop_over_detections(frame, detections, prev_detections):
                consecFrames = 0
            prev_detections = detections[0, 0, :, 1]  # save objects, detected on current frame

        # increment the number of consecutive frames that contain
        # no action
        consecFrames += 1

        # if we are recording and reached a threshold on consecutive
        # number of frames with no action, stop recording the clip
        if kcw.recording and consecFrames == args["buffer_size"]:
            print(datetime.datetime.now(), 'Stop recording')
            kcw.finish()


        if not args['headless']:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            
        # update the FPS counter
        fps.update()
except:
    import traceback
    traceback.print_exc()
    pass

kcw.finish()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
