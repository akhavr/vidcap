# https://www.pyimagesearch.com/2016/02/29/saving-key-event-video-clips-with-opencv/
# https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

# import the necessary packages
from pyimagesearch.keyclipwriter import KeyClipWriter
from pyimagesearch.singlemotiondetector import SingleMotionDetector

from imutils.video import VideoStream
from imutils.video import FPS

from flask import Response
from flask import Flask
from flask import render_template
import threading

from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import argparse
import imutils
import datetime
import time
import cv2
import logging

import config

# Telegram notification
import telegram_notifier

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default=config.output,
	        help="path to output directory")
ap.add_argument("-p", "--prototxt", default=config.prototxt,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default=config.model,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=config.confidence,
                help="minimum probability to filter weak detections")
ap.add_argument("-b", "--buffer-size", type=int, default=config.buffer_size,
	        help="buffer size of video clip writer")
ap.add_argument("--codec", type=str, default=config.codec,
	        help="codec of output video")
ap.add_argument("-f", "--fps", type=int, default=config.fps,
	        help="FPS of output video")

ap.add_argument('--headless', dest='headless', action='store_true', help='Run headless')
ap.set_defaults(headless=config.headless)

ap.add_argument('--noblock', dest='block', action='store_false', help='Run detection in parallel')
ap.add_argument('--block', dest='block', action='store_true', help='Run detection in the main thread')
ap.set_defaults(block=config.block)

ap.add_argument('--motion', dest='motion', action='store_true', help='Run motion detection')
ap.add_argument('--object', dest='motion', action='store_false', help='Run object detection')
ap.set_defaults(motion=config.motion)

ap.add_argument('--memory', dest='memory', action='store_true', help='Debug memory leaks')
ap.set_defaults(memory=config.memory)

ap.add_argument("-i", "--ip", type=str, default=config.ip,
		help="ip address of the device")
ap.add_argument("--port", type=int, default=config.port,
		help="ephemeral port number of the server (1024 to 65535)")

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

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

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


def classify_frame(net, frame):
    frame = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame, 0.007843,
                                 (300, 300), 127.5)
    # set the blob as input to our deep learning object
    # detector and obtain the detections
    net.setInput(blob)
    return net.forward()


def loop_classify_frame(net, inputQueue, outputQueue):
    # keep looping
    while True:
        # check to see if there is a frame in our input queue
        if inputQueue.empty():
            continue
        # grab the frame from the input queue, resize it, and
        # construct a blob from it
        frame = inputQueue.get()
        detections = classify_frame(net, frame)
        # write the detections to the output queue
        outputQueue.put(detections)


def loop_over_detections(frame, detections, prev_detections, w, h, notifier=None):
    global url

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
                msg = []
                for o in difference:
                    msg.append(CLASSES[int(o)])
                msg = ', '.join(msg)
                ts = timestamp.strftime("%Y%m%d-%H%M%S")
                msg = '{}: {} appeared'.format(ts, msg)
                if url:
                    msg = msg + ' {}/video_feed'.format(url)
                print(msg)
                if notifier:
                    notifier(msg)
                p = "{}/{}.avi".format(args["output"], ts)
                print(timestamp, 'Start recording',p)
                kcw.start(p, cv2.VideoWriter_fourcc(*args["codec"]),
                          args["fps"])
    return detected

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")


def detect_motion(frameCount=32, notifier=None):
    global vs, outputFrame, lock, kcw
    global consecFrames

    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    if args['memory']:
        from pympler import tracker, muppy
        tr = tracker.SummaryTracker()

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it

        if args['memory'] and total % 1000 == 0:
            tr.print_diff()
            print(len(muppy.get_objects()))

        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # update the key frame clip buffer
        kcw.update(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frameCount:
            # detect motion in the image
            motion = md.detect(gray)
            # check to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                        (0, 0, 255), 2)

                frame = detect_object_in_frame(frame, kcw, notifier)

            # increment the number of consecutive frames that contain
            # no action
            consecFrames += 1

            # if we are recording and reached a threshold on consecutive
            # number of frames with no action, stop recording the clip
            if kcw.recording and consecFrames == args["buffer_size"]:
                print(datetime.datetime.now(), 'Stop recording')
                kcw.finish()


        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1

        if not args['headless']:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()

def detect_object_in_frame(frame, kcw, notifier=None):
    global detections, consecFrames, prev_detections
    global outputFrame, fps

    frame = imutils.resize(frame, width=400)

    # update the key frame clip buffer
    kcw.update(frame)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]

    if args['block']:
        # run detection in current process
        detections = classify_frame(net, frame)
    else:
        # if the input queue *is* empty, give the current frame to
        # classify
        if inputQueue.empty():
            inputQueue.put(frame)

        # if the output queue *is not* empty, grab the detections
        if not outputQueue.empty():
            detections = outputQueue.get()

    if detections is not None:
        if loop_over_detections(frame, detections, prev_detections, w, h, notifier):
            consecFrames = 0
        prev_detections = detections[0, 0, :, 1]  # save objects, detected on current frame

    # acquire the lock, set the output frame, and release the
    # lock
    with lock:
        outputFrame = frame.copy()

    # update the FPS counter
    fps.update()
    return frame

def detect_objects(notifier=None):
    global consecFrames

    # loop over the frames from the video stream
    try:
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()
            frame = detect_object_in_frame(frame, kcw, notifier)

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

    except:
        import traceback
        traceback.print_exc()
        pass


if __name__ == '__main__':
    updater = telegram_notifier.initialize(config.tg_token)

    # construct a child process *indepedent* from our main process of
    # execution
    if not args['block']:
        print("[INFO] starting process...")
        p = Process(target=loop_classify_frame, args=(net, inputQueue,
                                                      outputQueue,))
        p.daemon = True
        p.start()

    if args['motion']:
        t = threading.Thread(target=lambda: detect_motion(notifier=telegram_notifier.notify))
    else:
        t = threading.Thread(target=lambda: detect_objects(notifier=telegram_notifier.notify))
    t.daemon = True
    t.start()

    global url
    url = None
    if args['headless']:
        url = 'http://{}:{}'.format(args["ip"], args["port"])
        # start the flask app
        app.run(host=args["ip"], port=args["port"], debug=True,
                threaded=True, use_reloader=False)
    else:
        t.join()
        pass

    kcw.finish()
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
