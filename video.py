import cv2
import datetime
import numpy as np
import time
import threading
from multiprocessing import Queue

import imutils
from imutils.video import VideoStream
from imutils.video import FPS

from pyimagesearch.keyclipwriter import KeyClipWriter
from pyimagesearch.singlemotiondetector import SingleMotionDetector


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


class GenericDetector:
    def __init__(self, path, headless, buffer_size, codec, fps, net, confidence,
                 frameCount=32, notifier=None, url=None, debugmemory=False, blocking=False):
        # initialize the video stream, allow the camera sensor to warm up,
        # and initialize the FPS counter
        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=0).start()
        # vs = VideoStream(usePiCamera=True).start()
        time.sleep(2.0)
        self.fps = FPS().start()

        # initialize the key clip writer and the motionFrames
        # and consecFrames to track frames without motion
        self.buffer_size = buffer_size
        self.kcw = KeyClipWriter(bufSize=buffer_size)
        self.consecFrames = 0 # number of frames with no motion
        self.prev_detections = None

        # initialize the output frame and a lock used to ensure thread-safe
        # exchanges of the output frames (useful when multiple browsers/tabs
        # are viewing the stream)
        self.outputFrame = None
        self.lock = threading.Lock()

        self.path = path
        self.headless = headless
        self.codec = codec
        self.fps_rate = fps
        self.net = net
        self.confidence = confidence
        self.frameCount = frameCount
        self.notifier = notifier
        self.url = url
        self.debugmemory = debugmemory
        self.blocking = blocking

        if not self.blocking:
            self.inputQueue = Queue(maxsize=1)
            self.outputQueue = Queue(maxsize=1)

    def __del__(self):
        self.kcw.finish()
        # stop the timer and display FPS information
        self.vs.stop()
        self.fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    def loop_over_detections(self, frame, detections, w, h):
        detected = False
        msg = []

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confidence:
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
        if self.prev_detections is None:
            return

        if np.array_equal(self.prev_detections, detections[0, 0, :, 1]):
            return

        detected = True

        # if we are not already recording, start recording
        if not self.kcw.recording:
            difference = set(detections[0, 0, :, 1]).symmetric_difference(set(self.prev_detections))

            for o in difference:
                msg.append(CLASSES[int(o)])
                
            timestamp = datetime.datetime.now()
            ts = timestamp.strftime("%Y%m%d-%H%M%S")
            p = "{}/{}.avi".format(self.path, ts)
            print(timestamp, 'Start recording',p)
            self.kcw.start(p, cv2.VideoWriter_fourcc(*self.codec),
                           self.fps_rate)
            
            if len(msg)>0:
                msg = ', '.join(msg)
                msg = '{}: {} appeared'.format(ts, msg)
                if self.url:
                    msg = msg + ' {}/video_feed'.format(self.url)
                print(msg)
                if self.notifier:
                    self.notifier(msg)
            
        return detected

    def detect_object_in_frame(self, frame):
        frame = imutils.resize(frame, width=400)

        # update the key frame clip buffer
        self.kcw.update(frame)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]

        # run detection in current process
        detections = self.classify_frame(frame)
        
        if detections is not None:
            if self.loop_over_detections(frame, detections, w, h):
                consecFrames = 0
            self.prev_detections = detections[0, 0, :, 1]  # save objects, detected on current frame

        # acquire the lock, set the output frame, and release the lock
        with self.lock:
            self.outputFrame = frame.copy()

        # update the FPS counter
        self.fps.update()
        return frame        

    def _classify_frame(self, frame):
        frame = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(frame, 0.007843,
                                     (300, 300), 127.5)
        # set the blob as input to our deep learning object
        # detector and obtain the detections
        self.net.setInput(blob)
        return self.net.forward()

    def classify_frame(self, frame):
        if self.blocking:
            return self._classify_frame(frame)

        # if the input queue *is* empty, give the current frame to
        # classify
        if self.inputQueue.empty():
            self.inputQueue.put(frame)

        # if the output queue *is not* empty, grab the detections
        if not self.outputQueue.empty():
            return self.outputQueue.get()

    def loop_classify_frame(self):
        assert not self.blocking  # only for non-blocking case
        # keep looping
        while True:
            # check to see if there is a frame in our input queue
            if self.inputQueue.empty():
                continue
            # grab the frame from the input queue, resize it, and
            # construct a blob from it
            frame = self.inputQueue.get()
            detections = self._classify_frame(frame)
            # write the detections to the output queue
            self.outputQueue.put(detections)

    def generate(self):
        "Yield image/jpeg for web serving"
        # grab global references to the output frame and lock variables
        # loop over frames from the output stream
        while True:
            # wait until the lock is acquired
            with self.lock:
                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if self.outputFrame is None:
                    continue
                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", self.outputFrame)
                # ensure the frame was successfully encoded
                if not flag:
                    continue

            # yield the output frame in the byte format
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                  bytearray(encodedImage) + b'\r\n')


class DetectMotion(GenericDetector):
    def detect(self):
        # initialize the motion detector and the total number of frames
        # read thus far
        md = SingleMotionDetector(accumWeight=0.1)
        total = 0

        if self.debugmemory:
            from pympler import tracker, muppy
            tr = tracker.SummaryTracker()

        # loop over frames from the video stream
        while True:
            # read the next frame from the video stream, resize it,
            # convert the frame to grayscale, and blur it

            if self.debugmemory and total % 1000 == 0:
                tr.print_diff()
                print(len(muppy.get_objects()))

            frame = self.vs.read()
            frame = imutils.resize(frame, width=400)

            # update the key frame clip buffer
            self.kcw.update(frame)

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
            if total > self.frameCount:
                # detect motion in the image
                motion = md.detect(gray)
                # check to see if motion was found in the frame
                if motion is not None:
                    # unpack the tuple and draw the box surrounding the
                    # "motion area" on the output frame
                    (thresh, (minX, minY, maxX, maxY)) = motion
                    cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                            (0, 0, 255), 2)

                    frame = self.detect_object_in_frame(frame)

                # increment the number of consecutive frames that contain
                # no action
                self.consecFrames += 1

                # if we are recording and reached a threshold on consecutive
                # number of frames with no action, stop recording the clip
                if self.kcw.recording and self.consecFrames == self.buffer_size:
                    print(datetime.datetime.now(), 'Stop recording')
                    self.kcw.finish()

            # update the background model and increment the total number
            # of frames read thus far
            md.update(gray)
            total += 1

            if not self.headless:
                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            # acquire the lock, set the output frame, and release the
            # lock
            with self.lock:
                outputFrame = frame.copy()


class DetectObject(GenericDetector):
    def detect(self):
        # loop over the frames from the video stream
        try:
            while True:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                frame = self.vs.read()
                frame = self.detect_object_in_frame(frame)

                # increment the number of consecutive frames that contain
                # no action
                self.consecFrames += 1

                # if we are recording and reached a threshold on consecutive
                # number of frames with no action, stop recording the clip
                if self.kcw.recording and self.consecFrames == self.buffer_size:
                    print(datetime.datetime.now(), 'Stop recording')
                    self.kcw.finish()

                if not self.headless:
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


