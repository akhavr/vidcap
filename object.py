# https://www.pyimagesearch.com/2016/02/29/saving-key-event-video-clips-with-opencv/
# https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

# import the necessary packages

from multiprocessing import Process
import argparse
import threading

import config

# Telegram notification
import telegram_notifier

# Video handling
import cv2
import video

# Enable logging

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
ap.add_argument('--head', dest='headless', action='store_false', help='Run headless')
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


if __name__ == '__main__':
    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    notifier = None
    try:
        notifier = telegram_notifier.Notifier(config.tg_token)
        notifier = notifier.notify
    except ValueError:
        pass

    if args['motion']:
        detector = video.DetectMotion(path=args["output"],
                                      headless=args['headless'],
                                      buffer_size=args['buffer_size'],
                                      codec=args['codec'],
                                      fps=args['fps'],
                                      net=net,
                                      confidence=args["confidence"],
                                      notifier=notifier,
                                      url=config.url,
                                      debugmemory=args['memory'],
                                      blocking=args['block'])
    else:
        detector = video.DetectObject(path=args["output"],
                                      headless=args['headless'],
                                      buffer_size=args['buffer_size'],
                                      codec=args['codec'],
                                      fps=args['fps'],
                                      net=net,
                                      confidence=args["confidence"],
                                      notifier=notifier,
                                      url=config.url,
                                      debugmemory=args['memory'],
                                      blocking=args['block'])

    # construct a child process *indepedent* from our main process of
    # execution
    if not args['block']:
        print("[INFO] starting object detection process...")
        p = Process(target=detector.loop_classify_frame)
        p.daemon = True
        p.start()

    t = threading.Thread(target=lambda: detector.detect())
    t.daemon = True
    t.start()

    if args['headless']:
        import web
        app = web.create_flask(detector)
        url = config.url
        # start the flask app
        app.run(host=args["ip"], port=args["port"],
                debug=True,
                threaded=True,
                use_reloader=False)
    else:
        t.join()
        pass

    # do a bit of cleanup
    cv2.destroyAllWindows()
