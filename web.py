from flask import Response
from flask import Flask
from flask import render_template


def create_flask(detector):
    # initialize a flask object
    app = Flask(__name__)

    @app.route("/")
    def index():
        # return the rendered template
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        # return the response generated along with the specific media
        # type (mime type)
        return Response(detector.generate(),
                        mimetype = "multipart/x-mixed-replace; boundary=frame")

    return app

