import time
import cv2
import numpy as np
import mediapipe as mp

from flask import Flask
from flask import request, render_template, url_for, Response, make_response

from app import gpt3content, handges1
from app.utils.hand_gesture_recognition_1 import HAND_GESTURE_DEMO, generate_frames_for_webapp
import app.utils.helper as helper
import app.utils.constants as c
import app.utils.classification_ml as classification_utils

global switch
switch = 1


flask_app = Flask(__name__)
logger = helper.CustomLogger(prefix="[Flask App]").get_logger()


@flask_app.route("/")
def home():
    logger.info("Rendering home page")
    return render_template(
        "index.html",
        page="home"
    )


# MACHINE LEARNING ---------------------------------------------------------------------------
@flask_app.route("/machine-learning")
def machine_learning():
    logger.debug("Rendering about ML page")
    return render_template(
        "machine_learning/machine_learning.html",
        page="ml"
    )


# SUPERVISED MACHINE LEARNING ---------------------------------------------------------------------------
@flask_app.route("/machine-learning/supervised-learning")
def supervied_machine_learning():
    logger.debug("Rendering about supervised ML page")
    return render_template(
        "machine_learning/machine_learning.html",
        page="supervised_ml"
    )


# UNSUPERVISED MACHINE LEARNING ---------------------------------------------------------------------------
@flask_app.route("/machine-learning/unsupervised-learning")
def unsupervied_machine_learning():
    logger.debug("Rendering about unsupervised ML page")
    return render_template(
        "machine_learning/machine_learning.html",
        page="unsupervised_ml"
    )


# SUPERVISED MACHINE LEARNING MODEL ---------------------------------------------------------------------------
@flask_app.route("/machine-learning/supervised-learning-model", methods=["GET", "POST"])
def supervised_learning_model():
    """Supervised machine learning model.

    Includes classification and regression supervised
    machine learning models.
    """
    logger.debug("Rendering supervised ML demo page")
    ml_input = None
    pred = None

    if request.method == "POST":
        # User"s chosen model
        model_type = request.form["model"]
        chosen_model = request.form["category"]

        if model_type == "classification":
            classification = classification_utils.GenerateClassificationPredication(
                algorithm=c.MODEL_DICT[chosen_model],
                form_values=request.form,
            )
            ml_input = classification.input
            pred = classification.prediction

    return render_template(
        "machine_learning/supervised_model.html",
        page="model_supervised_ml",
        path="/machine-learning/supervised-learning-model",
        classification=c.CLASSIFICATION_DICT,
        regression=c.REGRESSION_DICT,
        inputs=ml_input,
        prediction=pred
    )


# UNSUPERVISED MACHINE LEARNING MODEL ---------------------------------------------------------------------------
@flask_app.route("/machine-learning/unsupervised-learning-model", methods=["GET", "POST"])
def unsupervised_learning_model():
    """Unsupervised machine learning model.

    Includes clustering machine learning models.
    """
    logger.debug("Rendering unsupervised ML demo page")
    # ml_input = None
    pred = None

    cluster_dict = {
        "Species": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    }

    if request.method == "POST":
        # User"s chosen model
        model_type = request.form["model"]
        algo = c.MODEL_DICT[model_type]
        model = helper.load_pickle_model(algo)

        if model_type == "clustering-kmeans":
            petal_len = float(request.form["cluster-petal-len"])
            petal_wid = float(request.form["cluster-petal-wid"])

            cluster_input = [[petal_len, petal_wid]]
            logger.info(
                f"[Unsupervised ML] [Clustering] [K-means] Received user input: {cluster_input}"
            )

            try:
                pred = cluster_dict["Species"][model.predict(cluster_input)[
                    0]]
            except Exception as e:
                logger.error(
                    f"[Unsupervised ML] [Clustering] [K-means] Could not get prediction due to error: {e}"
                )
                pred = None
        else:
            pass

        return render_template(
            "machine_learning/unsupervised_model.html", page="model_unsupervised_ml",
            path="/machine-learning/unsupervised-learning-model",
            inputs=cluster_input, prediction=pred
        )

    return render_template(
        "machine_learning/unsupervised_model.html", page="model_unsupervised_ml",
        path="/machine-learning/unsupervised-learning-model",
        clustering=None, inputs=None, prediction=None
    )


# NLP CONTENT GENERATOR ---------------------------------------------------------------------------
@flask_app.route("/nlp-content-generator", methods=["GET", "POST"])
def nlp_content_gen():
    logger.debug("Rendering NLP demo page")
    if request.method == "POST":
        prompt = request.form["blogContent"]
        blogT = gpt3content.generate_blog(prompt)
        blog = blogT.replace("\n", "<br />")

        return render_template(
            "deep_learning/nlp_content_gen.html", page="nlp_content_gen", blog=blog
        )

    return render_template(
        "deep_learning/nlp_content_gen.html", page="nlp_content_gen", blog=""
    )


# HAND GESTURE RECOGNITION 1 ---------------------------------------------------------------------------
@flask_app.route("/video_feed")
def video_feed():
    logger.debug("Rendering video feed page")
    return Response(
        generate_frames_for_webapp(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@flask_app.route("/hand-gesture-recognition-1", methods=["GET", "POST"])
def hand_ges_1():
    global switch, HAND_GESTURE_DEMO

    logger.debug("Rendering hand gesture 1 demo page")
    if request.method == "POST":
        if request.form.get("startCamera") == "startCamera":
            switch = 1
            HAND_GESTURE_DEMO.switch = 1
            HAND_GESTURE_DEMO.start_video_feed()

        elif request.form.get("stopCamera") == "stopCamera":
            switch = 0
            HAND_GESTURE_DEMO.switch = 0
            HAND_GESTURE_DEMO.stop_video_feed()

        elif request.form.get("recallCamera") == "recallCamera":
            switch = 1
            HAND_GESTURE_DEMO.switch = 1
            HAND_GESTURE_DEMO.num_frames = 0

        time.sleep(10)

    return render_template(
        "deep_learning/hand_gesture_1.html",
        page="hand_gesture_1", 
        switch=switch
    )


# ABOUT ---------------------------------------------------------------------------
@flask_app.route("/about")
def about():
    logger.debug("Rendering about page")
    return render_template("about.html", page="about")
