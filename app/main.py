import cv2
import numpy as np
import mediapipe as mp

from app import gpt3content, handges1
import app.utils.helper as helper
import app.utils.constants as c
import app.utils.classification_ml as classification_utils

from flask import Flask
from flask import request, render_template, url_for, Response, make_response

global switch
switch = 0


flask_app = Flask(__name__)
logger = helper.CustomLogger().get_logger()


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
def gen_frames():
    """Generate frame by frame from camera.
    """
    logger.debug("Rendering hand gesture demo page")
    global hands, num_frames, hand_ges_model1

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # mp hands
    hands = mp_hands.Hands(
        max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.4
    )

    # accumulated weight
    accumWeight = 0.5

    # labels in order of training output
    # labels = {
    #     0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    #     5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    #     10: "up", 11: "down", 12: "left", 13: "right", 14: "off",
    #     15: "on", 16: "ok", 17: "blank"
    # }

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )

    # ROI coordinates
    top, right, bottom, left = 10, 310, 310, 610

    # initializing number of frames
    num_frames = 0

    while True:
        success, frame = camera.read()
        if success:
            # flip the frame so that it is not the mirror view
            frame = cv2.flip(frame, 1)
            clone = frame.copy()
            height, width, channels = clone.shape

            clone.flags.writeable = False

            results = hands.process(clone)
            clone.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        clone, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark
                    coords_x = []
                    coords_y = []
                    for mark in landmarks:
                        coords_x.append(int(mark.x * width))
                        coords_y.append(int(mark.y * height))
                    # bounded_hands = get_hands(clone, coords_x, coords_y)
                    # cv2.imshow("Hands", bounded_hands)

            # get the ROI
            roi = frame[top:bottom, right:left]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # to get the background, keep looking till a threshold is reached
            # so that our weighted average model gets calibrated
            if num_frames < 30:
                handges1.run_avg(gray, accumWeight)
                if num_frames == 1:
                    logger.info(
                        "[Deep Learning] [Hand Gesture Recognition] Starting frame calibration"
                    )
                elif num_frames == 29:
                    logger.info(
                        "[Deep Learning] [Hand Gesture Recognition] Calibration successful"
                    )

            else:
                # segment the hand region
                hand = handges1.segment(gray)

                if hand is not None:
                    (thresholded, segmented) = hand

                    contours, hierarchy = cv2.findContours(
                        thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    hull = []
                    for i in range(len(contours)):
                        hull.append(cv2.convexHull(contours[i], False))

                    for i in range(len(contours)):
                        # green - color for contours
                        color_contours = (0, 255, 0)
                        color = (255, 0, 0)  # blue - color for convex hull
                        # draw ith contour
                        cv2.drawContours(thresholded, contours, i,
                                         color_contours, 1, 8, hierarchy)
                        # draw ith convex hull object
                        cv2.drawContours(thresholded, hull, i, color, 1, 8)

                    cv2.imwrite("temp_threshold.png", thresholded)

                    predictedClass = handges1.get_predicted_class(
                        hand_ges_model1
                    )

                    cv2.putText(clone, str(predictedClass), (70, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # cv2.imshow("Thesholded", thresholded)

                else:
                    cv2.putText(clone, "BLANK", (70, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(clone, (left, top), (right, bottom), (0, 0, 0), 2)

            # cv2.imshow("Gesture Recognition", clone)

            num_frames += 1

            try:
                c = clone
                thresholded = cv2.resize(thresholded, (640, 480))
                t = np.zeros_like(c)
                t[:, :, 0] = thresholded
                t[:, :, 1] = thresholded
                t[:, :, 2] = thresholded

                frame = np.hstack((c, t))
                ret, buffer = cv2.imencode(
                    ".jpg", frame
                )
                frame = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

            except Exception as e:
                if "thresholded" in str(e):
                    # contents = urllib3.PoolManager().urlopen(
                    #     "https://images.unsplash.com/photo-1623018035782-b269248df916?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80"
                    # ).read()
                    # yield base64.b64encode(contents)
                    pass
                else:
                    c = clone
                    thresholded = cv2.resize(thresholded, (640, 480))
                    t = np.zeros_like(c)
                    t[:, :, 0] = thresholded
                    t[:, :, 1] = thresholded
                    t[:, :, 2] = thresholded

                    frame = np.hstack((c, t))
                    ret, buffer = cv2.imencode(
                        ".jpg", frame
                    )
                    frame = buffer.tobytes()
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        else:
            pass


@flask_app.route("/video_feed")
def video_feed():
    logger.debug("Rendering video feed page")
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@flask_app.route("/hand-gesture-recognition-1", methods=["GET", "POST"])
def hand_ges_1():
    logger.debug("Rendering hand gesture 1 demo page")
    global switch, hands, camera, num_frames, hand_ges_model1

    # loading model weights
    hand_ges_model1 = handges1.load_weights(
        "app" + url_for(
            "static", filename="models/10Apr_model_12.h5"
        ))

    if request.method == "POST":
        if request.form.get("startCamera") == "startCamera":
            if switch == 1:
                switch = 0
                hands.close()
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1

        elif request.form.get("recallCamera") == "recallCamera":
            num_frames = 0

        return render_template(
            "deep_learning/hand_gesture_1.html", page="hand_gesture_1", switch=switch
        )

    return render_template(
        "deep_learning/hand_gesture_1.html", page="hand_gesture_1", switch=switch
    )


# ABOUT ---------------------------------------------------------------------------
@flask_app.route("/about")
def about():
    logger.debug("Rendering about page")
    return render_template("about.html", page="about")
