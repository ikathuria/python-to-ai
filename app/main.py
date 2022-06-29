"""set FLASK_ENV=development"""

import os
import pickle
import urllib3
import base64
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp

from app import gpt3content, handges1

from flask import Flask
from flask import request, render_template, url_for, Response, make_response
from flask import redirect, send_from_directory

MODEL_DICT = {
    "Naive Bayes Classifier": "naive_bayes",
    "K Nearest Neighbour": "knn",
    "Logistic Regression": "log_reg",
    "Linear Regression": "lin_reg",
    "Decision Tree": "decision_tree",
    "clustering-kmeans": "kmeans",
    "Hidden Markov Model": "hmm",
    "Principal Component Analysis": "pca",
    "Perceptron": "perceptron",
    "Backpropogation": "backprop",
    "Neural Networks": "neural_nets",
    "Computer Vision": "comp_vision",
    "Natural Language Processing": "nlp",
    "Autoencoder": "ae",
}

global switch
switch = 0


APP = Flask(__name__)


# @APP.errorhandler(404)
# def not_found():
#     """Page not found.
#     """
#     return make_response(
#         render_template("404.html"), 404
#     )


def load_pickle_model(name):
    """Loads the model from the disk.

    Args:
        name: the name of the model.

    Returns:
        the loaded model if available.
    """
    filename = f"models/{name}.sav"
    return pickle.load(open(
        "app" + url_for("static", filename=filename),
        'rb'
    ))


@APP.route("/")
def home():
    error = request.args.get('error_message')
    return render_template(
        'index.html', page='home', error_message=error
    )


# SUPERVISED MACHINE LEARNING -----------------------------------------------------------------------------
@APP.route('/supervised-machine-learning', methods=["GET", "POST"])
def supervised_learning():
    classification_dict = {
        "workclass_dict": [
            'Private', 'Local-gov', '?', 'Self-emp-not-inc', 'Federal-gov',
            'State-gov', 'Self-emp-inc', 'Without-pay', 'Never-worked'
        ],
        "education_dict": [
            '11th', 'HS-grad', 'Assoc-acdm', 'Some-college', '10th', 'Prof-school',
            '7th-8th', 'Bachelors', 'Masters', 'Doctorate', '5th-6th', 'Assoc-voc',
            '9th', '12th', '1st-4th', 'Preschool'
        ],
        "marital_status_dict": [
            'Never-married', 'Married-civ-spouse', 'Widowed', 'Divorced',
            'Separated', 'Married-spouse-absent', 'Married-AF-spouse'
        ],
        "occupation_dict": [
            'Machine-op-inspct', 'Farming-fishing', 'Protective-serv', '?',
            'Other-service', 'Prof-specialty', 'Craft-repair', 'Adm-clerical',
            'Exec-managerial', 'Tech-support', 'Sales', 'Priv-house-serv',
            'Transport-moving', 'Handlers-cleaners', 'Armed-Forces'
        ],
        "relationship_dict": [
            'Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife',
            'Other-relative'
        ],
        "race_dict": [
            'Black', 'White', 'Asian-Pac-Islander',
            'Other', 'Amer-Indian-Eskimo'
        ],
        "gender_dict": ['Male', 'Female'],
        "native_country_dict": [
            'United-States', '?', 'Peru', 'Guatemala', 'Mexico',
            'Dominican-Republic', 'Ireland', 'Germany', 'Philippines', 'Thailand',
            'Haiti', 'El-Salvador', 'Puerto-Rico', 'Vietnam', 'South', 'Columbia',
            'Japan', 'India', 'Cambodia', 'Poland', 'Laos', 'England', 'Cuba',
            'Taiwan', 'Italy', 'Canada', 'Portugal', 'China', 'Nicaragua',
            'Honduras', 'Iran', 'Scotland', 'Jamaica', 'Ecuador', 'Yugoslavia',
            'Hungary', 'Hong', 'Greece', 'Trinadad&Tobago',
            'Outlying-US(Guam-USVI-etc)', 'France', 'Holand-Netherlands'
        ],
        "income_dict": ['<=50K', '>50K']
    }

    regression_dict = {}

    if request.method == 'POST':
        # User's chosen model
        model_type = request.form['model']
        chosen_model = request.form['category']
        algo = MODEL_DICT[chosen_model]
        model = load_pickle_model(algo)

        if model_type == 'classification':
            class_age = int(request.form['class-age'])
            class_workclass = classification_dict["workclass_dict"].index(
                request.form['class-workclass'])
            class_edu = classification_dict["education_dict"].index(
                request.form['class-edu'])
            class_marital = classification_dict["marital_status_dict"].index(
                request.form['class-marital'])
            class_occ = classification_dict["occupation_dict"].index(
                request.form['class-occ'])
            class_rel = classification_dict["relationship_dict"].index(
                request.form['class-rel'])
            class_race = classification_dict["race_dict"].index(
                request.form['class-race'])
            class_gender = classification_dict["gender_dict"].index(
                request.form['class-gender'])
            class_cg = int(request.form['class-cg'])
            class_cl = int(request.form['class-cl'])
            class_hpw = int(request.form['class-hpw'])
            class_country = classification_dict["native_country_dict"].index(
                request.form['class-country'])

            class_input = [[
                class_age, class_workclass, class_edu, class_marital,
                class_occ, class_rel, class_race, class_gender, class_cg,
                class_cl, class_hpw, class_country
            ]]

            try:
                pred = classification_dict["income_dict"][model.predict(class_input)[
                    0]]
            except:
                pred = None

        else:
            pass

        return render_template(
            "supervised_ml.html", page='supervised_ml',
            classification=classification_dict, regression=None,
            inputs=class_input, prediction=pred
        )

    return render_template(
        "supervised_ml.html", page='supervised_ml',
        classification=None, regression=None,
        inputs=None, prediction=None
    )


# UNSUPERVISED MACHINE LEARNING ---------------------------------------------------------------------------
@APP.route('/unsupervised-machine-learning', methods=["GET", "POST"])
def unsupervised_learning():
    cluster_dict = {
        'Species': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    }

    if request.method == 'POST':
        # User's chosen model
        model_type = request.form['model']
        algo = MODEL_DICT[model_type]
        model = load_pickle_model(algo)

        if model_type == 'clustering-kmeans':
            petal_len = float(request.form['cluster-petal-len'])
            petal_wid = float(request.form['cluster-petal-wid'])

            cluster_input = [[petal_len, petal_wid]]
            print(cluster_input)

            try:
                pred = cluster_dict["Species"][model.predict(cluster_input)[
                    0]]
            except:
                pred = None
        else:
            pass

        return render_template(
            "unsupervised_ml.html", page='unsupervised_ml',
            inputs=cluster_input, prediction=pred
        )

    return render_template(
        "unsupervised_ml.html", page='unsupervised_ml',
        clustering=None, inputs=None, prediction=None
    )


# NLP CONTENT GENERATOR ---------------------------------------------------------------------------------
@APP.route('/nlp-content-generator', methods=["GET", "POST"])
def nlp_content_gen():
    if request.method == 'POST':
        prompt = request.form['blogContent']
        blogT = gpt3content.generate_blog(prompt)
        blog = blogT.replace("\n", "<br />")

        return render_template(
            'nlp_content_gen.html', page='nlp_content_gen', blog=blog
        )

    return render_template(
        'nlp_content_gen.html', page='nlp_content_gen', blog=''
    )


# HAND GESTURE RECOGNITION 1 ------------------------------------------------------------------------------
def gen_frames():
    """Generate frame by frame from camera.
    """
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
    labels = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "up", 11: "down", 12: "left", 13: "right", 14: "off",
        15: "on", 16: "ok", 17: "blank"
    }

    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2
    )

    # region of interest (ROI) coordinates
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
                    for l in landmarks:
                        coords_x.append(int(l.x * width))
                        coords_y.append(int(l.y * height))
                    # bounded_hands = get_hands(clone, coords_x, coords_y)
                    # cv2.imshow('Hands', bounded_hands)

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
                    print("\n[STATUS] please wait! calibrating...")
                elif num_frames == 29:
                    print("[STATUS] calibration successfull...")
                    print("Press 'c' to recalibrate background")
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

                    cv2.imwrite('temp_threshold.png', thresholded)

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
                t[:,:,0] = thresholded
                t[:,:,1] = thresholded
                t[:,:,2] = thresholded

                frame = np.hstack((c, t))
                ret, buffer = cv2.imencode(
                    '.jpg', frame
                )
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            except Exception as e:
                if 'thresholded' in str(e):
                    # contents = urllib3.PoolManager().urlopen(
                    #     "https://images.unsplash.com/photo-1623018035782-b269248df916?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80"
                    # ).read()
                    # yield base64.b64encode(contents)
                    pass
                else:
                    c = clone
                    thresholded = cv2.resize(thresholded, (640, 480))
                    t = np.zeros_like(c)
                    t[:,:,0] = thresholded
                    t[:,:,1] = thresholded
                    t[:,:,2] = thresholded

                    frame = np.hstack((c, t))
                    ret, buffer = cv2.imencode(
                        '.jpg', frame
                    )
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        else:
            pass


@APP.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@APP.route('/hand-gesture-recognition-1', methods=["GET", "POST"])
def hand_ges_1():
    global switch, hands, camera, num_frames, hand_ges_model1

    # loading model weights
    hand_ges_model1 = handges1.load_weights(
        "app" + url_for(
            'static', filename='models/10Apr_model_12.h5'
    ))

    if request.method == 'POST':
        if request.form.get('startCamera') == 'startCamera':
            if switch == 1:
                switch = 0
                hands.close()
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1

        elif request.form.get('recallCamera') == 'recallCamera':
            num_frames = 0

        return render_template(
            'hand_gesture_1.html', page='hand_gesture_1', switch=switch
        )

    return render_template(
        'hand_gesture_1.html', page='hand_gesture_1', switch=switch
    )


# ABOUT ----------------------------------------------------------------------------------------------------
@APP.route("/about")
def about():
    return render_template('about.html', page='about')


if __name__ == "__main__":
    APP.run()
