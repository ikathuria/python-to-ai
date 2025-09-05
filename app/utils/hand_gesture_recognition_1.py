"""Visualizing the results with OpenCV.

Functions:
    1. find_biggest_contour
    2. run_avg
    3. segment
    4. detect_hands
    5. get_hands
    6. load_weights
    7. get_predicted_class
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

from flask import url_for

import app.utils.helper as helper


class HandGestureRecognition:
    def __init__(self, model_path):
        """Hand gesture recognition demo.
        """
        self.logger = helper.CustomLogger(
            prefix="[Deep Learning] [Hand Gesture Recognition]"
        ).get_logger()
        self.logger.info("Setting up")

        # switch for video feed
        self.switch = 0

        # loading model weights
        self.model = self.load_weights(model_path)

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2
        )

        self.bg = None

    def start_video_feed(self, local=False):
        """Setup and start camera feed.
        """

        self.logger.info("Switching on camera")

        # accumulated weight
        accumWeight = 0.5

        # region of interest (ROI) coordinates
        top, right, bottom, left = 10, 310, 310, 610

        self.switch = 1

        # initializing number of frames
        self.num_frames = 0

        self.camera = cv2.VideoCapture(0)
        while self.switch:
            success, self.frame = self.camera.read()
            if success:
                # flip the frame so that it is not the mirror view
                self.frame = cv2.flip(self.frame, 1)
                self.clone = self.frame.copy()
                height, width, channels = self.clone.shape
                self.clone.flags.writeable = False

                results = self.hands.process(self.clone)
                self.clone.flags.writeable = True

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            self.clone,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        landmarks = hand_landmarks.landmark

                        coords_x = []
                        coords_y = []
                        for mark in landmarks:
                            coords_x.append(int(mark.x * width))
                            coords_y.append(int(mark.y * height))

                        # bounded_hands = self.get_hands(
                        #     self.clone,
                        #     coords_x,
                        #     coords_y
                        # )
                        # cv2.imshow('Hands', bounded_hands)

                # get the ROI
                roi = self.frame[top:bottom, right:left]

                # convert the roi to grayscale and blur it
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)

                # to get the background, keep looking till a threshold is reached
                # so that our weighted average model gets calibrated
                if self.num_frames < 30:
                    self.run_avg(gray, accumWeight)
                    if self.num_frames == 1:
                        self.logger.info("Starting calibration")
                    elif self.num_frames == 29:
                        self.logger.info("Calibration successful")
                else:
                    # segment the hand region
                    self.hand = self.segment(gray)

                    if self.hand is not None:
                        (self.thresholded, segmented) = self.hand

                        contours, hierarchy = cv2.findContours(
                            self.thresholded,
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_NONE
                        )

                        hull = []
                        for i in range(len(contours)):
                            hull.append(cv2.convexHull(contours[i], False))

                        for i in range(len(contours)):
                            # green - color for contours
                            color_contours = (0, 255, 0)
                            # blue - color for convex hull
                            color = (255, 0, 0)
                            # draw ith contour
                            cv2.drawContours(
                                image=self.thresholded,
                                contours=contours,
                                contourIdx=i,
                                color=color_contours,
                                thickness=1,
                                lineType=8,
                                hierarchy=hierarchy
                            )
                            # draw ith convex hull object
                            cv2.drawContours(
                                image=self.thresholded,
                                contours=hull,
                                contourIdx=i,
                                color=color,
                                thickness=1,
                                lineType=8,
                            )

                        cv2.imwrite('temp_threshold.png', self.thresholded)

                        predictedClass = self.get_predicted_class(
                            self.model
                        )

                        cv2.putText(
                            self.clone,
                            str(predictedClass),
                            (70, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2
                        )

                        if local:
                            cv2.imshow("Thesholded", self.thresholded)

                    else:
                        cv2.putText(
                            self.clone,
                            "BLANK",
                            (70, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2
                        )

                cv2.rectangle(
                    self.clone,
                    (left, top),
                    (right, bottom),
                    (0, 0, 0), 2
                )

                self.num_frames += 1

                if local:
                    cv2.imshow("Gesture Recognition", self.clone)

                    keypress = cv2.waitKey(1) & 0xFF
                    if keypress == ord("q"):
                        self.switch = 0
                        break
                    elif keypress == ord("c"):
                        self.num_frames = 0

            else:
                self.logger.error(f"Camera failed to start due to error")
                self.switch = 0
                break

    def stop_video_feed(self):
        """Stop camera feed.
        """
        self.logger.info("Switching off camera")
        try:
            self.switch = 0
            self.hands.close()
            self.camera.release()
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"Failed to stop video due to error: {e}")

    def find_biggest_contour(self, image):
        """Find biggest contour.

        Args:
            image: the image to be processed.

        Returns:
            biggest_contour: the biggest contour.
        """
        contours, _ = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        biggest = np.array([])
        max_area = 0

        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area

        return biggest

    def run_avg(self, image, aWeight):
        """Set real-time background.

        Args:
            image: the background image.
            aWeight: accumulated weight.
        """
        # initialize the background
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.bg, aWeight)

    def segment(self, image, threshold=25):
        """Segment the image.

        Args:
            image: the image to be segmented.
            threshold: the threshold value, 25 by default.
        """
        # find the absolute difference between background and current frame
        diff = cv2.absdiff(self.bg.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        self.thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        # get the contours in the thresholded image
        (cnts, _) = cv2.findContours(
            self.thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)
            return (self.thresholded, segmented)

    def detect_hands(self, image, draw_image):
        """Return hand part of image.

        Args:
            image: the image to be processed.
            draw_image: image to display mediapipe skeleton.

        Returns:
            roi: image of hand.
        """
        height, width, channels = image.shape

        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    draw_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                landmarks = hand_landmarks.landmark
                coords_x = []
                coords_y = []
                for mark in landmarks:
                    coords_x.append(int(mark.x * width))
                    coords_y.append(int(mark.y * height))
            # bounded_hands = get_hands(image, coords_x, coords_y)
            return draw_image
        return None

    def get_hands(self, image, x, y):
        """Return hand part of image.

        Args:
            image: the image to be processed.
            x: x coordinates.
            y: y coordinates.

        Returns:
            image: image of hand.
        """
        minx = min(x)
        miny = min(y)
        maxx = max(x)
        maxy = max(y)
        cv2.rectangle(image, (minx, miny), (maxx, maxy), (255, 0, 0), 2)
        return image

    def load_weights(self, latest_model):
        """Load Model Weights.

        Returns:
            the loaded model if available, otherwise None.
        """
        model = None
        try:
            model = tf.keras.models.load_model(latest_model)

        except Exception as e:
            self.logger.error(f"Model failed to load due to error: {e}")

        return model

    def get_predicted_class(self, model):
        """Get the predicted class.

        Args:
            model: the loaded model.

        Returns:
            the predicted class.
        """
        # labels in order of training output
        labels = {
            0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
            5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
            10: "up", 11: "down", 12: "left", 13: "right", 14: "off",
            15: "on", 16: "ok", 17: "blank"
        }

        image = cv2.imread('temp_threshold.png')

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (100, 100))
        gray_image = gray_image.reshape(1, 100, 100, 1)

        prediction = model.predict_on_batch(gray_image)
        predicted_class = np.argmax(prediction)

        return labels[predicted_class].upper()


def generate_frames_for_webapp():
    global HAND_GESTURE_DEMO
    HAND_GESTURE_DEMO.logger.info("Generating frames for webapp")

    while True:
        c = HAND_GESTURE_DEMO.clone
        thresholded = cv2.resize(HAND_GESTURE_DEMO.thresholded, (640, 480))
        t = np.zeros_like(c)
        t[:, :, 0] = thresholded
        t[:, :, 1] = thresholded
        t[:, :, 2] = thresholded

        byte_frame = np.hstack((c, t))
        ret, buffer = cv2.imencode(
            ".jpg", byte_frame
        )
        byte_frame = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + byte_frame + b"\r\n"
        )


MODEL_PATH = "app/static/models/10Apr_model_12.h5"
HAND_GESTURE_DEMO = HandGestureRecognition(MODEL_PATH)

if __name__ == "__main__":
    model_path = "app/static/models/10Apr_model_12.h5"
    hand_gesture_demo = HandGestureRecognition(MODEL_PATH)
    hand_gesture_demo.switch = 1
    hand_gesture_demo.start_video_feed(local=True)
