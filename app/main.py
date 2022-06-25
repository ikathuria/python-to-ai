"""set FLASK_ENV=development"""

from dataclasses import replace
import os
import pandas as pd
import pickle

from app import gpt3content

from flask import Flask
from flask import request, render_template, redirect, send_from_directory, url_for

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


def page_not_found(e):
    return render_template('404.html'), 404


APP = Flask(__name__)
APP.register_error_handler(404, page_not_found)


def load_pickle_model(name):
    """Loads the model from the disk."""
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


@APP.route("/about")
def about():
    return render_template('about.html', page='about')


if __name__ == "__main__":
    APP.run()
