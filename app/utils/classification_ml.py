from app.utils.helper import *
import app.utils.constants as c


class GenerateClassificationPredication:
    """Generate prediction for classification models.

    Args:
        algorithm: type of classification model
        form_values: user input from form
    """

    def __init__(self, algorithm, form_values):
        self.logger_prefix = "[Machine Learning][Classification]"

        self.model = load_pickle_model(algorithm)
        self.form_values = form_values

        # generate user input
        try:
            self.input = self.generate_input()
        except Exception as e:
            self.input = None
            print(
                f"{self.logger_prefix} Failed to get input because of error: {e}"
            )
            raise

        # generate prediction
        try:
            self.prediction = self.generate_prediction()
            print(
                f"{self.logger_prefix} Got prediction for {algorithm}: {self.prediction}"
            )
        except Exception as e:
            self.prediction = None
            print(
                f"{self.logger_prefix} Failed to generate prediction because of error: {e}"
            )

    def generate_input(self):
        """Generate input for algoritm.
        """
        age = int(self.form_values["class-age"])
        workclass = get_index_from_dictionary(
            "workclass_dict",
            self.form_values["class-workclass"],
            c.CLASSIFICATION_DICT
        )
        education = get_index_from_dictionary(
            "education_dict",
            self.form_values["class-edu"],
            c.CLASSIFICATION_DICT
        )
        marital = get_index_from_dictionary(
            "marital_status_dict",
            self.form_values["class-marital"],
            c.CLASSIFICATION_DICT
        )
        occ = get_index_from_dictionary(
            "occupation_dict",
            self.form_values["class-occ"],
            c.CLASSIFICATION_DICT
        )
        relationship = get_index_from_dictionary(
            "relationship_dict",
            self.form_values["class-rel"],
            c.CLASSIFICATION_DICT
        )
        race = get_index_from_dictionary(
            "race_dict",
            self.form_values["class-race"],
            c.CLASSIFICATION_DICT
        )
        gender = get_index_from_dictionary(
            "gender_dict",
            self.form_values["class-gender"],
            c.CLASSIFICATION_DICT
        )
        cg = int(self.form_values["class-cg"])
        cl = int(self.form_values["class-cl"])
        hpw = int(self.form_values["class-hpw"])
        country = get_index_from_dictionary(
            "native_country_dict",
            self.form_values["class-country"],
            c.CLASSIFICATION_DICT
        )

        return [[
            age, workclass, education, marital,
            occ, relationship, race, gender, cg,
            cl, hpw, country
        ]]

    def generate_prediction(self):
        """Get prediction based on user input.
        """
        return c.CLASSIFICATION_DICT["income_dict"][self.model.predict(self.input)[0]]
