import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging
import pickle
from flask import url_for


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
        "rb"
    ))


def get_index_from_dictionary(key, value, dictionary):
    """Gets the index for value from a dictionary.

    Args:
        value: value present in dictionary.

    Returns:
        the index if value is present in dictionary.
    """
    return dictionary[key].index(value)


class CustomLogger:
    def __init__(self, name="app_logger", log_level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.configure_console_handler()

        # Suppress noisy loggers
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("tensorflow_lite").setLevel(logging.ERROR)
        logging.getLogger("tensorflow_lite.experimental.XNNPACK").setLevel(logging.ERROR)

    def configure_console_handler(self):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(pathname)s] %(message)s",
            datefmt="%d/%b/%Y %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
