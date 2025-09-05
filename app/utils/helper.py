import os
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
    def __init__(self, prefix="[Default]", log_level=logging.DEBUG):
        self.prefix = prefix
        self.logger = logging.getLogger(self.prefix)
        self.logger.setLevel(log_level)
        self.configure_console_handler()

        # Suppress noisy loggers
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        os.environ
        logging.getLogger("tensorflow_lite").setLevel(logging.ERROR)

    def configure_console_handler(self):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            f"[%(asctime)s] [%(levelname)s] {self.prefix} %(message)s",
            datefmt="%d/%b/%Y %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        self.logger.propagate = False
        return self.logger
