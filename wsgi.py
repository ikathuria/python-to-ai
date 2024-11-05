from app.utils.helper import CustomLogger
from app.main import flask_app


logger = CustomLogger().get_logger()
flask_app.logger.handlers = logger.handlers
flask_app.logger.setLevel(logger.level)

if __name__ == "__main__":
    try:
        logger.info("Starting application on port 8080")
        flask_app.run(host="0.0.0.0", port=8080)
    except Exception as e:
        logger.error(f"Could not start app because of error: {e}")
        raise
