import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from app.config import log_level_from_env
from app.main import create_app
from app.utils.helper import CustomLogger


application = create_app()
logger = CustomLogger(level=log_level_from_env()).get_logger()
application.logger.handlers = logger.handlers
application.logger.setLevel(logger.level)

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", os.environ.get("PORT", "8080")))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app = create_app("development") if debug else application

    try:
        if debug:
            logger.info("Starting Flask development server on %s:%s", host, port)
            app.run(debug=True, host=host, port=port, use_reloader=False)
        else:
            from waitress import serve

            logger.info("Starting Waitress on %s:%s", host, port)
            serve(app, host=host, port=port)
    except Exception as e:
        logger.error("Could not start app due to error: %s", e)
        raise
