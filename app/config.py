import logging
import os


def _default_config_name() -> str:
    explicit = os.environ.get("FLASK_CONFIG")
    if explicit:
        return explicit.lower()
    # Common PaaS pattern: platform assigns PORT for the web process
    if os.environ.get("PORT"):
        return "production"
    return "development"


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY")
    SEND_FILE_MAX_AGE_DEFAULT = int(os.environ.get("SEND_FILE_MAX_AGE", "3600"))
    USE_PROXY_FIX = os.environ.get("USE_PROXY_FIX", "").lower() in ("1", "true", "yes")
    PREFERRED_URL_SCHEME = os.environ.get("PREFERRED_URL_SCHEME", "http")


class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
    SESSION_COOKIE_SECURE = False
    # Local dev without exporting SECRET_KEY (not for production)
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-only-change-me"


class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PREFERRED_URL_SCHEME = os.environ.get("PREFERRED_URL_SCHEME", "https")

    @staticmethod
    def validate(app):
        if not app.config.get("SECRET_KEY"):
            raise RuntimeError(
                "SECRET_KEY must be set in the environment for production "
                "(use a long random string; all workers must share the same value)."
            )


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    SECRET_KEY = "test-secret-key"
    SEND_FILE_MAX_AGE_DEFAULT = 0


_CONFIG = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config_class(name: str | None):
    key = (name or _default_config_name()).lower()
    if key not in _CONFIG:
        raise ValueError(f"Unknown FLASK_CONFIG / config name: {key!r}")
    return _CONFIG[key]


def log_level_from_env() -> int:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)
