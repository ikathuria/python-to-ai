import os

from flask import Flask, abort, jsonify, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from app.config import ProductionConfig, get_config_class

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES_DIR = os.path.join(BASE_DIR, "app", "pages")
ONNX_DIR = os.path.join(BASE_DIR, "app", "onnx_models")


def create_app(config_name: str | None = None) -> Flask:
    config_cls = get_config_class(config_name)
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config_cls)

    if config_cls is ProductionConfig:
        ProductionConfig.validate(app)

    if app.config["USE_PROXY_FIX"]:
        app.wsgi_app = ProxyFix(
            app.wsgi_app,
            x_for=1,
            x_proto=1,
            x_host=1,
            x_prefix=1,
        )

    register_routes(app)
    register_error_handlers(app)
    register_security_headers(app)
    return app


def register_routes(app: Flask) -> None:
    def _send_html(directory: str, name: str):
        if not name.lower().endswith(".html"):
            abort(404)
        return send_from_directory(directory, name)

    @app.route("/")
    def index():
        return send_from_directory(BASE_DIR, "index.html")

    @app.route("/index.html")
    def index_html():
        return send_from_directory(BASE_DIR, "index.html")

    @app.route("/app/pages/<path:filename>")
    def serve_page(filename: str):
        return _send_html(PAGES_DIR, filename)

    @app.route("/app/onnx_models/<path:filename>")
    def serve_onnx(filename: str):
        if not filename.lower().endswith(".onnx"):
            abort(404)
        return send_from_directory(ONNX_DIR, filename)

    @app.route("/healthz")
    def healthz():
        return jsonify(status="ok"), 200

    @app.route("/readyz")
    def readyz():
        return jsonify(status="ready"), 200


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(_e):
        if app.config["TESTING"]:
            return "Not Found", 404
        return (
            "<!DOCTYPE html><html><head><title>Not found</title></head>"
            "<body><h1>Not found</h1><p>The requested page does not exist.</p></body></html>",
            404,
            {"Content-Type": "text/html; charset=utf-8"},
        )

    @app.errorhandler(500)
    def server_error(_e):
        app.logger.exception("Unhandled server error")
        if app.config["TESTING"]:
            return "Internal Server Error", 500
        return (
            "<!DOCTYPE html><html><head><title>Error</title></head>"
            "<body><h1>Something went wrong</h1>"
            "<p>Please try again later.</p></body></html>",
            500,
            {"Content-Type": "text/html; charset=utf-8"},
        )


def register_security_headers(app: Flask) -> None:
    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        response.headers.setdefault(
            "Referrer-Policy", "strict-origin-when-cross-origin"
        )
        response.headers.setdefault(
            "Permissions-Policy",
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()",
        )
        if app.config.get("SESSION_COOKIE_SECURE") and app.config.get(
            "PREFERRED_URL_SCHEME"
        ) == "https":
            response.headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response
