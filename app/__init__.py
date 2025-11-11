from flask import Flask
from app.route import main_bp

def create_app():
    app = Flask(__name__)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.register_blueprint(main_bp)
    return app
