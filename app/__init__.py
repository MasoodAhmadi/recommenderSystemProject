from flask import Flask

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

from app import rating
from app.rating import main_bp
app.register_blueprint(main_bp)


if __name__ == '__main__':
    app.run(debug=True)