import os
from flask import Blueprint,render_template,url_for

main_bp = Blueprint("main", __name__)

@main_bp.route('/')
def index():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'app','data', 'ml-100k', 'u.data')
    file_path = os.path.abspath(file_path)    
    with open(file_path, "r") as file:
        row_count = sum(1 for _ in file)
    return render_template("index.html", row_count=row_count)    # return "Hello, World!"

