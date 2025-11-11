
import os
import pandas as pd
from flask import Blueprint
from app.rating import index  # import your logic function
from app.sequential import sequential  # import your logic function


# Base directory for dataset
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "data", "smallest-100k"))
data_file = os.path.join(base_dir, "ratings.csv")


# Load dataset globally
df = pd.read_csv(data_file, sep='\t', names=['user_id','item_id','rating','timestamp'])
df = df.sort_values(['user_id','timestamp'])


# Create a blueprint
main_bp = Blueprint("main", __name__)

# Define the /part1 route here
@main_bp.route("/part1", methods=["GET", "POST"])
def part1():
    return index()


@main_bp.route("/part2", methods=["GET", "POST"])
def part2():
    return sequential(df)
