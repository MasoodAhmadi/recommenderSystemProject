# app/route.py
import os
from flask import Blueprint,render_template
from app.rating import index
from app.sequential import sequential
from app.diversity import show_diversity
from app.counterfactual import counterfactual

# Create blueprint
main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@main_bp.route("/part1", methods=["GET", "POST"])
def part1():
    return index()


@main_bp.route("/part2", methods=["GET", "POST"])
def part2():
    return sequential()

@main_bp.route("/part3", methods=["GET", "POST"])
def part3():
    return show_diversity()

@main_bp.route("/part4", methods=["GET", "POST"])
def part4():
    return counterfactual()
