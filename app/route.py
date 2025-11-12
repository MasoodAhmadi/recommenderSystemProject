# app/route.py
import os
from flask import Blueprint
from app.rating import index
from app.sequential import sequential

# Create blueprint
main_bp = Blueprint("main", __name__)

@main_bp.route("/part1", methods=["GET", "POST"])
def part1():
    return index()


@main_bp.route("/part2", methods=["GET", "POST"])
def part2():
    return sequential()
