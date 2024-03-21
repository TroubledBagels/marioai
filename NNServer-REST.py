from flask import Flask, request, jsonify
import json
import random

app = Flask(__name__)

possible_buttons = [["A"], ["B"], ["LEFT"], ["RIGHT"], ["LEFT", "A"], ["RIGHT", "A"], ["LEFT", "B"], ["RIGHT", "B"], ["LEFT", "A", "B"], ["RIGHT", "A", "B"]]

def calculate_response(data):
    index = random.randint(0, len(possible_buttons) - 1)
    res = {
        "A": False,
        "B": False,
        "LEFT": False,
        "RIGHT": False
    }
    for button in possible_buttons[index]:
        res[button] = True
    return res

def marshall(data):
    return json.dumps(data)

@app.post("/receive")
def receive():
    data = request.json
    print(marshall(data))
    response = calculate_response(data)
    print(response)
    return response