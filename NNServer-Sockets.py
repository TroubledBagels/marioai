import json
import random
import socket
import NNClasses
import sys

possible_buttons = [["A"], ["B"], ["LEFT"], ["RIGHT"], ["LEFT", "A"], ["RIGHT", "A"], ["LEFT", "B"], ["RIGHT", "B"], ["LEFT", "A", "B"], ["RIGHT", "A", "B"]]

datatypes = ["Grounded", "Direction", "HSpeed", "Move_Dir", "Mario X", "Mario Y", "Swimming", "Powerup_State", "Frame Num", "State", "Fallen", "Dead", "Screen"]

sample_data = ['1', '2', '244', '2', '527', '164', '0', '1', '151', '8', '1', 'false', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '0', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0']

def marshall(data):
    data = data.decode("utf-8")
    data = data.strip()
    if "ack" in data:
        return "ack"
    if '0' not in data or '1' not in data:
        return "ack"
    print(data)
    midpoint = []
    blankarr = []
    inScreen = False

    clean = {}
    blank = ""
    for char in data[1:-1]:
        if char == ",":
            if inScreen:
                blankarr.append(blank)
            else:
                midpoint.append(blank)
            blank = ""
        elif char == "[":
            inScreen = True
            blank = ""
        elif char == "]":
            inScreen = False
            blankarr.append(blank)
            blank = ""
            midpoint.append(blankarr)
            blankarr = []
        else:
            blank += str(char)

    for i in range(len(midpoint)):
        clean[datatypes[i]] = midpoint[i]

    return clean


def marshallForNetwork(data):
    clean = []
    for key in data:
        print("Key: " + key)
        if key == "Screen":
            for i in range(len(data[key])):
                clean.append(data[key][i])
        else:
            clean.append(data[key])
    converted = []
    for item in clean:
        if item == "true":
            converted.append(1)
        elif item == "false":
            converted.append(0)
        elif item.isdigit():
            converted.append(int(item))
        elif item[0] == "-":
            converted.append(int(item))
        else:
            converted.append(item)
    return converted

def random_response():
    index = random.randint(0, len(possible_buttons) - 1)
    res = {
        "A": False,
        "B": False,
        "LEFT": False,
        "RIGHT": False
    }
    for button in possible_buttons[index]:
        res[button] = True

    binaryRes = ""
    for key in res:
        binaryRes += str(int(res[key]))
    print(possible_buttons[index])
    print(binaryRes)
    return binaryRes.encode("utf-8")

'''
    Input format:
    index - meaning
        0 - Grounded, 0: grounded, 1: in air from jump, 2: in air from fall, 3: in air from flagpole
        1 - Direction, 0: not on screen, 1: left, 2: right
        2 - HSpeed, float
        3 - Move Direction: 1: right, 2: left
        4 - Player X, int
        5 - Player Y, int
        6 - Swimming, 0: swimming, 1: not swimming
        7 - Powerup State, 0: small, 1: big, 2: fire
        8 - Frame, int (0-255)
        9 - Player State, 0: leftmost, 1: climbing vine, 2: entering pipe, 3: going down pipe, 4 and 5: autowalk, 6: dies, 7: entering area, 8: normal, 9: turning big, 10: turning small, 11: dying, 12: turning fire 
       10 - Fallen off screen, 0: not fallen, >1: fallen
       11 - Dead: 0: not dead, 1: dead - value derived from 10 and 9
'''
def networked_response(data):
    output = thePool.evaluateCurrent(data)
    print(output)

    return output.encode("utf-8")

def read_config(filename):
    file = open(filename, "r")
    data = file.read()
    file.close()
    return json.loads(data)


def initialise_network(filename=""):
    if filename == "":
        config = {
            "mutRates": {
                "connections": 0.5,
                "link": 2.0,
                "bias": 0.3,
                "node": 0.5,
                "enable": 0.2,
                "disable": 0.4,
                "step": 0.1,
                "perturb": 0.9
            },
            "population": 50,
            "deltaDisjoint": 2.0,
            "deltaWeights": 0.4,
            "deltaThreshold": 1.0,
            "staleSpecies": 15,
            "mutChance": 0.5,
            "crossoverChance": 0.75,
            "stepSize": 0.1,
            "timeout": 20,
            "maxNodes": 1000000
        }
    else:
        config = read_config(filename)

    pool = NNClasses.initialisePool(config)
    pool.writeToFile("temp.txt")

    return pool

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
if len(sys.argv) > 1:
    if sys.argv[1] == "init":
        initialise_network("config.json")
        sys.exit(0)
    if sys.argv[1] == "test":
        thePool = initialise_network("config.json")
        print(networked_response(sample_data))
        sys.exit(0)
server.bind(("localhost", 8080))
server.listen(1)
print("Server is listening")
client, address = server.accept()
print(f"Connection from {address} has been established")
thePool = initialise_network()
while True:
    data = client.recv(1024)
    cleaned = marshall(data)
    print(cleaned)
    if cleaned == "ack":
        print(data.decode())
        continue
    if not data:
        continue
    response = networked_response(marshallForNetwork(cleaned))
    print(response)
    client.send(response + b"\n")
