from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/", methods=["GET"])
def test():
    return "This is test get method."


@app.route("/talk", methods=["POST"])
def talk():
    message = request.json.get("result").get("parameters").get("message")
    print(message)
    response = {"response": "response" + message}
    return jsonify(response)


if __name__ == "__main__":
    app.run(port=8000)
