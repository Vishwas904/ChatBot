from flask import Flask, jsonify, request
from chatbot import getAnswer
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/api/chat', methods=['POST'])
def generateAnswer():
    try:
        data_json = request.get_json()
        if 'prompt' in data_json:
            answer = getAnswer(data_json['prompt'])
            return jsonify({"answer": answer}), 200
        else:
            return jsonify({"error": "Missing 'prompt' in JSON data"}), 400
    except Exception as e:
        error_message = str(e)
        return jsonify({"error": error_message}), 500


if __name__ == '__main__':
    app.run()
