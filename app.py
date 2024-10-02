import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Usar el modelo DialoGPT
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "")
    # Generar respuesta
    response = chatbot(user_message, max_length=1000, pad_token_id=50256)
    return jsonify({"response": response[0]['generated_text']})

if __name__ == '__main__':
    app.run(debug=True)