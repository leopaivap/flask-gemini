from flask import Flask, jsonify, request
import numpy as np
import google.generativeai as generativeai
from google import genai
from google.genai import types
import pickle
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
CORS(app)

model = 'models/gemini-embedding-exp-03-07'
modeloEmbeddings = pickle.load(open('datasetEmbedding2025.pkl','rb'))
chave_secreta = os.getenv('API_KEY')
generativeai.configure(api_key=chave_secreta)

def gerarBuscarConsulta(consulta, dataset):
    embedding_consulta = generativeai.embed_content(
        model=model,
        content=consulta,
        task_type="retrieval_query",
    )

    produtos_escalares = np.dot(
        np.stack(dataset["Embeddings"]),
        embedding_consulta['embedding']
    )  # Cálculo de distância entre consulta e a base

    print(embedding_consulta)
    print(produtos_escalares)

    indice = np.argmax(produtos_escalares)
    print(produtos_escalares[indice])

    return dataset.iloc[indice]['Conteúdo']


def melhorarResposta(inputText):
    client = genai.Client(api_key=chave_secreta)
    model = "gemini-1.5-flash"

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=inputText),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_k=32,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""Considere a consulta e resposta, reescreva as sentenças de resposta de uma forma alternativa, não apresente opções de reescrita"""),
        ],
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return response.text

@app.route("/")
def home():
    consulta = "Quem é você ?"
    resposta = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Consulta: {consulta} Resposta: {resposta}"
    response = melhorarResposta(prompt)
    return response

@app.route("/api", methods=["POST"])
def results():
    # Verifique a chave de autorização
    auth_key = request.headers.get("Authorization")
    if auth_key != chave_secreta:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json(force=True)
    consulta = data["consulta"]
    resultado = gerarBuscarConsulta(consulta, modeloEmbeddings)
    prompt = f"Consulta: {consulta} Resposta: {resultado}"
    response = melhorarResposta(prompt)
    return jsonify({"mensagem": response})


