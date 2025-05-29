import google.generativeai as generativeai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
chave_secreta = os.getenv('API_KEY')
print(chave_secreta)
generativeai.configure(api_key=chave_secreta)

csv_url = 'https://docs.google.com/spreadsheets/d/11QU1ibjUAlNKLwLWF1s-kSpRH2UBOiVbLyl1pJIyeSk/export?format=csv&id=11QU1ibjUAlNKLwLWF1s-kSpRH2UBOiVbLyl1pJIyeSk'
df = pd.read_csv(csv_url)
print(df.head())

model = 'models/gemini-embedding-exp-03-07'
def gerarEmbeddings(title, text):
  result = generativeai.embed_content(model=model,
                                content=text,
                                task_type="retrieval_document",
                                title=title)
  return result['embedding']



df["Embeddings"] = df.apply(lambda row: gerarEmbeddings(row["Titulo"],row["Conteúdo"]), axis=1)
print(df)

import pickle
pickle.dump(df, open('datasetEmbedding2025.pkl','wb'))

modeloEmbeddings = pickle.load(open('datasetEmbedding2025.pkl','rb'))
print(modeloEmbeddings)