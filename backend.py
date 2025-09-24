from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Inicializar app FastAPI
app = FastAPI()

# Carregar datasets
catalogo = pd.read_csv("catalogo.csv")
avaliacoes = pd.read_csv("avaliacoes.csv")

# Modelo de entrada para recomendação
class UsuarioInput(BaseModel):
    usuario_id: int
    n_recomendacoes: int = 5

def gerar_recomendacoes(usuario_id: int, n_recomendacoes: int = 5):
    # Criar matriz usuário-item
    matriz = avaliacoes.pivot_table(index="usuario_id", columns="item_id", values="nota").fillna(0)

    if usuario_id not in matriz.index:
        return {"erro": "Usuário não encontrado"}

    # Similaridade entre usuários
    similaridade = cosine_similarity(matriz)
    similaridade_df = pd.DataFrame(similaridade, index=matriz.index, columns=matriz.index)

    # Pegar usuários mais parecidos
    similares = similaridade_df[usuario_id].sort_values(ascending=False)
    similares = similares.drop(usuario_id)  # tirar o próprio usuário

    # Itens que o usuário ainda não avaliou
    itens_avaliados = set(avaliacoes[avaliacoes["usuario_id"] == usuario_id]["item_id"])
    candidatos = catalogo[~catalogo["item_id"].isin(itens_avaliados)]

    # Calcular uma pontuação média ponderada pelas similaridades
    recomendacoes = []
    for item in candidatos["item_id"]:
        notas = avaliacoes[avaliacoes["item_id"] == item]
        if notas.empty:
            continue
        soma, peso = 0, 0
        for _, row in notas.iterrows():
            sim = similares.get(row["usuario_id"], 0)
            soma += sim * row["nota"]
            peso += abs(sim)
        if peso > 0:
            score = soma / peso
            recomendacoes.append((item, score))

    # Ordenar pelas melhores pontuações
    recomendacoes = sorted(recomendacoes, key=lambda x: x[1], reverse=True)[:n_recomendacoes]

    # Retornar títulos
    resultado = []
    for item_id, score in recomendacoes:
        filme = catalogo[catalogo["item_id"] == item_id].iloc[0]
        resultado.append({"titulo": filme["titulo"], "categoria": filme["categoria"], "score": round(score, 2)})

    return resultado

@app.post("/recomendar")
def recomendar(dados: UsuarioInput):
    return gerar_recomendacoes(dados.usuario_id, dados.n_recomendacoes)

