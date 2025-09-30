from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

app = FastAPI(title="Recomendador MovieLens 100K - Cosine")

class RequisicaoRecomendacao(BaseModel):
    usuario_id: int
    n_recomendacoes: int = 5

class RequisicaoAvaliacao(BaseModel):
    usuario_id: int
    movie_id: int
    rating: int

# Aponta para a pasta criada pelo seu novo script de conversão
RATINGS_FILE = os.path.join("converted_data", "ratings.csv")
MOVIES_FILE = os.path.join("converted_data", "movies.csv")

# Ler ratings e filmes
try:
    ratings = pd.read_csv(RATINGS_FILE)
    movies = pd.read_csv(MOVIES_FILE)
except FileNotFoundError:
    raise RuntimeError("ERRO: Ficheiros CSV não encontrados na pasta 'converted_data'. Execute o seu script de conversão primeiro.")

# Renomear coluna 'title' para 'titulo' para manter a consistência com o resto do código
if 'title' in movies.columns:
    movies.rename(columns={'title': 'titulo'}, inplace=True)

# Extrair ano
def extrair_ano(d):
    try:
        if isinstance(d, str):
            # Tenta extrair o ano a partir da data completa (formato YYYY-MM-DD)
            return pd.to_datetime(d).year
    except: pass
    return None

movies["ano"] = movies["release_date"].apply(extrair_ano)

# --- Variáveis Globais para o Modelo (para que possam ser atualizadas) ---
user_item = None
user_matrix = None
user_sim_matrix = None
user_id_to_index = None
index_to_user_id = None

def construir_modelo():
    """Função para construir ou reconstruir o modelo de recomendação em memória."""
    global user_item, user_matrix, user_sim_matrix, user_id_to_index, index_to_user_id, ratings

    print("A construir/atualizar o modelo de recomendação...")
    # Garante que os IDs sejam inteiros para o pivot
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['movie_id'] = ratings['movie_id'].astype(int)
    
    user_item = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
    user_matrix = user_item.values
    user_sim_matrix = cosine_similarity(user_matrix)
    
    user_ids = user_item.index.to_numpy()
    user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    index_to_user_id = {idx: uid for uid, idx in user_id_to_index.items()}
    print("Modelo pronto.")

# Construir o modelo inicial ao arrancar
construir_modelo()


def gerar_recomendacoes(usuario_id: int, n_recomendacoes: int = 5):
    if usuario_id not in user_id_to_index:
        return {"erro": f"Usuário {usuario_id} não encontrado no dataset."}

    uidx = user_id_to_index[usuario_id]
    sims = user_sim_matrix[uidx]
    
    user_ratings = user_matrix[uidx]
    
    numerador = user_matrix.T.dot(sims)
    denominador = np.sum(np.abs(sims)) + 1e-9
    scores_est = numerador / denominador
    
    movie_ids = user_item.columns.to_numpy()
    nao_vistos_mask = (user_ratings == 0)
    candidatos_ids = movie_ids[nao_vistos_mask]
    candidatos_scores = scores_est[nao_vistos_mask]

    idx_sorted = np.argsort(-candidatos_scores)
    top_idx = idx_sorted[:n_recomendacoes]

    resultado = []
    for i in top_idx:
        mid = int(candidatos_ids[i])
        sc = float(candidatos_scores[i])
        row = movies[movies["movie_id"] == mid]
        if not row.empty:
            row = row.iloc[0]
            resultado.append({
                "movie_id": mid,
                "titulo": row["titulo"],
                "score": float(round(sc, 3)),
                "ano": int(row["ano"]) if not pd.isna(row["ano"]) else None
            })
    return resultado

def calcular_acuracia(usuario_id: int, n_recomendacoes: int):
    if usuario_id not in user_id_to_index:
        return {"erro": f"Usuário {usuario_id} não encontrado no dataset."}

    user_ratings_df = ratings[ratings['user_id'] == usuario_id]
    filmes_gostados = user_ratings_df[user_ratings_df['rating'] >= 4]['movie_id'].tolist()
    
    if len(filmes_gostados) < 4:
        return {"erro": "Usuário não tem avaliações positivas suficientes para o teste (mínimo 4 com nota >= 4)."}
    
    np.random.shuffle(filmes_gostados)
    meio = len(filmes_gostados) // 2
    # teste_ids são os filmes que vamos "esconder"
    teste_ids = filmes_gostados[meio:]

    # --- INÍCIO DA CORREÇÃO ---
    # Em vez de remover os filmes de teste de TODO o dataset,
    # criamos uma cópia e removemos as avaliações APENAS do utilizador-alvo.
    
    # 1. Copiar o dataframe de ratings original
    ratings_temp = ratings.copy()
    
    # 2. Criar uma máscara para encontrar as avaliações a serem escondidas
    mask = (ratings_temp['user_id'] == usuario_id) & (ratings_temp['movie_id'].isin(teste_ids))
    
    # 3. Criar o conjunto de treino temporário removendo apenas essas avaliações
    ratings_treino = ratings_temp[~mask]
    # --- FIM DA CORREÇÃO ---

    # Construir um modelo temporário com os dados de treino
    modelo_temp_item = ratings_treino.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
    # Alinhar as colunas com o modelo original para garantir a mesma forma
    modelo_temp_item = modelo_temp_item.reindex(columns=user_item.columns, fill_value=0)
    modelo_temp_sim = cosine_similarity(modelo_temp_item.values)
    
    # Gerar recomendações com o modelo temporário
    uidx = user_id_to_index[usuario_id]
    sims = modelo_temp_sim[uidx]
    
    matriz_temp_valores = modelo_temp_item.values
    numerador = matriz_temp_valores.T.dot(sims)
    denominador = np.sum(np.abs(sims)) + 1e-9
    scores_est = numerador / denominador
    
    # Encontrar filmes que o utilizador não viu NO MODELO TEMPORÁRIO
    nao_vistos_mask = (matriz_temp_valores[uidx] == 0)
    candidatos_ids = modelo_temp_item.columns.to_numpy()[nao_vistos_mask]
    candidatos_scores = scores_est[nao_vistos_mask]
    
    idx_sorted = np.argsort(-candidatos_scores)
    recomendacoes_ids = candidatos_ids[idx_sorted[:n_recomendacoes]]
    
    # Calcular acertos
    acertos = len(set(recomendacoes_ids) & set(teste_ids))
    
    # --- INÍCIO DA CORREÇÃO ---
    # O frontend espera uma chave chamada 'acuracia'.
    # A métrica que corresponde a (acertos / n_recomendacoes) é a Precisão.
    # Vamos calcular e adicionar a chave 'acuracia' à resposta.
    
    # Cálculo da Precisão (o que o frontend chama de acurácia)
    precisao_valor = acertos / n_recomendacoes if n_recomendacoes > 0 else 0
    
    # Cálculo do Recall
    recall_valor = acertos / len(teste_ids) if len(teste_ids) > 0 else 0

    return {
        "acertos": acertos,
        "total_recomendado": n_recomendacoes,
        "total_no_teste": len(teste_ids),
        "acuracia": round(precisao_valor, 3), # Chave adicionada para o frontend
        "precisao": round(precisao_valor, 3), # Mantendo o nome técnico correto
        "recall": round(recall_valor, 3)
    }
    # --- FIM DA CORREÇÃO ---


# ===== Endpoints =====

@app.post("/recomendar")
def recomendar(req: RequisicaoRecomendacao):
    return gerar_recomendacoes(req.usuario_id, req.n_recomendacoes)

# Alterado o nome do endpoint para ser mais descritivo
@app.post("/calcular-acuracia-usuario")
def endpoint_calcular_acuracia(req: RequisicaoRecomendacao):
    # n_recomendacoes aqui é o "k" do Precision@k/Recall@k
    return calcular_acuracia(req.usuario_id, req.n_recomendacoes)

@app.post("/avaliar-filme")
def avaliar_filme(req: RequisicaoAvaliacao):
    global ratings
    nova_avaliacao = pd.DataFrame([{"user_id": req.usuario_id, "movie_id": req.movie_id, "rating": req.rating, "timestamp": int(time.time())}])
    
    # Evitar duplicados se o utilizador reavaliar um filme
    ratings = ratings[~((ratings['user_id'] == req.usuario_id) & (ratings['movie_id'] == req.movie_id))]
    ratings = pd.concat([ratings, nova_avaliacao], ignore_index=True)
    
    # O modelo precisa de ser reconstruído com os novos dados
    construir_modelo()
    return {"status": "sucesso", "mensagem": "Avaliação registada e modelo atualizado."}