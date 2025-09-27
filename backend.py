from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Recomendador MovieLens 100K - Cosine")

# ===== Modelo de entrada =====
class RequisicaoRecomendacao(BaseModel):
    usuario_id: int
    n_recomendacoes: int = 5

# ===== Carregar dados MovieLens 100k (arquivos dentro da pasta ml-100k) =====
# u.data: user_id \t movie_id \t rating \t timestamp
# u.item: movie_id | movie title | release date | video release date | IMDb URL | genres...
RATINGS_FILE = "ml-100k/u.data"
MOVIES_FILE = "ml-100k/u.item"

# Ler ratings
ratings = pd.read_csv(
    RATINGS_FILE,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
    usecols=["user_id", "movie_id", "rating"],
    engine="python"
)

# Ler filmes (u.item tem encoding latin-1)
movies_cols = ["movie_id", "titulo", "release_date", "video_release_date", "IMDb_URL"]
movies = pd.read_csv(
    MOVIES_FILE,
    sep="|",
    names=movies_cols + [f"g{i}" for i in range(19)],
    encoding="latin-1",
    engine="python"
)[movies_cols]

# Normalizar tipos
movies["movie_id"] = movies["movie_id"].astype(int)
ratings["movie_id"] = ratings["movie_id"].astype(int)
ratings["user_id"] = ratings["user_id"].astype(int)

# Extrair ano (quando disponível) do release_date
def extrair_ano(d):
    try:
        if isinstance(d, str) and len(d) >= 4:
            return int(d.strip()[-4:])
    except:
        pass
    return None

movies["ano"] = movies["release_date"].apply(extrair_ano)

# Criar pivot user-item (users x movies)
user_item = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)

# Precomputar similaridade entre usuários (cosine)
user_ids = user_item.index.to_numpy()
user_matrix = user_item.values
user_sim_matrix = cosine_similarity(user_matrix)

# Helper: mapping user_id -> index na matrix
user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
index_to_user_id = {idx: uid for uid, idx in user_id_to_index.items()}

def gerar_recomendacoes(usuario_id: int, n_recomendacoes: int = 5):
    if usuario_id not in user_id_to_index:
        return {"erro": f"Usuário {usuario_id} não encontrado no dataset."}

    uidx = user_id_to_index[usuario_id]
    sims = user_sim_matrix[uidx]
    sims[uidx] = 0.0

    user_ratings = user_matrix[uidx]

    sim_weights = sims.reshape(1, -1)
    ratings_matrix = user_matrix.T
    numerador = ratings_matrix.dot(sims)
    denominador = np.sum(np.abs(sims)) + 1e-9

    scores_est = numerador / denominador
    movie_ids = user_item.columns.to_numpy()

    nao_vistos_mask = (user_ratings == 0)
    candidatos_ids = movie_ids[nao_vistos_mask]
    candidatos_scores = scores_est[nao_vistos_mask]

    if len(candidatos_ids) == 0:
        media_filmes = ratings.groupby("movie_id")["rating"].mean().sort_values(ascending=False)
        top = media_filmes.head(n_recomendacoes).index.tolist()
        resultado = []
        for mid in top:
            row = movies[movies["movie_id"] == mid].iloc[0]
            resultado.append({
                "movie_id": int(mid),
                "titulo": row["titulo"],
                "score": float(round(media_filmes.loc[mid], 2)),
                "ano": int(row["ano"]) if not pd.isna(row["ano"]) else None
            })
        return resultado

    idx_sorted = np.argsort(-candidatos_scores)
    top_idx = idx_sorted[:n_recomendacoes]

    resultado = []
    for i in top_idx:
        mid = int(candidatos_ids[i])
        sc = float(candidatos_scores[i])
        row = movies[movies["movie_id"] == mid]
        if not row.empty:
            row = row.iloc[0]
            titulo = row["titulo"]
            ano = int(row["ano"]) if not pd.isna(row["ano"]) else None
        else:
            titulo = str(mid)
            ano = None
        resultado.append({
            "movie_id": mid,
            "titulo": titulo,
            "score": float(round(sc, 3)),
            "ano": ano
        })

    return resultado

# --- INÍCIO DAS NOVAS FUNÇÕES DE ACURÁCIA ---

def gerar_recomendacoes_para_avaliacao(usuario_id: int, n_recomendacoes: int, treino_ids: list):
    """Gera recomendações com base em um histórico parcial de um usuário para avaliação."""
    uidx = user_id_to_index[usuario_id]
    sims = user_sim_matrix[uidx].copy() # Usar cópia para não alterar o original
    sims[uidx] = 0.0

    user_ratings_treino = np.zeros(user_item.shape[1])
    for movie_id in treino_ids:
        rating_original = user_item.loc[usuario_id, movie_id]
        col_idx = user_item.columns.get_loc(movie_id)
        user_ratings_treino[col_idx] = rating_original

    numerador = user_matrix.T.dot(sims)
    denominador = np.sum(np.abs(sims)) + 1e-9
    scores_est = numerador / denominador

    movie_ids = user_item.columns.to_numpy()
    nao_vistos_mask = (user_ratings_treino == 0)
    candidatos_ids = movie_ids[nao_vistos_mask]
    candidatos_scores = scores_est[nao_vistos_mask]

    idx_sorted = np.argsort(-candidatos_scores)
    top_ids = candidatos_ids[idx_sorted[:n_recomendacoes]]
    return top_ids.tolist()

def calcular_acuracia(usuario_id: int, n_recomendacoes: int):
    """Calcula a acurácia do modelo para um usuário específico."""
    user_ratings_df = ratings[ratings['user_id'] == usuario_id]
    filmes_gostados = user_ratings_df[user_ratings_df['rating'] >= 4]['movie_id'].tolist()

    if len(filmes_gostados) < 4:
        return {"erro": "Usuário não possui avaliações positivas suficientes (mínimo 4) para o teste."}

    np.random.shuffle(filmes_gostados)
    meio = len(filmes_gostados) // 2
    treino_ids = filmes_gostados[:meio]
    teste_ids = filmes_gostados[meio:]

    recomendacoes_ids = gerar_recomendacoes_para_avaliacao(usuario_id, n_recomendacoes, treino_ids)

    acertos = len(set(recomendacoes_ids) & set(teste_ids))
    acuracia_calc = acertos / n_recomendacoes if n_recomendacoes > 0 else 0

    return {
        "acertos": acertos,
        "total_recomendado": n_recomendacoes,
        "acuracia": round(acuracia_calc, 2),
    }

# ===== Endpoints =====

@app.post("/recomendar")
def recomendar(req: RequisicaoRecomendacao):
    try:
        recomendacoes = gerar_recomendacoes(req.usuario_id, req.n_recomendacoes)
        return recomendacoes
    except Exception as e:
        return {"erro": f"Erro ao gerar recomendações: {str(e)}"}

@app.post("/avaliar")
def avaliar(req: RequisicaoRecomendacao):
    """Novo endpoint para calcular a acurácia."""
    try:
        resultado = calcular_acuracia(req.usuario_id, req.n_recomendacoes)
        return resultado
    except Exception as e:
        return {"erro": f"Erro ao calcular acurácia: {str(e)}"}
