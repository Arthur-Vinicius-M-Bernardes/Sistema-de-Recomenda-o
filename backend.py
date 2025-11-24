# backend.py
# FastAPI backend: filtragem baseada em conteúdo usando TF-IDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize as sk_normalize
from typing import List, Optional
import uvicorn
# (Suporte a Surprise/colaborativo removido: modo somente conteúdo)

app = FastAPI(title="Recomendador - Filtragem por Conteúdo")

# Paths dos CSVs (ajuste se necessário)
ITEMS_CSV = "filmes.csv"
EVAL_CSV = "aval.csv"  # arquivo de avaliações (usuario_id,item_id,nota)

# Configs
POSITIVE_RATING_THRESHOLD = 4  # nota >= 4 -> "gostou"
TEST_SIZE_PER_USER = 0.2  # fração para teste nas métricas


# --- carga de dados e preparação ---
items_df = pd.DataFrame()
eval_df = pd.DataFrame()
item_ids = []
item_id_to_idx = {}
idx_to_item_id = {}
tfidf = None
item_vectors = None
item_popularity = None
# Variáveis de filtragem colaborativa e SBERT removidas (não utilizadas)
# Concatena TF-IDF word + TF-IDF char_wb para tentar capturar sinais distintos
CONCAT_WORD_CHAR_TFIDF = True  # habilitado para melhorar cobertura lexical
USE_STOPWORDS = True  # remover stopwords básicas para reduzir ruído
WORD_MAX_FEATURES = 6000
CHAR_MAX_FEATURES = 4000
# Pesos por campo para reforçar sinais mais úteis
TITLE_REPEAT = 3
GENRE_REPEAT = 2
TAGS_REPEAT = 1
DESC_REPEAT = 1
YEAR_REPEAT = 1
# Parametros TF-IDF ajustáveis (padrões mais robustos)
TFIDF_MAX_FEATURES = 10000
TFIDF_MIN_DF = 1  # ajustado (tuning) para capturar termos raros e melhorar F1 – ver seção de tuning no README
WORD_NGRAM = (1, 2)
POP_ALPHA = 0.05  # mistura de popularidade no ranking

def load_data():
    global items_df, eval_df, item_ids, item_id_to_idx, idx_to_item_id
    # Carrega itens (usa latin-1 para suportar acentuação do dataset)
    # muitos arquivos do MovieLens (u.item) usam '|' como separador e não têm header
    # definir nomes de colunas compatíveis com u.item
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"]
    genre_cols = ["unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    all_cols = cols + genre_cols
    items_df = pd.read_csv(ITEMS_CSV, dtype=str, encoding='latin-1', sep='|', engine='python', header=None, names=all_cols).fillna("")
    # Se as colunas de gênero existem (flags 0/1), transformar em uma coluna 'categoria' legível
    try:
        genre_names = genre_cols
        # os valores nas colunas podem ser '0'/'1' ou 0/1
        def _genres_from_row(r):
            parts = []
            for g in genre_names:
                if g in items_df.columns:
                    val = str(r.get(g, "")).strip()
                    if val in ("1", "True", "true", "T"):
                        parts.append(g)
            return ", ".join(parts)
        items_df["categoria"] = items_df.apply(_genres_from_row, axis=1)
    except Exception:
        pass
    # tenta detectar as colunas common (item_id,nome,descricao,tags,categoria)
    # normalizar nomes:
    # Aceita colunas como: item_id, id, nome, title, descricao, description, tags, genero, categoria
    colmap = {}
    cols = [c.lower() for c in items_df.columns]
    for c in items_df.columns:
        lc = c.lower()
        if lc in ("item_id", "id"):
            colmap[c] = "item_id"
        elif lc in ("nome", "title", "name"):
            colmap[c] = "nome"
        elif "desc" in lc:
            colmap[c] = "descricao"
        elif "tag" in lc:
            colmap[c] = "tags"
        elif lc in ("categoria", "genero", "genre"):
            colmap[c] = "categoria"
    items_df = items_df.rename(columns=colmap)
    if "item_id" not in items_df.columns:
        # tenta criar item_id a partir do index
        items_df["item_id"] = items_df.index.astype(str)
    # Coloca colunas ausentes
    for required in ("nome", "descricao", "tags", "categoria"):
        if required not in items_df.columns:
            items_df[required] = ""

    # criar representação tokenizada de gêneros: 'Action, Drama' -> 'genre_Action genre_Drama'
    def _genre_tokens(s):
        try:
            if not isinstance(s, str):
                return ""
            parts = [p.strip() for p in s.split(',') if p.strip()]
            tokens = [f"genre_{p.replace(' ', '_')}" for p in parts]
            return " ".join(tokens)
        except Exception:
            return ""

    items_df["genre_tokens"] = items_df["categoria"].astype(str).apply(_genre_tokens)

    # Normalizar títulos: 'Matrix, The' -> 'The Matrix'
    def _normalize_title(t):
        try:
            if not isinstance(t, str):
                return t
            t = t.strip()
            # pattern: 'Something, The' or 'Something, A' or 'Something, An'
            import re
            m = re.match(r"^(.*),\s*(The|A|An)$", t, flags=re.IGNORECASE)
            if m:
                article = m.group(2).strip()
                body = m.group(1).strip()
                return f"{article} {body}"
            return t
        except Exception:
            return t

    items_df["nome"] = items_df["nome"].astype(str).apply(_normalize_title)

    # extrair ano de release se possível e criar token year_YYYY
    def _extract_year(rd):
        try:
            if not isinstance(rd, str):
                return ""
            rd = rd.strip()
            if len(rd) >= 4:
                import re
                m = re.search(r"(19|20)\d{2}", rd)
                if m:
                    return f"year_{m.group(0)}"
            return ""
        except Exception:
            return ""

    items_df["year_token"] = items_df["release_date"].astype(str).apply(_extract_year)

    item_ids = list(items_df["item_id"].astype(str))
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    idx_to_item_id = {i: iid for iid, i in item_id_to_idx.items()}

    # Carrega avaliações
    try:
        # avaliações: formato u.data (user\titem\trating\ttimestamp) — ler com sep='\t'
        eval_df = pd.read_csv(
            EVAL_CSV,
            sep='\t',
            names=["usuario_id", "item_id", "nota", "timestamp"],
            encoding='latin-1',
            engine='python'
        )
    except Exception:
        # se não existir, cria um df vazio com colunas esperadas
        eval_df = pd.DataFrame(columns=["usuario_id", "item_id", "nota"])

    # normaliza tipos
    eval_df["usuario_id"] = eval_df["usuario_id"].astype(str)
    eval_df["item_id"] = eval_df["item_id"].astype(str)

load_data()

def build_item_corpus(df: pd.DataFrame):
    # Concatena campos relevantes em um único texto para TF-IDF
    # aumentar peso do título repetindo-o algumas vezes para reforçar sua importância
    # incluir genre_tokens (tokens prefixados) para dar peso claro a gênero
    # limpeza básica de campos
    def _clean_tags(t):
        try:
            if not isinstance(t, str):
                return ""
            parts = [p.strip().lower() for p in re.split('[,|/\\;]', t) if p.strip()]
            # remover duplicados mantendo ordem
            seen = set()
            dedup = []
            for p in parts:
                if p not in seen:
                    seen.add(p)
                    dedup.append(p)
            return " ".join(dedup)
        except Exception:
            return str(t).lower()

    import re
    # constrói texto com pesos por campo via repetição simples (robusta para TF-IDF)
    # normalização simples (lowercase)
    title_part = (df["nome"].fillna("").str.lower() + " ").astype(str)
    genre_part = df.get("genre_tokens", pd.Series([""]*len(df))).fillna("").str.lower().astype(str) + " "
    year_part = df.get("year_token", pd.Series([""]*len(df))).fillna("").str.lower().astype(str) + " "
    tags_part = df["tags"].fillna("").apply(_clean_tags).astype(str) + " "
    desc_part = df["descricao"].fillna("").str.lower().astype(str)

    def repeat_text(s: pd.Series, times: int):
        if times <= 1:
            return s
        return (s * times)

    texts = (
        repeat_text(title_part, TITLE_REPEAT) +
        repeat_text(genre_part, GENRE_REPEAT) +
        repeat_text(year_part, YEAR_REPEAT) +
        repeat_text(tags_part, TAGS_REPEAT) +
        repeat_text(desc_part, DESC_REPEAT)
    ).astype(str)
    # limpar strings: pode-se adicionar preprocess se desejar
    return texts.tolist()

def fit_vectorizer():
    # Vetorização somente conteúdo (TF-IDF) + popularidade
    global tfidf, item_vectors, item_popularity
    corpus = build_item_corpus(items_df)
    stop_words = None
    if USE_STOPWORDS:
        stop_words = ['a','o','os','as','de','do','da','dos','das','e','é','em','para','por','um','uma','no','na','nos','nas','com','se','que','the','of','and','in','to','for','on','at','by','an','is','it']
    if CONCAT_WORD_CHAR_TFIDF:
        tfidf_word = TfidfVectorizer(max_features=WORD_MAX_FEATURES, stop_words=stop_words, analyzer='word', ngram_range=WORD_NGRAM, sublinear_tf=True, min_df=TFIDF_MIN_DF)
        Xw = tfidf_word.fit_transform(corpus)
        tfidf_char = TfidfVectorizer(max_features=CHAR_MAX_FEATURES, analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True, min_df=TFIDF_MIN_DF)
        Xc = tfidf_char.fit_transform(corpus)
        from scipy.sparse import hstack
        Xw_norm = sk_normalize(Xw, norm='l2', axis=1)
        Xc_norm = sk_normalize(Xc, norm='l2', axis=1)
        item_vectors = sk_normalize(hstack([Xw_norm, Xc_norm]), norm='l2', axis=1)
        tfidf = tfidf_word
    else:
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words=stop_words, analyzer='word', ngram_range=WORD_NGRAM, sublinear_tf=True, min_df=TFIDF_MIN_DF)
        X = tfidf.fit_transform(corpus)
        try:
            item_vectors = sk_normalize(X, norm='l2', axis=1)
        except Exception:
            item_vectors = X
    try:
        pos_counts = eval_df[eval_df['nota'] >= POSITIVE_RATING_THRESHOLD].groupby('item_id').size()
        counts = [pos_counts.get(iid, 0) for iid in item_ids]
        arr = np.array(counts, dtype=float)
        item_popularity = arr / arr.max() if arr.max() > 0 else np.zeros(len(item_ids))
    except Exception:
        item_popularity = np.zeros(len(item_ids), dtype=float)


 # (Suporte a filtragem colaborativa removido)

fit_vectorizer()


# --- perfil do usuário e recomendação ---
def user_profile_from_item_ids(item_ids_list: List[str], ratings_map: Optional[dict] = None):
    """
    Constrói o perfil do usuário a partir de uma lista de item_ids.
    Se ratings_map for fornecido (dicionário item_id -> rating), os vetores dos itens
    serão ponderados pela nota correspondente antes de somar.
    """
    idxs = [item_id_to_idx[iid] for iid in item_ids_list if iid in item_id_to_idx]
    if not idxs:
        return None
    vectors = item_vectors[idxs]
    try:
        # aplicar pesos quando ratings_map fornecido
        if ratings_map:
            # construir vetor de pesos na ordem dos idxs
            weights = [float(ratings_map.get(backend_id, 1.0)) if ratings_map.get(backend_id, None) is not None else 1.0 for backend_id in [idx_to_item_id[i] for i in idxs]]
            w = np.array(weights, dtype=float)
            # tratar vetores densos (numpy) e esparsos
            if isinstance(vectors, np.ndarray):
                # vectors shape (n_items, n_features)
                weighted = (vectors.T * w).T
                summed = weighted.sum(axis=0)
            else:
                dense = vectors.toarray()
                weighted = (dense.T * w).T
                summed = weighted.sum(axis=0)
        else:
            if isinstance(vectors, np.ndarray):
                summed = vectors.sum(axis=0)
            else:
                summed = vectors.sum(axis=0)

        # ensure ndarray 2D
        if hasattr(summed, "toarray"):
            arr = np.asarray(summed.toarray()).reshape(1, -1)
        else:
            arr = np.asarray(summed).reshape(1, -1)
        arr = sk_normalize(arr, norm='l2')
        return arr
    except Exception:
        # fallback: média sem pesos
        profile = vectors.mean(axis=0)
        if hasattr(profile, "toarray"):
            return np.asarray(profile.toarray())
        return np.asarray(profile)

def recommend_for_profile(profile_vector, top_n=10, exclude_ids: Optional[List[str]] = None, usuario_id: Optional[str] = None):
    # profile_vector: (1, n_features)
    if profile_vector is None:
        return []
    # sklearn does not accept np.matrix; convert to ndarray
    try:
        if hasattr(profile_vector, "toarray"):
            pv = profile_vector.toarray()
        else:
            pv = np.asarray(profile_vector)
        # ensure 2D shape (1, n_features)
        if pv.ndim == 1:
            pv = pv.reshape(1, -1)
    except Exception:
        pv = np.asarray(profile_vector)

    sims = cosine_similarity(pv, item_vectors).flatten()  # len = n_items
    # combine similarity with item popularity to slightly favor well-liked items
    try:
        if item_popularity is not None:
            sims = (1 - POP_ALPHA) * sims + POP_ALPHA * item_popularity
    except Exception:
        pass

    # (Sem blending colaborativo)
    # monta DataFrame temporário
    df = pd.DataFrame({
        "item_id": [idx_to_item_id[i] for i in range(len(sims))],
        "score": sims
    })
    if exclude_ids:
        df = df[~df["item_id"].isin(exclude_ids)]
    df = df.sort_values("score", ascending=False).head(top_n)
    # inclui nome
    merged = df.merge(items_df, how="left", left_on="item_id", right_on="item_id")
    results = merged[["item_id", "nome", "score"]].to_dict(orient="records")
    return results

# --- avaliação (Precision/Recall/F1) ---
def evaluate_global():
    # Para cada usuário, separa itens "positivos" (nota >= threshold) em train/test,
    # cria perfil com train e recomenda top-k (k = len(test) or fixed) e calcula métricas.
    if eval_df.empty:
        return {"precision": None, "recall": None, "f1": None, "users_evaluated": 0}
    users = eval_df["usuario_id"].unique()
    precisions = []
    recalls = []
    f1s = []
    evaluated_users = 0
    for u in users:
        user_ratings = eval_df[eval_df["usuario_id"] == u]
        # itens positivos
        pos = user_ratings[user_ratings["nota"] >= POSITIVE_RATING_THRESHOLD]["item_id"].astype(str).unique().tolist()
        if len(pos) < 2:  # precisa de pelo menos 2 para fazer split train/test
            continue
        train, test = train_test_split(pos, test_size=TEST_SIZE_PER_USER, random_state=42)

        # incorporar feedback negativo (todas as notas abaixo do threshold)
        neg_df = user_ratings[user_ratings["nota"] < POSITIVE_RATING_THRESHOLD]
        neg_ids = neg_df["item_id"].astype(str).unique().tolist()

        # construir mapa de pesos: positivos (apenas treino) recebem peso positivo; negativos recebem peso negativo
        ratings_map = {}
        # pesos positivos normalizados (>= threshold)
        for _, row in user_ratings.iterrows():
            iid = str(row["item_id"])
            try:
                r = float(row.get("nota", 0.0))
            except Exception:
                r = 0.0
            if iid in train and r >= POSITIVE_RATING_THRESHOLD:
                # normaliza em [0.2, 1.0]
                base = POSITIVE_RATING_THRESHOLD - 0.5
                num = max(0.0, r - base)
                den = max(0.5, 5.0 - base)
                w = 0.2 + 0.8 * (num / den)
                ratings_map[iid] = max(ratings_map.get(iid, 0.0), w)
            elif iid in neg_ids:
                # peso negativo proporcional à distância para o limiar
                base = POSITIVE_RATING_THRESHOLD - 0.5
                num = max(0.0, base - r)
                den = max(0.5, base - 1.0)
                scale = (num / den) if den > 0 else 0.0
                w = -0.6 * min(1.0, scale)  # até -0.6
                # acumular o menor (mais negativo)
                ratings_map[iid] = min(ratings_map.get(iid, 0.0), w)

        # construir perfil com itens de treino + negativos
        profile_items = list(ratings_map.keys())
        profile = user_profile_from_item_ids(profile_items, ratings_map=ratings_map)
        if profile is None:
            continue
        # recomendar top K onde K = len(test) * 2 (heurística) ou ao menos 1
        k = max(1, len(test))
        # excluir itens de treino e todos os negativos; permitir itens de teste aparecerem
        exclude = set(train) | set(neg_ids)
        # passar usuario_id para que a parte colaborativa seja usada na recomendação
        recs = recommend_for_profile(profile, top_n=k*2, exclude_ids=list(exclude), usuario_id=u)
        rec_ids = [r["item_id"] for r in recs]
        # calcula métricas simples
        hits = len(set(rec_ids).intersection(set(test)))
        precision = hits / len(rec_ids) if rec_ids else 0.0
        recall = hits / len(test) if test else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        evaluated_users += 1

    if evaluated_users == 0:
        return {"precision": None, "recall": None, "f1": None, "users_evaluated": 0}
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "users_evaluated": int(evaluated_users)
    }

# --- Pydantic models ---
class RecommendRequest(BaseModel):
    usuario_id: Optional[str] = None
    n: Optional[int] = 10
    use_favorites: Optional[bool] = False
    favorite_item_ids: Optional[List[str]] = None  # se use_favorites=True, usa essa lista

# --- Endpoints ---
@app.get("/itens")
def get_itens():
    # Retorna os itens básicos
    out = items_df[["item_id", "nome", "categoria", "tags", "descricao"]].to_dict(orient="records")
    return {"count": len(out), "itens": out}

@app.get("/usuarios")
def get_usuarios():
    if eval_df.empty:
        return {"count": 0, "usuarios": []}
    user_stats = eval_df.groupby("usuario_id").agg({'item_id': 'count'}).reset_index().rename(columns={'item_id': 'n_avaliacoes'})
    return {"count": len(user_stats), "usuarios": user_stats.to_dict(orient="records")}

@app.post("/recomendar")
def recomendar(req: RecommendRequest):
    # se use_favorites está True, constrói perfil a partir de favorite_item_ids
    if req.use_favorites:
        favs = req.favorite_item_ids or []
        profile = user_profile_from_item_ids(favs)
        recs = recommend_for_profile(profile, top_n=req.n, exclude_ids=favs)
        return {"usuario_id": req.usuario_id, "n": req.n, "recommendations": recs}
    # caso contrário, tenta montar perfil a partir de avaliacoes do usuario
    if not req.usuario_id:
        raise HTTPException(status_code=400, detail="usuario_id obrigatório quando use_favorites=False")
    u = str(req.usuario_id)
    if eval_df.empty or u not in eval_df["usuario_id"].astype(str).unique():
        raise HTTPException(status_code=404, detail=f"usuario_id {u} não encontrado nas avaliações")
    user_ratings = eval_df[eval_df["usuario_id"].astype(str) == u]
    pos_items = user_ratings[user_ratings["nota"] >= POSITIVE_RATING_THRESHOLD]["item_id"].astype(str).unique().tolist()
    if not pos_items:
        # sem itens positivos: retorna itens populares (por nota média) como fallback
        # fallback: itens mais avaliados positivamente globalmente
        if eval_df.empty:
            fallback = items_df.head(req.n)[["item_id", "nome"]].to_dict(orient="records")
            return {"usuario_id": u, "n": req.n, "recommendations": [{"item_id": r["item_id"], "nome": r["nome"], "score": None} for r in fallback]}
        pos_global = eval_df[eval_df["nota"] >= POSITIVE_RATING_THRESHOLD].groupby("item_id").size().sort_values(ascending=False).head(req.n).index.tolist()
        recs = [{"item_id": iid, "nome": items_df[items_df["item_id"] == iid]["nome"].values[0], "score": None} for iid in pos_global]
        return {"usuario_id": u, "n": req.n, "recommendations": recs}

    # construir mapa item->peso com positivos e negativos
    neg_df = user_ratings[user_ratings["nota"] < POSITIVE_RATING_THRESHOLD]
    neg_ids = neg_df["item_id"].astype(str).unique().tolist()

    ratings_map = {}
    for _, row in user_ratings.iterrows():
        iid = str(row["item_id"])
        try:
            r = float(row.get("nota", 0.0))
        except Exception:
            r = 0.0
        if r >= POSITIVE_RATING_THRESHOLD:
            base = POSITIVE_RATING_THRESHOLD - 0.5
            num = max(0.0, r - base)
            den = max(0.5, 5.0 - base)
            w = 0.2 + 0.8 * (num / den)
            ratings_map[iid] = max(ratings_map.get(iid, 0.0), w)
        else:
            base = POSITIVE_RATING_THRESHOLD - 0.5
            num = max(0.0, base - r)
            den = max(0.5, base - 1.0)
            scale = (num / den) if den > 0 else 0.0
            w = -0.6 * min(1.0, scale)
            ratings_map[iid] = min(ratings_map.get(iid, 0.0), w)

    profile_items = list(ratings_map.keys())
    profile = user_profile_from_item_ids(profile_items, ratings_map=ratings_map)
    exclude = set(pos_items) | set(neg_ids)
    recs = recommend_for_profile(profile, top_n=req.n, exclude_ids=list(exclude), usuario_id=u)
    return {"usuario_id": u, "n": req.n, "recommendations": recs}

@app.get("/avaliacao")
def avaliacao():
    metrics = evaluate_global()
    return metrics

# endpoint para re-treinar vetorizador (útil se alterar filmes.csv)
class RebuildRequest(BaseModel):
    title_repeat: Optional[int] = None
    genre_repeat: Optional[int] = None
    tags_repeat: Optional[int] = None
    desc_repeat: Optional[int] = None
    year_repeat: Optional[int] = None
    tfidf_max_features: Optional[int] = None
    tfidf_min_df: Optional[int] = None
    word_ngram_high: Optional[int] = None  # 1 ou 2
    concat_word_char_tfidf: Optional[bool] = None
    word_max_features: Optional[int] = None
    char_max_features: Optional[int] = None
    pop_alpha: Optional[float] = None


@app.post("/rebuild_vectors")
def rebuild_vectors(req: RebuildRequest = None):
    try:
        # permite sobrepor parâmetros TF-IDF rapidamente via body JSON
        global TITLE_REPEAT, GENRE_REPEAT, TAGS_REPEAT, DESC_REPEAT, YEAR_REPEAT
        global TFIDF_MAX_FEATURES, TFIDF_MIN_DF, WORD_NGRAM, CONCAT_WORD_CHAR_TFIDF
        global WORD_MAX_FEATURES, CHAR_MAX_FEATURES, POP_ALPHA
    # (Parâmetros SBERT removidos)
        if req is not None:
            if req.title_repeat is not None:
                TITLE_REPEAT = int(req.title_repeat)
            if req.genre_repeat is not None:
                GENRE_REPEAT = int(req.genre_repeat)
            if req.tags_repeat is not None:
                TAGS_REPEAT = int(req.tags_repeat)
            if req.desc_repeat is not None:
                DESC_REPEAT = int(req.desc_repeat)
            if req.year_repeat is not None:
                YEAR_REPEAT = int(req.year_repeat)
            if req.tfidf_max_features is not None:
                TFIDF_MAX_FEATURES = int(req.tfidf_max_features)
            if req.tfidf_min_df is not None:
                TFIDF_MIN_DF = int(req.tfidf_min_df)
            if req.word_ngram_high is not None:
                n = int(req.word_ngram_high)
                WORD_NGRAM = (1, max(1, min(3, n)))
            if req.concat_word_char_tfidf is not None:
                CONCAT_WORD_CHAR_TFIDF = bool(req.concat_word_char_tfidf)
            if req.word_max_features is not None:
                WORD_MAX_FEATURES = int(req.word_max_features)
            if req.char_max_features is not None:
                CHAR_MAX_FEATURES = int(req.char_max_features)
            if req.pop_alpha is not None:
                POP_ALPHA = float(req.pop_alpha)
        load_data()
        fit_vectorizer()
        return {"status": "ok", "n_items": len(item_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
