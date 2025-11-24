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
from sklearn.decomposition import TruncatedSVD
from typing import List, Optional
import uvicorn
# optional Surprise (better collaborative filtering)
try:
    from surprise import Dataset, Reader, SVD as SurpriseSVD
    _HAS_SURPRISE = True
except Exception:
    _HAS_SURPRISE = False

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
# collaborative globals
user_ids = []
user_id_to_idx = {}
user_factors = None
item_factors_collab = None
collab_k = 50

# blending weight between content and collaborative (0..1). Higher => more CF
# running in content-only mode by default: disable collaborative blending
collab_beta = 0.0
collab_method = 'svd'  # kept for reference; CF training will not run when collab_beta == 0.0

# SBERT / semantic embeddings (content-only alternative)
USE_SBERT = False
SBERT_MODEL = 'all-MiniLM-L6-v2'  # small, fast, good quality
# Se True, concatena TF-IDF (dense) com SBERT (dense) e normaliza o vetor final
CONCAT_SBERT = False
# Concatena TF-IDF word + TF-IDF char_wb para tentar capturar sinais distintos
CONCAT_WORD_CHAR_TFIDF = False
WORD_MAX_FEATURES = 4000
CHAR_MAX_FEATURES = 4000
# Parametros TF-IDF ajustáveis (padrões ótimizados encontrados)
TITLE_REPEAT = 1
TFIDF_MAX_FEATURES = 8000
TFIDF_MIN_DF = 1

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
            # separar por vírgula ou pipe ou espaço
            parts = [p.strip() for p in re.split('[,|/\\;]', t) if p.strip()]
            return " ".join(parts)
        except Exception:
            return str(t)

    import re
    texts = ((df["nome"].fillna("") + " ") * TITLE_REPEAT +
             df.get("genre_tokens", pd.Series([""]*len(df))).fillna("") + " " +
             df.get("year_token", pd.Series([""]*len(df))).fillna("") + " " +
             df["tags"].fillna("").apply(_clean_tags) + " " +
             df["descricao"].fillna("")) .astype(str)
    # limpar strings: pode-se adicionar preprocess se desejar
    return texts.tolist()

def fit_vectorizer():
    global tfidf, item_vectors, USE_SBERT
    corpus = build_item_corpus(items_df)
    # Option: use SBERT embeddings; optionally concat TF-IDF + SBERT
    if USE_SBERT:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(SBERT_MODEL)
            # encode returns numpy array (n_items, dim)
            embeddings = model.encode(corpus, show_progress_bar=False, convert_to_numpy=True, batch_size=32)
            emb_norm = sk_normalize(embeddings, norm='l2', axis=1)
            # build TF-IDF as well if we will concat
            if CONCAT_SBERT:
                tfidf = TfidfVectorizer(max_features=8000, stop_words=None, ngram_range=(1,1), sublinear_tf=True, min_df=1)
                X = tfidf.fit_transform(corpus)  # sparse (n_items, n_features)
                try:
                    X_dense = X.toarray()
                except Exception:
                    X_dense = np.asarray(X.todense())
                X_norm = sk_normalize(X_dense, norm='l2', axis=1)
                # concatenar e normalizar vetor final
                concat = np.hstack([X_norm, emb_norm])
                item_vectors = sk_normalize(concat, norm='l2', axis=1)
            else:
                item_vectors = emb_norm
        except Exception:
            # if SBERT fails for any reason, fall back to TF-IDF
            USE_SBERT = False
    if not USE_SBERT:
        # opção: concatenar TF-IDF word + char_wb
        if CONCAT_WORD_CHAR_TFIDF:
            # word-level TF-IDF (unigrams+bigrams)
            tfidf_word = TfidfVectorizer(max_features=WORD_MAX_FEATURES, stop_words=None, analyzer='word', ngram_range=(1,2), sublinear_tf=True, min_df=1)
            Xw = tfidf_word.fit_transform(corpus)
            # char-level TF-IDF (char_wb 3-5)
            tfidf_char = TfidfVectorizer(max_features=CHAR_MAX_FEATURES, stop_words=None, analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True, min_df=1)
            Xc = tfidf_char.fit_transform(corpus)
            try:
                Xw_dense = Xw.toarray()
            except Exception:
                Xw_dense = np.asarray(Xw.todense())
            try:
                Xc_dense = Xc.toarray()
            except Exception:
                Xc_dense = np.asarray(Xc.todense())
            Xw_norm = sk_normalize(Xw_dense, norm='l2', axis=1)
            Xc_norm = sk_normalize(Xc_dense, norm='l2', axis=1)
            concat = np.hstack([Xw_norm, Xc_norm])
            item_vectors = sk_normalize(concat, norm='l2', axis=1)
            # keep last tfidf reference as word tfidf
            tfidf = tfidf_word
        else:
            # usar unigrams e incluir termos raros (min_df configurável)
            tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words=None, analyzer='word', ngram_range=(1,1), sublinear_tf=True, min_df=TFIDF_MIN_DF)
            X = tfidf.fit_transform(corpus)  # sparse (n_items, n_features)
            # Usar vetores TF-IDF normalizados por linha (sem LSA) — manter representações de conteúdo
            try:
                # normalizar cada vetor de item (linha) para norma L2
                item_vectors = sk_normalize(X, norm='l2', axis=1)
            except Exception:
                # se normalização falhar, manter matriz TF-IDF crua
                item_vectors = X
    # compute item popularity (normalized count of positive ratings) to use as a small boost
    global item_popularity
    try:
        pos_counts = eval_df[eval_df["nota"] >= POSITIVE_RATING_THRESHOLD].groupby("item_id").size()
        counts = [pos_counts.get(iid, 0) for iid in item_ids]
        arr = np.array(counts, dtype=float)
        if arr.max() > 0:
            item_popularity = arr / arr.max()
        else:
            item_popularity = np.zeros(len(item_ids), dtype=float)
    except Exception:
        item_popularity = np.zeros(len(item_ids), dtype=float)

    # após construir item_vectors, também ajustar componente colaborativa
    try:
        # somente treinar componente colaborativa se o blend estiver ativo (collab_beta > 0)
        if collab_beta and collab_beta > 0.0:
            fit_collaborative()
    except Exception:
        # se falhar, manter collab desabilitada (None)
        pass


def fit_collaborative(k: int = None):
    """Treina um modelo SVD simples sobre a matriz usuário-item (ratings) e popula
    user_factors e item_factors_collab."""
    global user_ids, user_id_to_idx, user_factors, item_factors_collab, collab_k
    if k is None:
        k = collab_k
    # precisa de eval_df
    if eval_df is None or eval_df.empty:
        user_factors = None
        item_factors_collab = None
        return
    # construir mapeamentos
    user_ids = list(eval_df["usuario_id"].astype(str).unique())
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    n_users = len(user_ids)
    n_items = len(item_ids)
    if n_users == 0 or n_items == 0:
        user_factors = None
        item_factors_collab = None
        return

    # montar matriz densa (n_users x n_items) — ml-100k é pequeno, denso aceitável
    R = np.zeros((n_users, n_items), dtype=float)
    for _, row in eval_df.iterrows():
        u = str(row["usuario_id"])
        iid = str(row["item_id"])
        if u in user_id_to_idx and iid in item_id_to_idx:
            ui = user_id_to_idx[u]
            ii = item_id_to_idx[iid]
            try:
                R[ui, ii] = float(row.get("nota", 0.0))
            except Exception:
                R[ui, ii] = 0.0

    # centrar por usuário (remover média) ajuda SVD
    user_means = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1) + 1e-9)
    R_centered = R - user_means[:, np.newaxis]
    # preencher zeros (não-avaliações) com 0 (já estão) — SVD lidará com isso

    # preferir Surprise SVD quando disponível e configurado
    if collab_method == 'surprise' and _HAS_SURPRISE:
        try:
            # Surprise espera colunas user,item,rating em dataframe
            df = eval_df[["usuario_id", "item_id", "nota"]].copy()
            df["usuario_id"] = df["usuario_id"].astype(str)
            df["item_id"] = df["item_id"].astype(str)
            reader = Reader(rating_scale=(df["nota"].min(), df["nota"].max()))
            data = Dataset.load_from_df(df[["usuario_id", "item_id", "nota"]], reader)
            trainset = data.build_full_trainset()
            n_comp = min(k, trainset.n_users - 1, trainset.n_items - 1) if (trainset.n_users > 1 and trainset.n_items > 1) else 0
            if n_comp <= 0:
                user_factors = None
                item_factors_collab = None
                return
            algo = SurpriseSVD(n_factors=n_comp, random_state=42)
            algo.fit(trainset)
            # build user mapping and factors
            user_ids = [trainset.to_raw_uid(i) for i in range(trainset.n_users)]
            user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
            user_factors = algo.pu  # shape (n_users_train, n_comp)
            # build item_factors aligned to backend.item_ids ordering
            item_factors_collab = np.zeros((len(item_ids), n_comp), dtype=float)
            for inner_i in range(trainset.n_items):
                raw_iid = trainset.to_raw_iid(inner_i)
                if raw_iid in item_id_to_idx:
                    idx = item_id_to_idx[raw_iid]
                    item_factors_collab[idx, :] = algo.qi[inner_i]
            collab_k = n_comp
            return
        except Exception:
            # se Surprise falhar, cair para SVD tradicional
            pass

    n_comp = min(k, n_users - 1, n_items - 1) if (n_users > 1 and n_items > 1) else 0
    if n_comp <= 0:
        user_factors = None
        item_factors_collab = None
        return

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    user_factors = svd.fit_transform(R_centered)  # shape (n_users, n_comp)
    # components_ : shape (n_comp, n_items) -> item_factors: (n_items, n_comp)
    item_factors_collab = svd.components_.T
    collab_k = n_comp

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
            alpha = 0.0  # default: não misturar popularidade (grid mostrou beta alto com alpha baixo é melhor)
            sims = (1 - alpha) * sims + alpha * item_popularity
    except Exception:
        pass

    # componente colaborativa: se usuario_id conhecido e modelo collab treinado
    try:
        if usuario_id is not None and item_factors_collab is not None and user_id_to_idx:
            uid = str(usuario_id)
            if uid in user_id_to_idx:
                uidx = user_id_to_idx[uid]
                # user_factors shape (n_users, k), item_factors_collab shape (n_items, k)
                cf_scores = np.dot(user_factors[uidx], item_factors_collab.T)
                # normalizar CF scores
                if np.ptp(cf_scores) > 0:
                    cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())
                else:
                    cf_scores = np.zeros_like(cf_scores)
                # blend content and collaborative
                beta = collab_beta
                sims = (1 - beta) * sims + beta * cf_scores
    except Exception:
        pass
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
        profile = user_profile_from_item_ids(train)
        if profile is None:
            continue
        # recomendar top K onde K = len(test) * 2 (heurística) ou ao menos 1
        k = max(1, len(test))
        # passar usuario_id para que a parte colaborativa seja usada na recomendação
        recs = recommend_for_profile(profile, top_n=k*2, exclude_ids=train, usuario_id=u)
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

    # construir mapa item->rating para ponderar o perfil pelo quanto o usuário gostou
    ratings_map = {row["item_id"]: row["nota"] for _, row in user_ratings.iterrows()}
    profile = user_profile_from_item_ids(pos_items, ratings_map=ratings_map)
    recs = recommend_for_profile(profile, top_n=req.n, exclude_ids=pos_items, usuario_id=u)
    return {"usuario_id": u, "n": req.n, "recommendations": recs}

@app.get("/avaliacao")
def avaliacao():
    metrics = evaluate_global()
    return metrics

# endpoint para re-treinar vetorizador (útil se alterar filmes.csv)
class RebuildRequest(BaseModel):
    title_repeat: Optional[int] = None
    tfidf_max_features: Optional[int] = None
    tfidf_min_df: Optional[int] = None


@app.post("/rebuild_vectors")
def rebuild_vectors(req: RebuildRequest = None):
    try:
        # permite sobrepor parâmetros TF-IDF rapidamente via body JSON
        global TITLE_REPEAT, TFIDF_MAX_FEATURES, TFIDF_MIN_DF
        if req is not None:
            if req.title_repeat is not None:
                TITLE_REPEAT = int(req.title_repeat)
            if req.tfidf_max_features is not None:
                TFIDF_MAX_FEATURES = int(req.tfidf_max_features)
            if req.tfidf_min_df is not None:
                TFIDF_MIN_DF = int(req.tfidf_min_df)
        load_data()
        fit_vectorizer()
        return {"status": "ok", "n_items": len(item_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
