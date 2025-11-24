# frontend.py
# Streamlit frontend que consome o backend FastAPI implementado acima.
import streamlit as st
import requests
import pandas as pd
import os
import json
import html

# --- TMDB configuration (used to fetch poster, overview and genres) ---
TMDB_API_KEY = "5a0a89104a4fc273fb664827c8682454"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_GENRE_URL = "https://api.themoviedb.org/3/genre/movie/list"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://via.placeholder.com/300x450?text=Sem+Imagem"


def normalize_title(title: str) -> str:
    """Move trailing articles like ", The" to the front: 'Matrix, The' -> 'The Matrix'."""
    if not title or not isinstance(title, str):
        return title
    parts = [", the", ", a", ", an", ", the".upper(), ", a".upper(), ", an".upper()]
    # common patterns: ", The" or ", the" etc.
    for art in [", The", ", A", ", An", ", the", ", a", ", an"]:
        if title.endswith(art):
            body = title[:-len(art)].strip()
            article = art.strip().strip(',')
            return f"{article} {body}"
    return title

@st.cache_data(show_spinner=False)
def get_genre_map():
    if not TMDB_API_KEY:
        return {}
    try:
        resp = requests.get(TMDB_GENRE_URL, params={"api_key": TMDB_API_KEY, "language": "pt-BR"}, timeout=5)
        if resp.status_code == 200:
            genres = resp.json().get("genres", [])
            return {g["id"]: g["name"] for g in genres}
    except Exception:
        return {}
    return {}


@st.cache_data(show_spinner=False)
def buscar_info_tmdb(titulo: str, ano=None):
    """Busca poster, overview e gêneros no TMDB para um título de filme."""
    if not TMDB_API_KEY:
        return {"poster": PLACEHOLDER, "release_date": "", "overview": "", "genres": ""}
    genre_map = get_genre_map()
    titulo_limpo = (titulo or "").strip()
    # remove sufixo de ano entre parênteses se existir
    if " (" in titulo_limpo:
        titulo_limpo = titulo_limpo.split(" (")[0]
    params = {"api_key": TMDB_API_KEY, "query": titulo_limpo, "language": "pt-BR", "page": 1}
    if ano is not None:
        try:
            params["primary_release_year"] = int(ano)
        except Exception:
            pass
    try:
        resp = requests.get(TMDB_SEARCH_URL, params={k: v for k, v in params.items() if v}, timeout=6)
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                # prefer results com poster
                with_poster = [r for r in results if r.get("poster_path")]
                chosen = None
                if with_poster:
                    chosen = sorted(with_poster, key=lambda x: x.get("popularity", 0), reverse=True)[0]
                else:
                    chosen = results[0]
                genres_list = [genre_map.get(gid) for gid in chosen.get("genre_ids", []) if genre_map.get(gid)]
                return {
                    "poster": f"{TMDB_IMAGE_BASE}{chosen.get('poster_path')}" if chosen.get("poster_path") else PLACEHOLDER,
                    "release_date": chosen.get("release_date", ""),
                    "overview": chosen.get("overview", ""),
                    "genres": ", ".join(genres_list[:3])
                }
    except Exception:
        pass
    return {"poster": PLACEHOLDER, "release_date": "", "overview": "", "genres": ""}

BACKEND = "http://localhost:8000"

st.set_page_config(page_title="Recomendador - Filtragem por Conteúdo", layout="wide")

st.title("Sistema de Recomendação — Filtragem por Conteúdo")

col1, col2 = st.columns([2, 1])

# --- CSS para cards uniformes (garante imagens alinhadas usando object-fit) ---
st.markdown("""
<style>
.card-container{
    background-color: #0f1720; color: #e6eef8; border-radius: 8px; padding: 8px; margin-bottom: 12px;
}
.card-img{ width:100%; height:240px; overflow:hidden; border-radius:6px; display:flex; align-items:center; justify-content:center; background:#111218 }
.card-img img{ width:100%; height:100%; object-fit:cover; display:block }
.movie-title{ font-weight:600; margin:8px 0 4px 0; font-size:0.95rem; line-height:1.1; height:44px; overflow:hidden }
.card-meta{ font-size:0.8rem; color: #9aa4b2; margin-bottom:6px }
.card-score{ font-size:0.85rem; color:#a8d08d; margin-bottom:6px }
.card-actions{ margin-top:6px }
.modal-overlay{ display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:9999; align-items:center; justify-content:center }
.modal-overlay:target{ display:flex }
.modal-content{ background:#0f1720; color:#e6eef8; border-radius:8px; max-width:900px; width:90%; padding:18px; display:flex; gap:12px; position:relative; z-index:10001 }
.modal-content .left{ flex:0 0 300px }
.modal-content .right{ flex:1 }
.modal-close{ position:absolute; inset:0; z-index:10000 }
</style>
""", unsafe_allow_html=True)


with col1:
    st.header("Itens (catálogo)")
    # carregar itens do backend
    try:
        r = requests.get(f"{BACKEND}/itens", timeout=5)
        itens = r.json().get("itens", [])
        items_df = pd.DataFrame(itens)
    except Exception as e:
        st.error(f"Erro ao conectar ao backend: {e}")
        items_df = pd.DataFrame(columns=["item_id", "nome", "categoria", "tags", "descricao"])

    # Mostrar botão para editar favoritos (abre um multiselect pesquisável)
    st.write("Escolha itens como favoritos (opcional):")
    favorites_file = os.path.join("converted_data", "user_favorites.json")

    def load_favorites():
        if os.path.exists(favorites_file):
            try:
                with open(favorites_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_favorites(mapping: dict):
        os.makedirs(os.path.dirname(favorites_file), exist_ok=True)
        with open(favorites_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    favs_map = load_favorites()

    # detect title column (nome or title)
    if not items_df.empty:
        if "nome" in items_df.columns:
            title_col = "nome"
        elif "title" in items_df.columns:
            title_col = "title"
        else:
            # fallback to second column
            title_col = items_df.columns[1]

        # map title -> item_id (use last occurrence if duplicates)
        title_to_id = pd.Series(items_df["item_id"].astype(str).values, index=items_df[title_col].astype(str)).to_dict()

        # button to toggle editing favorites
        if st.button("Editar Favoritos"):
            st.session_state.show_fav_editor = True

        if st.session_state.get("show_fav_editor"):
            # load saved favorites for current user (user_choice may be empty)
            uid = str(st.session_state.get('user_choice')) if st.session_state.get('user_choice') else "anonymous"
            saved_ids = favs_map.get(uid, [])
            # convert saved ids to titles for default selection
            default_titles = [t for t, iid in title_to_id.items() if str(iid) in saved_ids]

            selected_titles = st.multiselect("Pesquisar e selecionar favoritos (pesquise por nome)", options=sorted(title_to_id.keys()), default=default_titles)

            # convert back to ids
            selected = [title_to_id[t] for t in selected_titles if t in title_to_id]

            if st.button("Salvar Favoritos"):
                uid = str(st.session_state.get('user_choice')) if st.session_state.get('user_choice') else "anonymous"
                favs_map[uid] = selected
                save_favorites(favs_map)
                st.success("Favoritos salvos para o usuário.")
                # hide editor after save
                st.session_state.show_fav_editor = False
        else:
            # if editor not shown, set selected to saved list for payload
            uid = str(st.session_state.get('user_choice')) if st.session_state.get('user_choice') else "anonymous"
            selected = favs_map.get(uid, [])
        st.markdown("---")
    else:
        st.info("Nenhum item carregado.")

    st.header("Gerar recomendações")
    use_manual = st.checkbox("Usar favoritos selecionados manualmente", value=False)
    n_rec = st.number_input("Número de recomendações", min_value=1, max_value=50, value=10)

with col2:
    st.header("Usuários (avaliacoes)")
    try:
        r2 = requests.get(f"{BACKEND}/usuarios", timeout=5)
        users = r2.json().get("usuarios", [])
        users_df = pd.DataFrame(users)
    except Exception:
        users_df = pd.DataFrame(columns=["usuario_id", "n_avaliacoes"])
    if not users_df.empty:
        user_choice = st.selectbox("Selecione um usuário existente (opcional)", options=[""] + users_df["usuario_id"].astype(str).tolist())
    else:
        user_choice = st.text_input("Ou digite um usuario_id manualmente")

    st.markdown("---")
    # Botão para avaliação global
    if st.button("Mostrar métricas (Precision/Recall/F1)"):
        try:
            metrics = requests.get(f"{BACKEND}/avaliacao").json()
            # Mostrar métricas como porcentagem quando disponíveis
            def fmt_pct(x):
                try:
                    if x is None:
                        return "N/A"
                    return f"{float(x)*100:.2f}%"
                except Exception:
                    return str(x)

            st.metric("Precision (média)", value=fmt_pct(metrics.get("precision")))
            st.metric("Recall (média)", value=fmt_pct(metrics.get("recall")))
            st.metric("F1-score (média)", value=fmt_pct(metrics.get("f1")))
            st.write(f"Usuários avaliados: {metrics.get('users_evaluated')}")
        except Exception as e:
            st.error(f"Erro ao buscar métricas: {e}")

# Gera recomendação (envia pedido ao backend e armazena em session_state)
if st.button("Gerar Recomendações"):
    payload = {}
    if use_manual:
        payload = {"use_favorites": True, "favorite_item_ids": selected, "n": int(n_rec)}
    else:
        uid = user_choice if user_choice else None
        if not uid:
            st.warning("Selecione um usuário ou use favoritos manuais.")
        payload = {"usuario_id": uid, "use_favorites": False, "n": int(n_rec)}
    try:
        resp = requests.post(f"{BACKEND}/recomendar", json=payload, timeout=10)
        if resp.status_code != 200:
            st.error(f"Erro do backend: {resp.status_code} - {resp.text}")
        else:
            data = resp.json()
            recs = data.get("recommendations", [])
            # salvar recomendações na sessão para que cliques em botões não os apaguem
            st.session_state.recs = recs
    except Exception as e:
        st.error(f"Erro ao chamar backend: {e}")

# Renderizar recomendações armazenadas na sessão (permanece após reruns)
recs = st.session_state.get('recs') if st.session_state.get('recs') else None
if recs:
    st.subheader("Recomendações")
    if not recs:
        st.info("Nenhuma recomendação retornada.")
    else:
        per_row = 5
        cols = st.columns(per_row)
        for i, r in enumerate(recs):
            with cols[i % per_row]:
                title = normalize_title(r.get("nome") or r.get("item_id"))
                score = r.get("score")
                # buscar info do TMDB (cached)
                info = buscar_info_tmdb(title)
                poster = info.get("poster", PLACEHOLDER)
                release = info.get("release_date", "")
                genres = info.get("genres", "")
                overview = info.get("overview", "")
                score_pct = None
                if score is not None:
                    try:
                        score_pct = int(round(float(score) * 100))
                    except Exception:
                        score_pct = None
                card_html = f'''<div class="card-container"> 
                    <div class="card-img"><img src="{poster}" alt="{html.escape(title)}"/></div>
                    <div class="card-body">
                        <div class="movie-title">{html.escape(title)}</div>
                        <div class="card-meta">{release} {'— ' + genres if genres else ''}</div>
                        <div class="card-score">{('Score: ' + str(score_pct) + '%') if score_pct is not None else ''}</div>
                    </div>
                </div>'''
                # render card HTML
                st.markdown(card_html, unsafe_allow_html=True)
                # modal using :target anchor trick (click outside or on overlay to close)
                modal_id = f"modal-{i}"
                modal_html = f'''<a href="#{modal_id}" class="btn" style="display:inline-block;margin-top:6px;padding:6px 12px;border-radius:6px;background:#111827;color:#fff;text-decoration:none">Ver detalhes</a>
                <div id="{modal_id}" class="modal-overlay">
                  <a class="modal-close" href="#"></a>
                  <div class="modal-content">
                    <div class="left"><img src="{poster}" style="width:100%;height:auto;border-radius:6px"/></div>
                    <div class="right">
                      <h3 style="margin:0 0 8px 0">{html.escape(title)}</h3>
                      <div style="color:#9aa4b2;margin-bottom:8px">{release} {'— ' + genres if genres else ''}</div>
                      <div style="color:#a8d08d;margin-bottom:8px">{('Score: ' + str(score_pct) + '%') if score_pct is not None else ''}</div>
                      <div style="margin-bottom:12px">{html.escape(overview or 'Sinopse não disponível.')}</div>
                      <div><a href="https://www.themoviedb.org/search?query={requests.utils.requote_uri(title)}" target="_blank" style="color:#7dd3fc">Abrir no TMDB</a></div>
                    </div>
                  </div>
                </div>'''
                st.markdown(modal_html, unsafe_allow_html=True)
