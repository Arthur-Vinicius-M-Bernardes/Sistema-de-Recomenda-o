import os
import html
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sistema de Recomendação - MovieLens 100K", layout="wide")

# Configuração da API do TMDB (retirada da variável de ambiente)
TMDB_API_KEY = "5a0a89104a4fc273fb664827c8682454"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_GENRE_URL = "https://api.themoviedb.org/3/genre/movie/list"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://via.placeholder.com/300x450?text=Sem+Imagem"

# Mensagem se a chave não estiver definida
if not TMDB_API_KEY:
    st.warning("A variável de ambiente TMDB_API_KEY não está definida. Os pósteres podem não ser exibidos.")

st.title("🎬 Sistema de Recomendação de Filmes")
st.write("MovieLens 100K — Recomendação por similaridade (cosseno)")

# Entradas do utilizador
usuario_id = st.number_input("Digite o ID do usuário (1 a 943)", min_value=1, max_value=943, value=1, step=1)
n_recomendacoes = st.slider("Número de recomendações", min_value=1, max_value=20, value=10)

@st.cache_data(show_spinner=False)
def get_genre_map():
    """Busca o mapa de IDs de género para nomes da API do TMDB."""
    if not TMDB_API_KEY:
        return {}
    params = {"api_key": TMDB_API_KEY, "language": "pt-BR"}
    try:
        resp = requests.get(TMDB_GENRE_URL, params=params, timeout=5)
        if resp.status_code == 200:
            genres = resp.json().get("genres", [])
            return {genre["id"]: genre["name"] for genre in genres}
    except Exception as e:
        print(f"Erro ao buscar mapa de géneros: {e}")
    return {}

@st.cache_data(show_spinner=False)
def buscar_info_tmdb(titulo: str, ano=None):
    """Busca póster, data e outras informações no TMDB."""
    if not TMDB_API_KEY:
        return {"poster": PLACEHOLDER, "release_date": "", "overview": "", "genres": ""}

    genre_map = get_genre_map()
    titulo_limpo = titulo.strip()
    titulo_limpo = titulo_limpo.split(' (')[0]
    articles_to_move = [", the", ", a", ", an"]
    for article in articles_to_move:
        if titulo_limpo.lower().endswith(article):
            titulo_sem_artigo = titulo_limpo[:-len(article)]
            artigo_prefixo = article[2:].strip().capitalize()
            titulo_limpo = f"{artigo_prefixo} {titulo_sem_artigo}"
            break

    params = {
        "api_key": TMDB_API_KEY,
        "query": titulo_limpo,
        "language": "pt-BR",
        "include_adult": "false",
        "page": 1
    }
    if ano:
        params["primary_release_year"] = int(ano) if (isinstance(ano, (int, float)) and not pd.isna(ano)) else None
    params = {k: v for k, v in params.items() if v is not None}

    try:
        resp = requests.get(TMDB_SEARCH_URL, params=params, timeout=6)
        if resp.status_code == 200:
            results = resp.json().get("results", []) or []
            if not results and "primary_release_year" in params:
                params.pop("primary_release_year", None)
                resp2 = requests.get(TMDB_SEARCH_URL, params=params, timeout=6)
                if resp2.status_code == 200:
                    results = resp2.json().get("results", []) or []

            if results:
                results_with_posters = [res for res in results if res.get("poster_path")]
                if results_with_posters:
                    results_with_posters.sort(key=lambda x: x.get("popularity", 0), reverse=True)
                    best = results_with_posters[0]
                    poster_path = best.get("poster_path")
                    release = best.get("release_date", "")
                    overview = best.get("overview", "")
                    poster_url = f"{TMDB_IMAGE_BASE}{poster_path}"
                    genre_ids = best.get("genre_ids", [])
                    genres_list = [genre_map.get(gid) for gid in genre_ids if genre_map.get(gid)]
                    genres_str = ", ".join(genres_list[:3])
                    return {"poster": poster_url, "release_date": release, "overview": overview, "genres": genres_str}
    except Exception as e:
        print("Erro ao buscar no TMDB:", e)
    return {"poster": PLACEHOLDER, "release_date": "", "overview": "", "genres": ""}

# --- Estrutura com colunas para os botões ---
col1, col2 = st.columns(2)

with col1:
    if st.button("Gerar Recomendações"):
        with st.spinner("A consultar o backend e a preparar os cards..."):
            try:
                resp = requests.post("http://127.0.0.1:8000/recomendar",
                                     json={"usuario_id": int(usuario_id), "n_recomendacoes": int(n_recomendacoes)},
                                     timeout=12)
            except Exception as e:
                st.error(f"Erro ao conectar ao backend: {e}")
                st.stop()

            if resp.status_code != 200:
                st.error(f"O backend retornou um erro: {resp.status_code} - {resp.text}")
                st.stop()

            recomendacoes = resp.json()
            if isinstance(recomendacoes, dict) and "erro" in recomendacoes:
                st.error(recomendacoes["erro"])
                st.stop()

            df = pd.DataFrame(recomendacoes)
            if df.empty:
                st.info("Nenhuma recomendação encontrada.")
                st.stop()

            if "titulo" not in df.columns:
                st.error("A resposta do backend não contém a chave 'titulo'. Verifique o backend.")
                st.stop()
            if "score" not in df.columns:
                df["score"] = 0.0
            if "ano" not in df.columns:
                df["ano"] = None

            cards = []
            for _, row in df.iterrows():
                titulo_raw = str(row.get("titulo", "Título desconhecido"))
                info = buscar_info_tmdb(titulo_raw, row.get("ano"))
                
                try:
                    pct = int(round(float(row.get("score", 0.0)) / 5.0 * 100))
                    pct = max(0, min(100, pct))
                except:
                    pct = 0
                
                cards.append({
                    "titulo": html.escape(titulo_raw),
                    "score": pct,
                    "release_date": html.escape(info.get("release_date", "")),
                    "genres": html.escape(info.get("genres", "")),
                    "overview": html.escape(info.get("overview", "")),
                    "poster_url": info.get("poster", PLACEHOLDER)
                })
            
            # Placeholder para os cards serem renderizados depois
            st.session_state.cards_to_show = cards

with col2:
    if st.button("Avaliar Acurácia"):
        with st.spinner("A calcular a acurácia..."):
            try:
                resp = requests.post("http://127.0.0.1:8000/avaliar",
                                     json={"usuario_id": int(usuario_id), "n_recomendacoes": int(n_recomendacoes)},
                                     timeout=15) # Timeout um pouco maior para o cálculo
                if resp.status_code == 200:
                    resultado = resp.json()
                    if "erro" in resultado:
                        st.error(resultado["erro"])
                    else:
                        acuracia_pct = int(resultado["acuracia"] * 100)
                        acertos = resultado["acertos"]
                        total = resultado["total_recomendado"]
                        
                        st.success(f"Acurácia calculada para o usuário {usuario_id}!")
                        st.metric(
                            label="Acurácia do Modelo",
                            value=f"{acuracia_pct}%",
                            delta=f"{acertos} acertos de {total} recomendações",
                            delta_color="normal"
                        )
                else:
                    st.error(f"O backend retornou um erro: {resp.status_code} - {resp.text}")

            except Exception as e:
                st.error(f"Erro ao conectar ao backend para avaliação: {e}")

# --- Lógica de renderização dos cards (fora dos botões) ---
st.markdown(
    """
    <style>
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 18px; align-items: start; margin-top: 12px; }
    .card { background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.12); transition: transform .15s ease; position: relative; }
    .card:hover { transform: translateY(-6px); }
    .poster { width: 100%; height: 270px; object-fit: cover; display:block; }
    .info { padding: 10px; font-family: "Arial", sans-serif; color: #111; }
    .title { font-weight: 700; margin: 6px 0 4px 0; font-size: 14px; min-height: 38px; }
    .date { color: #666; font-size: 12px; margin-bottom: 6px; }
    .score-badge { position: absolute; top: 10px; left: 10px; width: 44px; height: 44px; border-radius: 22px; background: linear-gradient(180deg,#21d07a,#2bbf6f); color: white; display:flex; align-items:center; justify-content:center; font-weight:700; box-shadow: 0 2px 6px rgba(0,0,0,0.25); }
    .genres { font-size: 11px; color: #888; font-style: italic; margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .overview { font-size: 12px; color: #444; margin-top: 8px; display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden; text-overflow: ellipsis; }
    </style>
    """,
    unsafe_allow_html=True,
)

if 'cards_to_show' in st.session_state and st.session_state.cards_to_show:
    cards_html_list = ['<div class="grid">']
    for card in st.session_state.cards_to_show:
        card_html = (
            f'<div class="card">'
                f'<div style="position:relative;">'
                    f'<img class="poster" src="{card["poster_url"]}" alt="{card["titulo"]}">'
                    f'<div class="score-badge">{card["score"]}%</div>'
                f'</div>'
                f'<div class="info">'
                    f'<div class="title">{card["titulo"]}</div>'
                    f'<div class="date">{card["release_date"]}</div>'
                    f'<div class="genres">{card["genres"]}</div>'
                    f'<div class="overview">{card["overview"]}</div>'
                f'</div>'
            f'</div>'
        )
        cards_html_list.append(card_html)
    cards_html_list.append("</div>")
    st.markdown("".join(cards_html_list), unsafe_allow_html=True)
    # Limpa o estado para não mostrar os mesmos cards após uma atualização ou outra ação
    st.session_state.cards_to_show = None

