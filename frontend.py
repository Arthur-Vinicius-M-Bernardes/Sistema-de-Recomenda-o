import os
import html
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sistema de Recomendação - MovieLens 100K", layout="wide")

# Configuração da API do TMDB
TMDB_API_KEY = "5a0a89104a4fc273fb664827c8682454"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_GENRE_URL = "https://api.themoviedb.org/3/genre/movie/list"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER = "https://via.placeholder.com/300x450?text=Sem+Imagem"

if not TMDB_API_KEY:
    st.warning("A variável de ambiente TMDB_API_KEY não está definida.")

st.title("Sistema de Recomendação de Filmes")

# --- Carregar dados dos filmes para o seletor ---
@st.cache_data
def carregar_filmes():
    try:
        df_filmes = pd.read_csv(os.path.join("converted_data", "movies.csv"))
        if 'title' in df_filmes.columns:
            df_filmes.rename(columns={'title': 'titulo'}, inplace=True)
        mapa_titulo_id = pd.Series(df_filmes.movie_id.values, index=df_filmes.titulo).to_dict()
        return df_filmes, mapa_titulo_id
    except FileNotFoundError:
        st.error("ERRO: 'converted_data/movies.csv' não encontrado. Execute o script de conversão primeiro.")
        return pd.DataFrame(columns=['titulo']), {}

df_filmes, mapa_titulo_id = carregar_filmes()

# --- Funções de busca de dados ---
@st.cache_data(show_spinner=False)
def get_genre_map():
    if not TMDB_API_KEY: return {}
    params = {"api_key": TMDB_API_KEY, "language": "pt-BR"}
    try:
        resp = requests.get(TMDB_GENRE_URL, params=params, timeout=5)
        if resp.status_code == 200:
            genres = resp.json().get("genres", [])
            return {genre["id"]: genre["name"] for genre in genres}
    except: return {}

@st.cache_data(show_spinner=False)
def buscar_info_tmdb(titulo: str, ano=None):
    if not TMDB_API_KEY: return {"poster": PLACEHOLDER, "release_date": "", "overview": "", "genres": ""}
    genre_map = get_genre_map()
    titulo_limpo = titulo.strip().split(' (')[0]
    articles = [", the", ", a", ", an"]
    for art in articles:
        if titulo_limpo.lower().endswith(art):
            titulo_limpo = f"{art[2:].strip().capitalize()} {titulo_limpo[:-len(art)]}"
            break
    params = {"api_key": TMDB_API_KEY, "query": titulo_limpo, "language": "pt-BR", "page": 1}
    if ano: params["primary_release_year"] = int(ano) if pd.notna(ano) else None
    
    try:
        resp = requests.get(TMDB_SEARCH_URL, params={k: v for k, v in params.items() if v is not None})
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if not results and "primary_release_year" in params:
                params.pop("primary_release_year", None)
                resp = requests.get(TMDB_SEARCH_URL, params=params)
                if resp.status_code == 200: results = resp.json().get("results", [])
            
            results_with_posters = [r for r in results if r.get("poster_path")]
            if results_with_posters:
                best = sorted(results_with_posters, key=lambda x: x.get("popularity", 0), reverse=True)[0]
                genres_list = [genre_map.get(gid) for gid in best.get("genre_ids", []) if genre_map.get(gid)]
                return {
                    "poster": f"{TMDB_IMAGE_BASE}{best.get('poster_path')}",
                    "release_date": best.get("release_date", ""),
                    "overview": best.get("overview", ""),
                    "genres": ", ".join(genres_list[:3])
                }
    except Exception as e: print(f"Erro TMDB: {e}")
    return {"poster": PLACEHOLDER, "release_date": "", "overview": "", "genres": ""}

# --- Função do Modal (usando decorador para robustez) ---
@st.dialog("Detalhes do Filme")
def mostrar_detalhes_filme(info):
    """Cria e exibe o modal de detalhes do filme."""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(info.get('poster', PLACEHOLDER))
    with col2:
        st.subheader(info['titulo'])
        st.caption(f"Lançamento: {info.get('release_date', 'N/A')} | Score: {info.get('score', 0)}%")
        st.markdown(f"**Gêneros:** {info.get('genres', 'N/A')}")
        st.markdown("---")
        st.markdown(f"**Sinopse:**")
        st.markdown(f"<p style='text-align: justify;'>{info.get('overview', 'Não disponível.')}</p>", unsafe_allow_html=True)

    if st.button("Fechar", key="fechar_modal"):
        del st.session_state.filme_para_exibir
        st.rerun()

# --- Layout Principal ---
usuario_id = st.number_input("Digite o ID do usuário (1 a 943)", min_value=1, max_value=943, value=1, step=1)
n_recomendacoes = st.slider("Número de recomendações", min_value=1, max_value=20, value=10)

if not df_filmes.empty:
    with st.expander("📝 Avaliar um Filme"):
        filme_selecionado = st.selectbox(
            "Escolha um filme para avaliar",
            options=sorted(df_filmes['titulo'].dropna().unique()),
            index=0
        )
        nota_filme = st.slider("Sua nota (1 a 5 estrelas)", 1, 5, 3)
        
        if st.button("Enviar Avaliação"):
            id_filme_avaliado = mapa_titulo_id.get(filme_selecionado)
            if id_filme_avaliado:
                with st.spinner("A enviar a sua avaliação..."):
                    try:
                        payload = {"usuario_id": int(usuario_id), "movie_id": int(id_filme_avaliado), "rating": int(nota_filme)}
                        resp = requests.post("http://127.0.0.1:8000/avaliar-filme", json=payload, timeout=15)
                        if resp.status_code == 200:
                            st.success(f"Avaliação para '{filme_selecionado}' enviada!")
                        else:
                            st.error(f"O backend retornou um erro: {resp.status_code} - {resp.text}")
                    except Exception as e:
                        st.error(f"Erro ao conectar ao backend: {e}")

col1, col2 = st.columns(2)
with col1:
    if st.button("Gerar Recomendações"):
        with st.spinner("A consultar o backend..."):
            try:
                resp = requests.post("http://127.0.0.1:8000/recomendar", json={"usuario_id": int(usuario_id), "n_recomendacoes": int(n_recomendacoes)})
                if resp.status_code == 200:
                    st.session_state.recomendacoes = resp.json()
                    if 'filme_para_exibir' in st.session_state:
                         del st.session_state['filme_para_exibir']
                else: st.error(f"Erro no backend: {resp.text}")
            except Exception as e: st.error(f"Erro de conexão: {e}")

with col2:
    if st.button("Avaliar Acurácia"):
        with st.spinner("A calcular acurácia..."):
            try:
                resp = requests.post("http://127.0.0.1:8000/avaliar", json={"usuario_id": int(usuario_id), "n_recomendacoes": int(n_recomendacoes)})
                if resp.status_code == 200:
                    res = resp.json()
                    if "erro" in res: st.error(res["erro"])
                    else:
                        st.success(f"Acurácia para usuário {usuario_id}:")
                        st.metric(label="Acurácia", value=f"{int(res['acuracia']*100)}%", delta=f"{res['acertos']} acertos de {res['total_recomendado']}")
                else: st.error(f"Erro no backend: {resp.text}")
            except Exception as e: st.error(f"Erro de conexão: {e}")

# --- Chamada da função do Modal ---
if 'filme_para_exibir' in st.session_state:
    mostrar_detalhes_filme(st.session_state.filme_para_exibir)

# --- CSS para os cards ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
body { font-family: 'Inter', sans-serif; }
.card-container {
    background-color: #1a1a24; border-radius: 12px; padding: 1rem; border: 1px solid #2a2a38;
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; margin-bottom: 1rem;
}
.card-container:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.4); }
.movie-title {
    font-weight: 600; font-size: 1rem; color: #f0f0f5; margin-top: 0.5rem; height: 48px;
    overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
</style>
""", unsafe_allow_html=True)

# --- Renderização da Grelha de Recomendações ---
if 'recomendacoes' in st.session_state:
    recomendacoes = st.session_state.recomendacoes
    if isinstance(recomendacoes, dict) and "erro" in recomendacoes:
        st.error(recomendacoes["erro"])
    elif recomendacoes:
        num_cols = 5
        cols = st.columns(num_cols)
        for i, rec in enumerate(recomendacoes):
            with cols[i % num_cols]:
                with st.container():
                    st.markdown(f'<div class="card-container">', unsafe_allow_html=True)
                    info_api = buscar_info_tmdb(rec['titulo'], rec.get('ano'))
                    score_pct = int(round(rec.get("score", 0.0) / 5.0 * 100))
                    st.image(info_api['poster'])
                    st.markdown(f'<p class="movie-title">{html.escape(rec["titulo"])}</p>', unsafe_allow_html=True)
                    if st.button("Ver Detalhes", key=f"details_{rec['movie_id']}"):
                        st.session_state.filme_para_exibir = {
                            'titulo': rec['titulo'], 'poster': info_api['poster'], 'release_date': info_api['release_date'],
                            'genres': info_api['genres'], 'overview': info_api['overview'], 'score': score_pct
                        }
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

