import streamlit as st
import requests
import pandas as pd

st.title("🎬 Sistema de Recomendação de Filmes")
st.write("Baseado em filtragem colaborativa")

# Entrada: escolher usuário
usuario_id = st.number_input("Digite o ID do usuário (1 a 15)", min_value=1, max_value=15, step=1)
n_recomendacoes = st.slider("Número de recomendações", 1, 10, 5)

if st.button("Gerar Recomendações"):
    resposta = requests.post("http://127.0.0.1:8000/recomendar", json={
        "usuario_id": usuario_id,
        "n_recomendacoes": n_recomendacoes
    })

    if resposta.status_code == 200:
        recomendacoes = resposta.json()
        if isinstance(recomendacoes, dict) and "erro" in recomendacoes:
            st.error(recomendacoes["erro"])
        else:
            st.success("Recomendações geradas:")
            df = pd.DataFrame(recomendacoes)
            st.table(df)
    else:
        st.error("Erro ao conectar com o backend.")