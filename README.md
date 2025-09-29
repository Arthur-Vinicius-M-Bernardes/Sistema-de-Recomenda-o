* Objetivo do sistema
    Recomendar um número determinado de filmes para o usuário especificado

* Como executar o frontend e backend
    Executar os seguintes comandos no terminal
        uvicorn backend:app --reload
        streamlit run frontend.py

* Explicação da lógica de recomendação
    Foi utilizado a métrica de similaridade do Cosseno, onde os usuários são tratados como vetores então é medido o ângulo entre os vetores, para se ter a similaridade entre os gostos de cada usuário.

* Justificativa da métrica de similaridade usada
    Nossa base de dados é esparsa, cheia de zeros (ausência de avaliação). A métrica de similaridade do Cosseno lida melhor com esse tipo de estrutura, não tendo uma perda de qualidade significativa.
* Cálculo e análise da acurácia
    


# Sistema-de-Recomenda-o



