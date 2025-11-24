Sistema de Recomendação de Filmes - Filtragem Baseada em Conteúdo
Este projeto implementa um sistema de recomendação de filmes utilizando Filtragem Baseada em Conteúdo. O sistema analisa as características textuais dos filmes (título, gênero, tags, descrição) para sugerir novos itens similares ao perfil de preferências do usuário.

1. Objetivo do Sistema
O objetivo principal é desenvolver e avaliar um motor de recomendação que sugira itens relevantes ao usuário baseando-se exclusivamente nos atributos de conteúdo dos itens, sem utilizar filtragem colaborativa (histórico de outros usuários) como fator principal. O projeto inclui uma API (Backend) e uma interface web interativa (Frontend), além de um módulo de avaliação de métricas (Precision, Recall e F1-Score).


2. Cenário e Dados
O cenário escolhido foi Filmes. Recomendações são relevantes neste cenário devido à vasta quantidade de opções disponíveis, o que dificulta a escolha manual pelo usuário.

O sistema requer dois arquivos CSV na raiz do projeto:

filmes.csv: Contém item_id, title, generos, tags e descricao (atributos de conteúdo).

aval.csv: Contém usuario_id, item_id, nota (usado apenas para criar o perfil do usuário e validar métricas).

3. Arquitetura e Tecnologias
Backend: Python com FastAPI. Responsável pela lógica de recomendação, vetorização e cálculo de métricas.

Frontend: Python com Streamlit. Interface para interação com o usuário e visualização dos resultados.

Bibliotecas Principais: scikit-learn (Machine Learning), pandas (Manipulação de dados), numpy.

4. Como Rodar o Projeto
Pré-requisitos
Certifique-se de ter o Python instalado e as bibliotecas necessárias:

Bash

pip install fastapi uvicorn streamlit pandas scikit-learn numpy requests
Executando o Backend
No terminal, navegue até a pasta do projeto e execute:

Bash

uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
O backend estará rodando em http://localhost:8000.

Executando o Frontend
Em um novo terminal, execute:

Bash

streamlit run frontend.py
O navegador abrirá automaticamente a interface do sistema.

5. Implementação Técnica
Vetorização dos Itens
A transformação dos atributos textuais dos filmes em vetores numéricos foi realizada utilizando TF-IDF (Term Frequency-Inverse Document Frequency). O processo de construção do corpus textual para cada filme segue a seguinte lógica:

Título: Repetido para aumentar seu peso relativo.

Gêneros: Tokenizados especificamente (ex: genre_Action).

Ano: Extraído da data de lançamento.

Tags e Descrição: Concatenados e limpos.

O vetorizador TfidfVectorizer gera uma matriz esparsa que é normalizada (Norma L2) para garantir que a magnitude dos vetores não influencie o cálculo de similaridade.

Construção do Perfil do Usuário
O perfil do usuário é uma representação vetorial calculada a partir dos itens que ele interagiu positivamente (nota >= 4 ou selecionados manualmente).

Se houver notas: O vetor do perfil é a soma ponderada dos vetores dos filmes avaliados (onde o peso é a nota dada).

Se não houver notas (apenas seleção): O vetor é a soma simples ou média dos vetores dos itens escolhidos.

O vetor resultante é normalizado para manter a consistência com a matriz de itens.


Métrica de Similaridade
A métrica escolhida para comparar o vetor do perfil do usuário com os vetores de todos os filmes do catálogo foi a Similaridade do Cosseno. Ela mede o cosseno do ângulo entre dois vetores, resultando em um valor entre -1 e 1 (no nosso caso, entre 0 e 1 devido ao TF-IDF), onde 1 indica identidade máxima de conteúdo.

6. Avaliação do Sistema (Métricas)
O sistema possui um endpoint /avaliacao que calcula métricas de performance offline usando o dataset de avaliações (aval.csv).

Metodologia de Cálculo
Para cada usuário com pelo menos 2 avaliações positivas (nota >= 4):

O histórico é dividido em Treino (80%) e Teste (20%).

Um perfil é gerado usando apenas os itens de Treino.

O sistema gera recomendações baseadas nesse perfil.

Verifica-se quantos itens do conjunto de Teste (gabarito) apareceram nas recomendações.

Interpretação dos Resultados 

Precision: De todos os filmes recomendados, qual porcentagem o usuário realmente avaliou positivamente no conjunto de teste? (Indica qualidade/relevância).

Recall: De todos os filmes que o usuário gosta (no teste), qual porcentagem o sistema conseguiu encontrar e recomendar? (Indica cobertura).

F1-Score: Média harmônica entre Precision e Recall. É a métrica principal para balancear o compromisso entre recomendar apenas o "certo" (precisão) e encontrar "todos" os certos (recall).

Resultados próximos de 0 indicam que o conteúdo textual dos filmes pode não ser suficiente para explicar o gosto do usuário ou que o dataset é esparso. Resultados mais altos indicam que o usuário tende a gostar de filmes com palavras-chave e gêneros muito similares.