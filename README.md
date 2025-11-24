## Sumário
1. [Visão Geral](#-visão-geral)
2. [Principais Recursos](#-principais-recursos)
3. [Arquitetura](#-arquitetura)
4. [Dados e Preparação](#-dados-e-preparação)
5. [Abordagem de Recomendação](#-abordagem-de-recomendação)
6. [API (Backend FastAPI)](#-api-backend-fastapi)
7. [Frontend (Streamlit)](#-frontend-streamlit)
8. [Instalação e Execução](#-instalação-e-execução)
9. [Avaliação e Métricas](#-avaliação-e-métricas)

---

## Visão Geral
Este projeto implementa um sistema de recomendação baseado em **filtragem por conteúdo**. Em vez de usar o comportamento de múltiplos usuários (filtragem colaborativa), ele se apoia nos próprios atributos dos filmes para encontrar similaridade: título, gêneros tokenizados, ano, tags e descrição.

É composto por:
- Uma **API (FastAPI)** que expõe endpoints para itens, usuários, recomendações, avaliação e reconstrução dos vetores.
- Um **frontend (Streamlit)** simples para interação, seleção de favoritos e visualização dos resultados.

> Base de dados principal: adaptação do **MovieLens 100K** e arquivos auxiliares convertidos (pasta `converted_data/`).

## Principais Recursos
- Vetorização de conteúdo com **TF-IDF** (palavras + opção caractere) e normalização L2.
- Perfil do usuário construído por itens positivos (nota >= 4) ou favoritos selecionados manualmente (com penalização de avaliações negativas).
- Similaridade via **Cosseno** para ranking + pequeno fator de popularidade.
- Endpoint de métricas globais: Precision / Recall / F1.
- Mecanismo de reconstrução dos vetores (`/rebuild_vectors`) com ajuste rápido de hiperparâmetros relevantes (repetições, min_df, n-gram, limites de features).

## Arquitetura
| Camada | Tecnologia | Responsabilidade |
|--------|------------|------------------|
| Backend | FastAPI | Endpoints, carregamento de dados, vetorização, recomendações, métricas |
| Frontend | Streamlit | Interface para busca, seleção de favoritos e visualização das recomendações |
| Modelo | scikit-learn | TF-IDF + Similaridade do Cosseno |
| Dados | MovieLens / CSV | Fonte de itens e avaliações |

## Dados e Preparação
Arquivos esperados na raiz:
- `filmes.csv` – catálogo de filmes; aceita formato original `u.item` (pipe `|`) ou versão já convertida.
- `aval.csv` – avaliações (`usuario_id`, `item_id`, `nota` [, `timestamp`]). Se ausente, o sistema funciona apenas com favoritos.

Ao carregar, o backend:
1. Renomeia colunas para um formato canônico (`item_id`, `nome`, `descricao`, `tags`, `categoria`).
2. Normaliza títulos (`"Matrix, The" -> "The Matrix"`).
3. Constrói tokens de gênero: `Action, Drama -> genre_Action genre_Drama`.
4. Extrai ano: gera `year_YYYY` quando possível.
5. Prepara campo composto para vetorização.

## Abordagem de Recomendação
1. Constrói um corpus textual por item (título repetido para maior peso + tokens de gênero + ano + tags limpas + descrição).
2. Vetorização principal: **TF-IDF** combinando análise de palavras (unigramas e bigramas) e caractere (char_wb 3–5) com `min_df = 1` (mantém termos raros potencialmente discriminativos), remoção de stopwords básicas e normalização L2. A definição de `min_df = 1` faz parte do cálculo padrão para ampliar recall sem sacrificar a precisão observada.
3. Cria perfil do usuário somando (ou ponderando pelas notas) os vetores dos itens positivos (e penalizando itens avaliados negativamente) seguido de normalização.
4. Calcula similaridade do perfil com todos os itens (Cosine) e ordena; aplica leve mistura de popularidade (`POP_ALPHA = 0.05`) para favorecer itens bem recebidos.
5. (Somente conteúdo – sem mistura colaborativa neste escopo simplificado).

### Métrica de Similaridade
`similaridade = cos(v_perfil, v_item)` → valores entre 0 e 1 (não-negativos, pois TF-IDF). Quanto mais próximo de 1, maior alinhamento semântico.

### Perfil do Usuário
- Com avaliações: soma ponderada (peso = nota) → normalização.
- Sem avaliações (modo favoritos): soma simples dos vetores selecionados.

## API (Backend FastAPI)
Base URL padrão: `http://localhost:8000`

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/itens` | Lista itens do catálogo |
| GET | `/usuarios` | Lista usuários e nº de avaliações |
| POST | `/recomendar` | Gera recomendações (por usuário ou favoritos) |
| GET | `/avaliacao` | Retorna métricas globais (precision/recall/f1) |
| POST | `/rebuild_vectors` | Recarrega dados e refaz vetorização |

### Exemplo: Recomendar por usuário
```jsonc
POST /recomendar
{
	"usuario_id": "42",
	"n": 10
}
```

### Exemplo: Recomendar usando favoritos
```jsonc
POST /recomendar
{
	"use_favorites": true,
	"favorite_item_ids": ["50", "172", "250"],
	"n": 8
}
```

## Frontend (Streamlit)
Interface simples em `frontend.py` para:
- Seleção de usuário ou montagem manual de favoritos.
- Visualização de recomendações ordenadas.
- Exibição de métricas agregadas.

## Instalação e Execução

### 1. Clonar o repositório
```powershell
git clone https://github.com/Arthur-Vinicius-M-Bernardes/Sistema-de-Recomenda-o.git
cd Sistema-de-Recomenda-o
```

Se houver erro de certificado (Windows / schannel), você pode temporariamente:
```powershell
git config --global http.sslVerify false
# (Depois de clonar, recomende reativar) git config --global http.sslVerify true
```

### 2. Criar ambiente virtual (opcional, recomendado)
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

### 3. Instalar dependências
```powershell
pip install -r requirements.txt
```
Ou manualmente:
```powershell
pip install fastapi uvicorn streamlit pandas scikit-learn numpy requests
```

### 4. Iniciar Backend
```powershell
python -m uvicorn backend:app --reload --port 8000
```
Acesse: http://localhost:8000/docs

### 5. Iniciar Frontend
```powershell
python -m streamlit run frontend.py
```
Acesse: http://localhost:8501

Se o backend estiver em outra porta/host, você pode:
- Ajustar no próprio app (barra lateral → BACKEND URL), ou
- Definir antes de iniciar o Streamlit:

```powershell
$env:BACKEND_URL = "http://localhost:8000"  # altere se usar outra porta/host
python -m streamlit run frontend.py
```

## Avaliação e Métricas
Endpoint `/avaliacao` calcula métricas offline:
- **Precision**: proporção de recomendações que são relevantes.
- **Recall**: cobertura dos itens relevantes do usuário.
- **F1**: equilíbrio entre precision e recall.

A configuração corrente (min_df=1 + combinação word/char + leve prior de popularidade) mostrou bom equilíbrio entre precisão e recall em testes internos.

