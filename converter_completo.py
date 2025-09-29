import pandas as pd
import os
import glob

# --- Configuração ---
ML_100K_DIR = "ml-100k"
OUTPUT_DIR = "converted_data"

def criar_diretorio_saida():
    """Cria o diretório de saída se não existir"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Diretório '{OUTPUT_DIR}' criado.")

def converter_ratings():
    """Converte u.data (ratings) para CSV"""
    print("Convertendo ratings (u.data)...")
    try:
        ratings = pd.read_csv(
            os.path.join(ML_100K_DIR, "u.data"),
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python"
        )
        # Converter timestamp para formato legível
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings.to_csv(os.path.join(OUTPUT_DIR, "ratings.csv"), index=False)
        print(f"✅ ratings.csv criado com {len(ratings)} registros")
    except Exception as e:
        print(f"❌ Erro ao converter ratings: {e}")

def converter_movies():
    """Converte u.item (movies) para CSV"""
    print("Convertendo filmes (u.item)...")
    try:
        # Definir todas as colunas do u.item
        cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
        genre_cols = ["unknown", "Action", "Adventure", "Animation", "Children", 
                     "Comedy", "Crime", "Documentary", "Drama", "Fantasy", 
                     "Film-Noir", "Horror", "Musical", "Mystery", "Romance", 
                     "Sci-Fi", "Thriller", "War", "Western"]
        all_cols = cols + genre_cols
        
        movies = pd.read_csv(
            os.path.join(ML_100K_DIR, "u.item"),
            sep="|",
            names=all_cols,
            encoding="latin-1",
            engine="python"
        )
        
        # Converter data de lançamento
        movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
        
        movies.to_csv(os.path.join(OUTPUT_DIR, "movies.csv"), index=False)
        print(f"✅ movies.csv criado com {len(movies)} registros")
    except Exception as e:
        print(f"❌ Erro ao converter movies: {e}")

def converter_users():
    """Converte u.user (users) para CSV"""
    print("Convertendo usuários (u.user)...")
    try:
        users = pd.read_csv(
            os.path.join(ML_100K_DIR, "u.user"),
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            engine="python"
        )
        users.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)
        print(f"✅ users.csv criado com {len(users)} registros")
    except Exception as e:
        print(f"❌ Erro ao converter users: {e}")

def converter_genres():
    """Converte u.genre (genres) para CSV"""
    print("Convertendo gêneros (u.genre)...")
    try:
        genres = pd.read_csv(
            os.path.join(ML_100K_DIR, "u.genre"),
            sep="|",
            names=["genre", "genre_id"],
            engine="python"
        )
        genres.to_csv(os.path.join(OUTPUT_DIR, "genres.csv"), index=False)
        print(f"✅ genres.csv criado com {len(genres)} registros")
    except Exception as e:
        print(f"❌ Erro ao converter genres: {e}")

def converter_occupations():
    """Converte u.occupation (occupations) para CSV"""
    print("Convertendo ocupações (u.occupation)...")
    try:
        occupations = pd.read_csv(
            os.path.join(ML_100K_DIR, "u.occupation"),
            names=["occupation"],
            engine="python"
        )
        # Adicionar um ID para cada ocupação
        occupations['occupation_id'] = range(len(occupations))
        occupations = occupations[['occupation_id', 'occupation']]  # Reordenar colunas
        occupations.to_csv(os.path.join(OUTPUT_DIR, "occupations.csv"), index=False)
        print(f"✅ occupations.csv criado com {len(occupations)} registros")
    except Exception as e:
        print(f"❌ Erro ao converter occupations: {e}")

def converter_datasets_treino_teste():
    """Converte os datasets de treino e teste (u*.base e u*.test)"""
    print("Convertendo datasets de treino e teste...")
    
    # Padrões dos arquivos
    patterns = ["u?.base", "u?.test", "ua.base", "ua.test", "ub.base", "ub.test"]
    
    for pattern in patterns:
        files = glob.glob(os.path.join(ML_100K_DIR, pattern))
        for file_path in files:
            try:
                filename = os.path.basename(file_path)
                print(f"  Convertendo {filename}...")
                
                df = pd.read_csv(
                    file_path,
                    sep="\t",
                    names=["user_id", "movie_id", "rating", "timestamp"],
                    engine="python"
                )
                
                # Converter timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                
                output_name = filename.replace('.base', '_train.csv').replace('.test', '_test.csv')
                df.to_csv(os.path.join(OUTPUT_DIR, output_name), index=False)
                print(f"    ✅ {output_name} criado com {len(df)} registros")
                
            except Exception as e:
                print(f"    ❌ Erro ao converter {filename}: {e}")

def converter_info():
    """Converte u.info (informações do dataset)"""
    print("Convertendo informações do dataset (u.info)...")
    try:
        # u.info é um arquivo de texto com informações, vamos lê-lo como texto
        with open(os.path.join(ML_100K_DIR, "u.info"), 'r') as f:
            content = f.read()
        
        # Criar um DataFrame com as informações
        lines = content.strip().split('\n')
        info_data = []
        for line in lines:
            if line.strip():
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    info_data.append({'count': int(parts[0]), 'description': parts[1]})
        
        info_df = pd.DataFrame(info_data)
        info_df.to_csv(os.path.join(OUTPUT_DIR, "dataset_info.csv"), index=False)
        print(f"✅ dataset_info.csv criado com {len(info_df)} registros")
    except Exception as e:
        print(f"❌ Erro ao converter info: {e}")

def gerar_relatorio():
    """Gera um relatório dos arquivos convertidos"""
    print("\n=== RELATÓRIO DE CONVERSÃO ===")
    
    if not os.path.exists(OUTPUT_DIR):
        print("❌ Diretório de saída não encontrado!")
        return
    
    csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))
    total_size = 0
    
    print(f"📁 Diretório: {OUTPUT_DIR}")
    print(f"📊 Total de arquivos CSV: {len(csv_files)}")
    print("\n📋 Detalhes dos arquivos:")
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            file_size = os.path.getsize(csv_file)
            total_size += file_size
            
            print(f"  📄 {os.path.basename(csv_file)}")
            print(f"     - Linhas: {len(df):,}")
            print(f"     - Colunas: {len(df.columns)}")
            print(f"     - Tamanho: {file_size:,} bytes")
            print(f"     - Colunas: {', '.join(df.columns.tolist())}")
            print()
        except Exception as e:
            print(f"  ❌ Erro ao ler {os.path.basename(csv_file)}: {e}")
    
    print(f"💾 Tamanho total: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

def converter_dataset_completo():
    """Função principal que converte todo o dataset"""
    print("🚀 INICIANDO CONVERSÃO COMPLETA DO MOVIELENS 100K")
    print("=" * 50)
    
    # Verificar se a pasta ml-100k existe
    if not os.path.isdir(ML_100K_DIR):
        print(f"❌ ERRO: A pasta '{ML_100K_DIR}' não foi encontrada.")
        print("Por favor, certifique-se de que o dataset está na pasta correta.")
        return
    
    # Criar diretório de saída
    criar_diretorio_saida()
    
    # Converter arquivos principais
    converter_ratings()
    converter_movies()
    converter_users()
    converter_genres()
    converter_occupations()
    converter_info()
    
    # Converter datasets de treino e teste
    converter_datasets_treino_teste()
    
    print("\n🎉 CONVERSÃO CONCLUÍDA!")
    
    # Gerar relatório
    gerar_relatorio()

if __name__ == "__main__":
    converter_dataset_completo()