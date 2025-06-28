import pandas as pd

def extract_5g_data(input_csv_path='raw_dataset.csv', output_csv_path='5g_data.csv'):
    """
    Lê um arquivo CSV, filtra os dados para incluir apenas a tecnologia de rede '5G',
    e salva os dados filtrados em um novo arquivo CSV.

    Args:
        input_csv_path (str): O caminho para o arquivo CSV de entrada (dados brutos).
        output_csv_path (str): O caminho para o arquivo CSV de saída (apenas dados 5G).
    """
    try:
        # Carrega o arquivo CSV em um DataFrame do pandas
        df = pd.read_csv(input_csv_path)

        # Converte a coluna 'NetworkTech' para string e a coloca em maiúsculas para evitar problemas de capitalização
        df['NetworkTech'] = df['NetworkTech'].astype(str).str.upper()

        # Filtra as linhas onde 'NetworkTech' é '5G'
        df_5g = df[df['NetworkTech'] == '5G']

        # Salva o DataFrame filtrado em um novo arquivo CSV
        df_5g.to_csv(output_csv_path, index=False)

        print(f"Dados de rede 5G extraídos com sucesso para '{output_csv_path}'")
        print(f"Total de linhas no arquivo original: {len(df)}")
        print(f"Total de linhas com tecnologia 5G: {len(df_5g)}")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_csv_path}' não foi encontrado.")
    except KeyError:
        print("Erro: A coluna 'NetworkTech' não foi encontrada no arquivo CSV. Verifique o nome da coluna.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# Chama a função para executar o processo
if __name__ == "__main__":
    extract_5g_data()