# 📡 Simulador de Alocação de Recursos em Redes 5G com Interface Gráfica

Este projeto oferece uma aplicação interativa para simular, visualizar e comparar algoritmos de alocação de recursos em redes 5G. A interface foi desenvolvida com `CustomTkinter`, enquanto os dados são analisados com `Pandas` e os resultados visualizados com `Matplotlib`.

## 🔍 Visão Geral

O sistema é dividido em duas partes principais:

- **Classificador de Dados (`Classificador.py`)**  
  Filtra os dados de entrada para incluir apenas registros de redes 5G e os exporta para análise posterior.

- **Simulador com GUI (`5g_v2.py`)**  
  Apresenta uma interface gráfica onde o usuário pode executar simulações com diferentes algoritmos de alocação de recursos:
  
  - **Algoritmo Genético (AG)** otimizado
  - **Round Robin (RR)**
  - **Best CQI (ganancioso)**
  - **Proportional Fair (PF)**

A performance de cada algoritmo é avaliada com base em:
- **Satisfação do usuário**
- **Justiça na alocação (Índice de Jain)**

## 📁 Estrutura do Projeto

```
.
├── Classificador.py                # Filtro de dados para apenas registros 5G
├── 5g_v2.py                        # Simulador com GUI e algoritmos de alocação
├── Quality of Service 5G.csv      # Dataset original com múltiplas tecnologias
├── 5g_data.csv                    # Dataset filtrado contendo apenas dados 5G
├── 5g_data-classification.csv     # Dataset classificado por tipo de aplicação
```

## 🚀 Como Executar

### 1. Pré-requisitos

Você precisará das seguintes bibliotecas:

```
pip install pandas customtkinter matplotlib numpy
```

### 2. Filtrar Dados 5G (opcional)

Se estiver usando um dataset bruto:

```
python Classificador.py
```

Isso irá gerar `5g_data.csv`.

### 3. Rodar a Simulação

Execute a interface gráfica:

```
python 5g_v2.py
```

## 🎛️ Funcionalidades

- Interface gráfica escura e moderna
- Entrada de parâmetros para o AG:
  - Tamanho da população
  - Número de gerações
  - Taxa de mutação
  - Número de elites
- Comparativo visual entre algoritmos
- Exportação automática de classificação de usuários (`Application_Type`)

## 📊 Lógica de Classificação de Aplicações

Os usuários são classificados em:

- `Video_Streaming`
- `Video_Call`
- `Web_Browse`
- `VoIP_Call`
- `Background_Download`

A classificação é baseada em:

- `DL_bitrate`
- `UL_bitrate`
- `Ping`
- `CQI`

## 📈 Métricas

- **Satisfação:** Quanto a alocação atende às necessidades da aplicação do usuário.
- **Justiça (Jain):** Quão uniformemente os recursos são distribuídos.

## 🧠 Algoritmo Genético

Implementa operadores personalizados:

- Criação aleatória com reparo
- Crossover com correção de excesso de RBS
- Mutação híbrida (guiada + aleatória)
- Seleção por torneio

## 🧪 Exemplo de Resultado

```
Algoritmo         | Satisfação | Justiça (Jain)
---------------------------------------------
Alg. Genético     | 93.25%     | 89.45%
Round Robin       | 78.12%     | 97.23%
Best CQI          | 84.51%     | 81.10%
Prop. Fair        | 87.30%     | 85.60%
```

## 👨‍💻 Autor

Gustavo Coelho Domingos — Engenheiro em Eletônica e de Telecomunicações 
