
# 🎛️ Simulador de Alocação de Recursos 5G com Interface Gráfica

![Versão](https://img.shields.io/badge/versão-1.0-blue)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-darkgreen)
![Algoritmo](https://img.shields.io/badge/Algoritmo-Genético-purple)

Uma aplicação interativa para simular e comparar estratégias de alocação de recursos em redes 5G. Permite ao usuário ajustar parâmetros de um Algoritmo Genético (AG) e comparar seu desempenho com abordagens clássicas como Round Robin, Best CQI e Proportional Fair.

---

## 🧠 Funcionalidades Principais

### Interface Gráfica (arquivo `5g_v2.py`)
* **Interface Moderna:** Construída com `CustomTkinter` em modo escuro.
* **Parâmetros Personalizáveis:** Usuário pode definir população, gerações, taxa de mutação e elitismo.
* **Execução Assíncrona:** A simulação roda em background (via threads) para não travar a GUI.
* **Visualização de Resultados:**
  - Tabela comparativa de satisfação e justiça.
  - Gráficos de barras (comparativo de algoritmos).
  - Curvas de evolução do AG (fitness máximo e médio).
* **Exportação de Dados:** Classificação dos usuários salva automaticamente em CSV.

### Classificação e Pré-processamento (arquivo `Classificador.py`)
* Filtra dados 5G de um dataset maior.
* Converte colunas e trata valores inválidos.
* Classifica usuários com base em critérios de QoS:
  - Video Streaming
  - Video Call
  - Web Browse
  - VoIP Call
  - Background Download

---

## 🖼️ Pré-visualização

*Insira abaixo um print da aplicação em execução:*

![screenshot](coloque-seu-print-aqui.png)

---

## 📁 Estrutura dos Arquivos

```bash
.
├── 5g_v2.py                      # Interface gráfica e lógica da simulação
├── Classificador.py              # Filtro de dados para rede 5G
├── Quality of Service 5G.csv     # Dataset original com múltiplas tecnologias
├── 5g_data.csv                   # Dataset filtrado contendo apenas 5G
├── 5g_data-classification.csv    # Dataset com usuários classificados por aplicação
```

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos

Certifique-se de ter o Python 3.10+ instalado. Depois, instale as dependências:

```bash
pip install pandas customtkinter matplotlib numpy
```

### 2. Filtrar dados 5G (opcional)

```bash
python Classificador.py
```

### 3. Iniciar Simulação com GUI

```bash
python 5g_v2.py
```

---

## 🔧 Tecnologias Utilizadas

- **Python 3.10+**
- **Pandas** – análise e manipulação de dados
- **NumPy** – operações matemáticas
- **CustomTkinter** – interface gráfica moderna
- **Matplotlib** – geração de gráficos
- **Threading/Queue** – execução paralela segura
- **CSV** – formato de entrada e saída dos dados

---

## ✍️ Autor

**Gustavo**  
Engenheiro Eletrônica e de Telecomunicações e Mestrando em Engenharia Elétrica 


