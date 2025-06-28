# -*- coding: utf-8 -*-
"""
Analisador de Desempenho de Alocação de Recursos 5G

Este script implementa uma aplicação com interface gráfica (GUI) para simular,
analisar e comparar diferentes algoritmos de alocação de recursos em uma rede 5G.

O foco principal é um Algoritmo Genético (AG) otimizado para equilibrar a
satisfação do utilizador e a justiça (fairness) na distribuição de recursos.
O desempenho do AG é comparado com três algoritmos de baseline:
- Round Robin (RR)
- Best CQI (Greedy)
- Proportional Fair (PF)

A interface é construída com CustomTkinter e os gráficos com Matplotlib.
A simulação é executada em uma thread separada para manter a GUI responsiva.
"""

# ===================== BIBLIOTECAS =====================
import customtkinter as ctk
import pandas as pd
import random
import numpy as np
import threading
import queue
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===================== PRÉ-PROCESSAMENTO DE DADOS =====================
def load_5g_csv(path="5g_data.csv"):
    """
    Tenta carregar o ficheiro CSV de dados da rede, testando separadores comuns.
    
    Args:
        path (str): O caminho para o ficheiro CSV.

    Returns:
        tuple: Um DataFrame do pandas e um valor de erro (None se sucesso).
    """
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if "DL_bitrate" in df.columns:
                return df, None
        except FileNotFoundError:
            return None, f"Arquivo '{path}' não encontrado."
    return None, "Coluna 'DL_bitrate' não encontrada no CSV."

def classify_and_prepare_data():
    """
    Carrega, limpa e pré-processa os dados da rede.
    - Converte colunas numéricas, tratando erros.
    - Remove linhas com dados em falta (NaN).
    - Re-indexa o DataFrame para garantir uma sequência contínua.
    - Classifica o tipo de aplicação de cada utilizador com base em métricas de rede.
    """
    df, err = load_5g_csv()
    if err:
        return None, err

    # Converte colunas para tipo numérico. 'coerce' transforma erros em NaN.
    df["DL_bitrate"] = pd.to_numeric(df["DL_bitrate"], errors="coerce")
    df["UL_bitrate"] = pd.to_numeric(df["UL_bitrate"], errors="coerce")
    ping_col = next((c for c in df.columns if c.lower().startswith("ping")), None)
    if not ping_col: return None, "Coluna de ping não encontrada."
    if "CQI" not in df.columns: return None, "Coluna 'CQI' não encontrada no CSV."

    df[ping_col] = pd.to_numeric(df[ping_col], errors="coerce")

    # Remove qualquer linha que contenha valores NaN nas colunas críticas.
    df = df.dropna(subset=["DL_bitrate", "UL_bitrate", ping_col, "CQI"])
    
    # Passo CRÍTICO: Re-indexa o DataFrame. Garante que os índices são sequenciais (0, 1, 2...),
    # evitando erros de 'IndexError' em loops posteriores após a remoção de linhas.
    df = df.reset_index(drop=True)

    def classify(row):
        """Define o tipo de aplicação com base nos requisitos de rede."""
        if row["DL_bitrate"] > 15 and row["UL_bitrate"] > 5 and row[ping_col] < 60: return "Video_Streaming"
        elif row["DL_bitrate"] > 3 and row["UL_bitrate"] > 1 and row[ping_col] < 80: return "Video_Call"
        elif row["DL_bitrate"] > 1 and row[ping_col] < 120: return "Web_Browse"
        elif row["UL_bitrate"] > 0.5 and row[ping_col] < 100: return "VoIP_Call"
        else: return "Background_Download"

    df["Application_Type"] = df.apply(classify, axis=1)
    if "User_ID" not in df.columns: df.insert(0, "User_ID", range(1, len(df) + 1))
    
    # Salva o dataframe classificado para fins de depuração ou análise futura.
    df.to_csv("5g_data-classification.csv", sep=";", index=False)
    return df, None

# ===================== SIMULAÇÃO E ALGORITMOS =====================
def run_full_simulation(pop, gens, mut_rate, elitism, progress_cb=None):
    """
    Executa a simulação completa do Algoritmo Genético e dos baselines.

    Args:
        pop (int): Tamanho da população do AG.
        gens (int): Número de gerações.
        mut_rate (float): Taxa de mutação.
        elitism (int): Número de indivíduos de elite a serem preservados.
        progress_cb (function, optional): Callback para reportar o progresso.
    """
    df, err = classify_and_prepare_data()
    if err: return {"error": err}

    # --- Configurações Iniciais da Simulação ---
    users = len(df)
    TOTAL_RBS = users * 8  # Total de Blocos de Recursos a serem alocados.

    # Dicionário de requisitos por aplicação: (DL ideal, DL mínimo, Categoria)
    REQ = {
        "Video_Streaming": (25, 15, "eMBB"), "Video_Call": (18, 14, "URLLC"),
        "Web_Browse": (10,  4, "eMBB"), "VoIP_Call": (10,  8, "URLLC"),
        "Background_Download": (15, 5, "eMBB")
    }

    # --- Funções Fundamentais do AG e Métricas ---

    def satisfaction(gene, app):
        """
        Calcula a satisfação de um único utilizador (gene).
        Esta função "suavizada" é crucial para o desempenho do AG, pois fornece
        um gradiente que guia a evolução, mesmo para soluções subótimas.
        """
        req, minr, cat = REQ.get(app, (10, 5, "eMBB"))
        if cat == "URLLC":
            # Para serviços críticos, recompensa a aproximação ao mínimo.
            return 1.0 if gene >= minr else 0.8 * (gene / minr)
        else: # eMBB
            return min(1.0, gene / req) if req > 0 else 1.0

    def jain_fairness_index(scores):
        """
        Calcula o Índice de Justiça de Jain, uma métrica padrão para equidade.
        O valor varia de 1/n (pior caso) a 1 (justiça perfeita).
        """
        scores_np = np.array(scores)
        # Trata o caso de divisão por zero se todos os scores forem 0.
        if np.sum(scores_np) == 0: return 1.0
        sum_of_scores = np.sum(scores_np)
        sum_of_squares = np.sum(scores_np**2)
        # Fórmula de Jain: (Σx)² / (n * Σx²)
        return (sum_of_scores**2) / (len(scores_np) * sum_of_squares)

    def analyse(alloc):
        """Avalia uma solução final (alocação) e retorna as métricas de desempenho."""
        scores = [satisfaction(alloc[i], df.iloc[i]["Application_Type"]) for i in range(users)]
        fairness = jain_fairness_index(scores)
        return np.mean(scores) * 100, fairness * 100

    def fit(ind):
        """
        Função de Aptidão (Fitness Function).
        Mede a "qualidade" de um indivíduo (solução). É uma função multi-objetivo
        ponderada, que guia o AG a encontrar soluções que são simultaneamente
        eficientes (alta satisfação) e justas (alto índice de Jain).
        """
        scores = [satisfaction(ind[i], df.iloc[i]["Application_Type"]) for i in range(users)]
        satisfaction_score = np.mean(scores)
        fairness_score = jain_fairness_index(scores)
        # Os pesos (0.7 e 0.3) podem ser ajustados para priorizar um objetivo sobre o outro.
        return (0.7 * satisfaction_score) + (0.3 * fairness_score)

    # --- Operadores Genéticos ---

    def create():
        """Cria um indivíduo aleatório e válido para a população inicial."""
        ind, rem = [0]*users, TOTAL_RBS
        # Itera sobre os utilizadores em ordem aleatória para evitar viés.
        for i in random.sample(range(users), users):
            val = min(random.randint(0, 16), rem)
            ind[i] = val; rem -= val
        return ind

    def cross(p1, p2):
        """
        Operador de Cruzamento (Crossover) com reparação.
        Combina dois pais para gerar um filho. Se o filho for inválido (exceder
        os recursos totais), um mecanismo de reparação é ativado para o tornar válido.
        """
        pt = random.randint(1, users - 1) # Ponto de corte
        child = p1[:pt] + p2[pt:]
        total_child_rbs = sum(child)
        if total_child_rbs > TOTAL_RBS:
            excess = total_child_rbs - TOTAL_RBS
            # Mecanismo de reparação: remove o excesso de utilizadores aleatórios.
            while excess > 0:
                reducible_users = [i for i, rbs in enumerate(child) if rbs > 0]
                if not reducible_users: break
                user_to_reduce = random.choice(reducible_users)
                amount_to_reduce = random.randint(1, child[user_to_reduce])
                reduction = min(amount_to_reduce, excess)
                child[user_to_reduce] -= reduction
                excess -= reduction
        return child
        
    def mutate(ind):
        """
        Operador de Mutação Híbrido.
        Primeiro, tenta uma mutação guiada (move recursos de utilizadores satisfeitos
        para insatisfeitos). Se não for possível, recorre a uma mutação aleatória
        para garantir a diversidade genética e evitar estagnação.
        """
        if random.random() < mut_rate:
            # Tentativa de mutação guiada
            scores = [satisfaction(ind[i], df.iloc[i]["Application_Type"]) for i in range(len(ind))]
            satisfied_users = [i for i, s in enumerate(scores) if s > 0.9 and ind[i] > 0]
            needy_users = [i for i, s in enumerate(scores) if s < 0.7]
            if satisfied_users and needy_users:
                donor, receiver = random.choice(satisfied_users), random.choice(needy_users)
                if ind[donor] > 0:
                    amount = random.randint(1, ind[donor])
                    ind[donor] -= amount; ind[receiver] += amount
                    return ind # Termina se a mutação guiada for bem-sucedida
            
            # Fallback: mutação aleatória para garantir exploração
            i, j = random.sample(range(users), 2)
            if ind[i] > 0:
                amt = random.randint(1, ind[i])
                ind[i] -= amt; ind[j] += amt
        return ind

    def tournament_selection(population, fits, k=3):
        """
        Seleção por Torneio. Seleciona `k` indivíduos aleatoriamente e
        retorna o melhor entre eles. Aumenta a pressão seletiva de forma controlada.
        """
        best_ix = None
        for _ in range(k):
            ix = random.randint(0, len(population)-1)
            if best_ix is None or fits[ix] > fits[best_ix]: best_ix = ix
        return population[best_ix]

    # --- Ciclo Evolutivo Principal ---
    popu = [create() for _ in range(pop)]
    best_hist, avg_hist = [], []

    for g in range(gens):
        fits = [fit(ind) for ind in popu]
        best_hist.append(np.max(fits)); avg_hist.append(np.mean(fits))
        
        # Ordena a população pela aptidão para facilitar a seleção de elites.
        sorted_by_fit = sorted(zip(fits, popu), key=lambda x: x[0], reverse=True)
        
        # Elitismo: os melhores indivíduos passam diretamente para a próxima geração.
        next_gen = [ind for fit_val, ind in sorted_by_fit[:elitism]]
        
        # Preenche o resto da nova geração com filhos dos melhores pais.
        while len(next_gen) < pop:
            p1 = tournament_selection(popu, fits)
            p2 = tournament_selection(popu, fits)
            next_gen.append(mutate(cross(p1, p2)))
        
        popu = next_gen
        # Reporta o progresso para a GUI, se o callback for fornecido.
        if progress_cb: progress_cb((g + 1) / gens)

    # --- Algoritmos de Baseline para Comparação ---
    ag_alloc = max(popu, key=fit)
    rr_alloc = [TOTAL_RBS // users] * users

    cqi_alloc = [0]*users; rem = TOTAL_RBS
    for i in df.sort_values("CQI", ascending=False).index:
        if rem <= 0: break
        need = REQ.get(df.loc[i, "Application_Type"], (10,5))[0]
        give = min(need, rem); cqi_alloc[i] = give; rem -= give

    df["pf"] = df["DL_bitrate"] / (df["CQI"] + 1)
    pf_alloc = [0]*users; rem = TOTAL_RBS
    for i in df.sort_values("pf", ascending=False).index:
        if rem <= 0: break
        need = REQ.get(df.loc[i, "Application_Type"], (10,5))[0]
        give = min(need, rem); pf_alloc[i] = give; rem -= give

    # --- Compilação e Formatação dos Resultados ---
    results = {
        "Alg. Genético": analyse(ag_alloc), "Round Robin": analyse(rr_alloc),
        "Best CQI": analyse(cqi_alloc), "Prop. Fair": analyse(pf_alloc)
    }
    table = "Algoritmo         | Satisfação | Justiça (Jain)\n" + "-"*45 + "\n"
    for k,(sat,fair) in results.items():
        table += f"{k:<17} | {sat:9.2f}% | {fair:12.2f}%\n"

    fig_cmp = Figure(figsize=(6,4), tight_layout=True)
    ax = fig_cmp.add_subplot(111)
    x = np.arange(len(results)); sats = [v[0] for v in results.values()]; fairs = [v[1] for v in results.values()]
    ax.bar(x-0.2, sats, 0.4, label="Satisfação (%)"); ax.bar(x+0.2, fairs, 0.4, label="Justiça (Jain) (%)")
    ax.set_xticks(x); ax.set_xticklabels(list(results.keys()), rotation=15, ha="right")
    ax.set_ylim(0,105); ax.legend()

    fig_evo = Figure(figsize=(6,3), tight_layout=True)
    ax2 = fig_evo.add_subplot(111)
    ax2.plot(best_hist, label="Aptidão Máx."); ax2.plot(avg_hist, label="Aptidão Média")
    ax2.set_title("Evolução da Aptidão"); ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.6)

    return {"text": table, "fig_comp": fig_cmp, "fig_evo": fig_evo, "error": None}

# ===================== INTERFACE GRÁFICA (GUI) =====================
class App(ctk.CTk):
    """
    Classe principal da aplicação, que gere a interface gráfica e a interação
    com o utilizador.
    """
    def __init__(self):
        super().__init__()
        self.title("Comparativo 5G – AG • RR • CQI • PF"); self.geometry("1280x800")
        ctk.set_appearance_mode("dark"); self.queue = queue.Queue()
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)

        # --- Sidebar para controlo da simulação ---
        side = ctk.CTkFrame(self, width=260, corner_radius=0)
        side.grid(row=0, column=0, sticky="ns"); side.grid_rowconfigure(10, weight=1)
        ctk.CTkLabel(side, text="Parâmetros AG", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20,10))
        
        # Entradas para os parâmetros do AG
        self.e_pop = ctk.CTkEntry(side); self.e_pop.insert(0,"100")
        self.e_gen = ctk.CTkEntry(side); self.e_gen.insert(0,"150")
        self.e_mut = ctk.CTkEntry(side); self.e_mut.insert(0,"0.1")
        self.e_elit= ctk.CTkEntry(side); self.e_elit.insert(0,"2")

        def add(label, widget, row):
            ctk.CTkLabel(side,text=label).grid(row=row, column=0, sticky="w", padx=20)
            widget.grid(row=row+1, column=0, padx=20, pady=(0,10), sticky="ew")
        add("Tamanho da População", self.e_pop, 1); add("Número de Gerações", self.e_gen, 3)
        add("Taxa de Mutação (0–1)", self.e_mut, 5); add("Número de Elites", self.e_elit,7)

        # Botão de início e widgets de status
        self.bt_start = ctk.CTkButton(side, text="▶ Iniciar Simulação", command=self.start)
        self.bt_start.grid(row=9, column=0, padx=20, pady=10, sticky="ew")
        self.lb_status = ctk.CTkLabel(side, text="Pronto para iniciar.", text_color="gray")
        self.lb_status.grid(row=10, column=0, padx=20, pady=(0,6), sticky="w")
        self.pbar = ctk.CTkProgressBar(side); self.pbar.set(0)
        self.pbar.grid(row=11, column=0, padx=20, pady=(0,20), sticky="ew")

        # --- Painel de Resultados ---
        res = ctk.CTkFrame(self); res.grid(row=0,column=1,sticky="nsew", padx=20, pady=20)
        res.grid_columnconfigure((0,1), weight=1); res.grid_rowconfigure(1, weight=1)
        self.txt = ctk.CTkTextbox(res, height=140, font=("Courier New",12))
        self.txt.grid(row=0,column=0,columnspan=2,sticky="ew", padx=10, pady=(10,0))
        self.txt.insert("0.0","Resultados aparecerão aqui."); self.txt.configure(state="disabled")
        self.g1 = ctk.CTkFrame(res); self.g1.grid(row=1,column=0,sticky="nsew", padx=10, pady=10)
        self.g2 = ctk.CTkFrame(res); self.g2.grid(row=1,column=1,sticky="nsew", padx=10, pady=10)
        
        # Inicia o loop de verificação da fila de comunicação com a thread.
        self.after(100, self.poll_queue)

    def start(self):
        """Inicia a simulação numa thread separada para não congelar a GUI."""
        try:
            # Validação dos parâmetros de entrada
            pop, gen, mut, elit = int(self.e_pop.get()), int(self.e_gen.get()), float(self.e_mut.get()), int(self.e_elit.get())
            assert 0 <= mut <= 1 and pop > 0 and gen > 0 and elit >= 0 and elit < pop
        except (ValueError, AssertionError):
            self.lb_status.configure(text="⚠ Parâmetros inválidos."); return
        
        self.bt_start.configure(state="disabled"); self.pbar.set(0)
        self.lb_status.configure(text="Executando... 0%")
        
        # A simulação é executada em uma thread para manter a GUI responsiva.
        # A comunicação é feita via `queue` para segurança entre threads.
        threading.Thread(target=lambda: self.queue.put(("done", run_full_simulation(pop,gen,mut,elit,lambda p: self.queue.put(("prog",p))))), daemon=True).start()

    def poll_queue(self):
        """Verifica a fila por mensagens da thread de simulação sem bloquear."""
        try:
            typ, data = self.queue.get_nowait()
            if typ == "prog":
                # Atualiza a barra e o texto de progresso.
                self.pbar.set(data)
                self.lb_status.configure(text=f"Executando... {int(data * 100)}%")
            
            elif typ == "done":
                # A simulação terminou. Apresenta os resultados ou um erro.
                if data.get("error"): self.lb_status.configure(text=f"❌ {data['error']}")
                else: self.show_results(data); self.lb_status.configure(text="✅ Simulação concluída.")
                self.bt_start.configure(state="normal"); self.pbar.set(1)
        except queue.Empty:
            # Se a fila estiver vazia, não faz nada.
            pass
        # Agenda a próxima verificação da fila.
        self.after(100, self.poll_queue)

    def show_results(self, data):
        """Exibe os resultados (texto e gráficos) na GUI."""
        self.txt.configure(state="normal"); self.txt.delete("0.0","end")
        self.txt.insert("0.0", data["text"]); self.txt.configure(state="disabled")
        
        # Limpa os frames antes de desenhar novos gráficos.
        for fr in (self.g1,self.g2):
            for w in fr.winfo_children(): w.destroy()
        
        # Incorpora os gráficos do Matplotlib nos frames do Tkinter.
        self._embed(data["fig_comp"], self.g1)
        self._embed(data["fig_evo"] , self.g2)
        
    def _embed(self, fig, frame):
        """Função auxiliar para incorporar uma figura Matplotlib num frame Tkinter."""
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw(); canvas.get_tk_widget().pack(fill="both", expand=True)

# ===================== PONTO DE ENTRADA DA APLICAÇÃO =====================
if __name__ == "__main__":
    # Cria a instância da aplicação e inicia o loop principal da GUI.
    app = App()
    app.mainloop()