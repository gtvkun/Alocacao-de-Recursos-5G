import customtkinter as ctk
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Definir o backend ANTES de importar pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import queue

# ----------------------------------------------------------------
# UTIL – carregar o CSV detectando separadores
# ----------------------------------------------------------------

def load_5g_csv(path="5g_data.csv"):
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if "DL_bitrate" in df.columns:
                return df, None
        except FileNotFoundError:
            return None, f"Arquivo '{path}' não encontrado."
    return None, "Coluna 'DL_bitrate' não encontrada no CSV."

def classify_and_prepare_data():
    df, err = load_5g_csv()
    if err:
        return None, err

    df["DL_bitrate"] = pd.to_numeric(df["DL_bitrate"], errors="coerce")
    df["UL_bitrate"] = pd.to_numeric(df["UL_bitrate"], errors="coerce")
    ping_col = next((c for c in df.columns if c.lower().startswith("ping")), None)
    if not ping_col:
        return None, "Coluna de ping não encontrada."

    df[ping_col] = pd.to_numeric(df[ping_col], errors="coerce")
    df = df.dropna(subset=["DL_bitrate", "UL_bitrate", ping_col])

    def classify(row):
        if row["DL_bitrate"] > 15 and row["UL_bitrate"] > 5 and row[ping_col] < 60:
            return "Video_Streaming"
        elif row["DL_bitrate"] > 3 and row["UL_bitrate"] > 1 and row[ping_col] < 80:
            return "Video_Call"
        elif row["DL_bitrate"] > 1 and row[ping_col] < 120:
            return "Web_Browse"
        elif row["UL_bitrate"] > 0.5 and row[ping_col] < 100:
            return "VoIP_Call"
        else:
            return "Background_Download"

    df["Application_Type"] = df.apply(classify, axis=1)
    if "User_ID" not in df.columns:
        df.insert(0, "User_ID", range(1, len(df) + 1))
    df.to_csv("5g_data-classification.csv", sep=";", index=False)
    return df, None

def run_full_simulation(pop, gens, mut_rate, elitism):
    df, error = classify_and_prepare_data()
    if error:
        return {"error": error}

    users = len(df)
    requests = df[["User_ID", "Application_Type", "CQI", "DL_bitrate"]].copy()
    requests.fillna(0, inplace=True)
    TOTAL_RBS = users * 8

    REQ = {
        "Video_Streaming": (25, 15, "eMBB"),
        "Video_Call": (18, 14, "URLLC"),
        "Web_Browse": (10, 4, "eMBB"),
        "VoIP_Call": (10, 8, "URLLC"),
        "Background_Download": (15, 5, "eMBB")
    }

    def satisfaction_score(gene, app):
        req, minr, cat = REQ.get(app, (10, 5, "eMBB"))
        if cat == "URLLC":
            return 1.0 if gene >= minr else 0.0
        return min(1.0, gene / req)

    def analyze(alloc):
        scores = [satisfaction_score(alloc[i], requests.iloc[i]["Application_Type"]) for i in range(users)]
        return np.mean(scores) * 100, (1 - np.std(scores)) * 100

    def create_individual():
        rem = TOTAL_RBS
        ind = [0] * users
        for i in random.sample(range(users), users):
            val = min(random.randint(0, 16), rem)
            ind[i] = val
            rem -= val
        return ind

    def fitness(ind):
        scores = [satisfaction_score(ind[i], requests.iloc[i]["Application_Type"]) for i in range(users)]
        return sum(scores) - sum(s < 0.5 for s in scores)

    def crossover(p1, p2):
        point = random.randint(1, users - 1)
        child = p1[:point] + p2[point:]
        total = sum(child)
        if total > TOTAL_RBS:
            child = [int(c * TOTAL_RBS / total) for c in child]
        return child
def mutate(ind):
    if random.random() < mut_rate:
        i, j = random.sample(range(len(ind)), 2)
        if ind[i] > 0:
            amt = random.randint(1, ind[i])
            ind[i] -= amt
            ind[j] += amt
    return ind


    popu = [create_individual() for _ in range(pop)]
    bests, avgs = [], []

    for _ in range(gens):
        fits = [fitness(ind) for ind in popu]
        bests.append(max(fits))
        avgs.append(sum(fits) / len(fits))
        elite = sorted(popu, key=fitness, reverse=True)[:elitism]
        while len(elite) < pop:
            p1, p2 = random.choices(popu, k=2)
            elite.append(mutate(crossover(p1, p2)))
        popu = elite

    ag_result = popu[np.argmax([fitness(ind) for ind in popu])]

    # Baselines
    rr_result = [TOTAL_RBS // users] * users
    cqi_sorted = requests.sort_values("CQI", ascending=False)
    cqi_alloc = [0] * users
    rem = TOTAL_RBS
    for idx in cqi_sorted.index:
        if rem <= 0: break
        cqi_alloc[idx] = min(REQ.get(requests.loc[idx]["Application_Type"], (10, 5))[0], rem)
        rem -= cqi_alloc[idx]

    pf_sorted = requests.copy()
    pf_sorted["PF_score"] = pf_sorted["DL_bitrate"] / (pf_sorted["CQI"] + 1)
    pf_sorted.sort_values("PF_score", ascending=False, inplace=True)
    pf_alloc = [0] * users
    rem = TOTAL_RBS
    for idx in pf_sorted.index:
        if rem <= 0: break
        pf_alloc[idx] = min(REQ.get(requests.loc[idx]["Application_Type"], (10, 5))[0], rem)
        rem -= pf_alloc[idx]

    results = {
        "Alg. Genético": analyze(ag_result),
        "Round Robin": analyze(rr_result),
        "Best CQI": analyze(cqi_alloc),
        "Prop. Fair": analyze(pf_alloc),
    }

    # Texto
    text = "Algoritmo         | Satisfação | Justiça\n"
    text += "-" * 40 + "\n"
    for name, (sat, fair) in results.items():
        text += f"{name:<17} | {sat:9.2f}% | {fair:7.2f}%\n"

    # Gráfico
    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    x = np.arange(len(results))
    sat = [v[0] for v in results.values()]
    fair = [v[1] for v in results.values()]
    ax.bar(x - 0.2, sat, width=0.4, label="Satisfação (%)")
    ax.bar(x + 0.2, fair, width=0.4, label="Justiça (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(results.keys()))
    ax.set_ylim(0, 100)
    ax.legend()

    fig2 = Figure(figsize=(6, 3))
    ax2 = fig2.add_subplot(111)
    ax2.plot(bests, label="Melhor Aptidão")
    ax2.plot(avgs, label="Média")
    ax2.set_title("Evolução da Aptidão")
    ax2.legend()

    return {"text": text, "fig_comp": fig, "fig_evo": fig2, "error": None}

# ----------------------------------------------------------------
# INTERFACE GRÁFICA
# ----------------------------------------------------------------

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Analisador de Dados 5G")
        self.geometry("1000x800")

        # --- Variáveis de estado ---
        self.df = None
        self.simulation_thread = None
        self.paused = False
        self.simulation_paused = threading.Event()
        self.simulation_paused.set()  # Inicia "desbloqueado"

        # --- Fila para comunicação entre threads ---
        self.queue = queue.Queue()

        # --- Listas para os dados do gráfico ---
        self.timestamps = []
        self.dl_bitrates = []
        self.cqi_values = []

        # --- Configurar a interface gráfica ---
        self.create_widgets()

    def create_widgets(self):
        """Cria e organiza todos os widgets na janela."""
        # Frame principal para os controlos
        control_frame = ctk.CTkFrame(self)
        control_frame.pack(pady=20, padx=20, fill="x")

        # Botão para carregar o ficheiro CSV
        self.btn_load = ctk.CTkButton(control_frame, text="Carregar CSV 5G", command=self.load_5g_csv)
        self.btn_load.pack(side="left", padx=10)

        # Botão para iniciar a simulação
        self.btn_start = ctk.CTkButton(control_frame, text="Iniciar Simulação", command=self.run_simulation_thread, state="disabled")
        self.btn_start.pack(side="left", padx=10)

        # Botão de Pausa/Retoma
        self.btn_pause = ctk.CTkButton(control_frame, text="Pausar", command=self.toggle_pause, state="disabled")
        self.btn_pause.pack(side="left", padx=10)

        # Slider de velocidade
        self.speed_label = ctk.CTkLabel(control_frame, text="Velocidade (s):")
        self.speed_label.pack(side="left", padx=(20, 5))
        self.speed_slider = ctk.CTkSlider(control_frame, from_=0.0, to=1.0, number_of_steps=100)
        self.speed_slider.set(0.1)
        self.speed_slider.pack(side="left", padx=5)
        
        # --- NOVO: Barra de Progresso ---
        # Rótulo para a barra de progresso
        self.progress_label = ctk.CTkLabel(self, text="Progresso da Simulação:")
        self.progress_label.pack(pady=(10, 0), padx=20, fill="x")

        # Widget da barra de progresso
        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal")
        self.progress_bar.set(0) # Inicia em 0%
        self.progress_bar.pack(pady=(5, 20), padx=20, fill="x", expand=False)
        # --- FIM DA ADIÇÃO ---

        # Gráfico
        self.create_plot()

    def create_plot(self):
        """Cria a área do gráfico Matplotlib."""
        self.figure, self.ax = plt.subplots(figsize=(12, 6))

        # Guarda as linhas como atributos para atualização eficiente
        self.line_dl, = self.ax.plot([], [], label="DL Bitrate (Mbps)", color='blue', marker='o', linestyle='-')
        self.line_cqi, = self.ax.plot([], [], label="CQI", color='red', marker='x', linestyle='--')

        self.ax.set_title("Desempenho da Rede 5G em Tempo Real")
        self.ax.set_xlabel("Timestamp")
        self.ax.set_ylabel("Valores")
        self.ax.legend()
        self.ax.grid(True)
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right")
        
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
        self.figure.tight_layout()

    def load_5g_csv(self):
        """Carrega e processa o ficheiro CSV."""
        filepath = ctk.filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return

        try:
            self.df = pd.read_csv(filepath)
            self.classify_and_prepare_data()
            print("CSV carregado e processado com sucesso.")
            
            # Limpa dados de simulações anteriores
            self.timestamps, self.dl_bitrates, self.cqi_values = [], [], []
            self.line_dl.set_data([], [])
            self.line_cqi.set_data([], [])
            self.canvas.draw()
            
            # --- NOVO: Reiniciar a barra de progresso ---
            self.progress_bar.set(0)

            # Ativa os botões
            self.btn_start.configure(state="normal")
            self.btn_pause.configure(state="normal")

        except Exception as e:
            print(f"Erro ao carregar o ficheiro: {e}")

    def classify_and_prepare_data(self):
        """Prepara os dados para a simulação."""
        if 'DL_bitrate' in self.df.columns:
            self.df['DL_bitrate'] = pd.to_numeric(self.df['DL_bitrate'], errors='coerce') / 1_000_000
        
        if 'CQI' in self.df.columns:
            self.df['CQI'] = pd.to_numeric(self.df['CQI'], errors='coerce')
            cqi_mean = self.df['CQI'].mean()
            self.df['CQI'].fillna(cqi_mean, inplace=True)
        
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], format='%Y.%m.%d_%H.%M.%S', errors='coerce')
        self.df.dropna(subset=['Timestamp'], inplace=True) # Remove linhas onde a data falhou
        self.df.sort_values(by='Timestamp', inplace=True)
    
    def run_simulation_thread(self):
        """Inicia a thread da simulação e a atualização do gráfico."""
        if self.df is None:
            print("Nenhum dado carregado.")
            return

        self.btn_start.configure(state="disabled") # Desativa o botão de iniciar
        self.simulation_thread = threading.Thread(target=self.start_simulation, daemon=True)
        self.simulation_thread.start()
        # Inicia a primeira chamada para a atualização do gráfico
        self.update_plot()
        
    def start_simulation(self):
        """
        Inicia a simulação percorrendo os dados do DataFrame de forma otimizada.
        """
        # Converte para uma lista de dicionários para iteração rápida
        data_records = self.df.to_dict('records')
        total_rows = len(data_records)

        for i, row in enumerate(data_records):
            if self.paused:
                self.simulation_paused.wait()

            simulation_speed = self.speed_slider.get()
            time.sleep(simulation_speed)
            
            # --- NOVO: Atualizar a barra de progresso ---
            progress_value = (i + 1) / total_rows
            self.progress_bar.set(progress_value)
            
            data_point = {
                'Timestamp': row.get('Timestamp'),
                'DL_bitrate': row.get('DL_bitrate'),
                'CQI': row.get('CQI'),
                'Operatorname': row.get('Operatorname')
            }
            self.queue.put(data_point)

        self.queue.put(None) # Sinal de fim de simulação

    def update_plot(self):
        """
        Atualiza o gráfico de forma eficiente, consumindo dados da fila.
        """
        try:
            while not self.queue.empty():
                data = self.queue.get_nowait()
                if data is None:
                    print("Simulação terminada.")
                    self.btn_start.configure(state="normal") # Reativa o botão de iniciar
                    return

                self.timestamps.append(data['Timestamp'])
                self.dl_bitrates.append(data['DL_bitrate'])
                self.cqi_values.append(data['CQI'])

            # Limita a visualização aos últimos 50 pontos
            if len(self.timestamps) > 50:
                self.timestamps.pop(0)
                self.dl_bitrates.pop(0)
                self.cqi_values.pop(0)
            
            # Atualiza os dados das linhas do gráfico
            self.line_dl.set_data(self.timestamps, self.dl_bitrates)
            self.line_cqi.set_data(self.timestamps, self.cqi_values)

            # Reajusta os limites e redesenha o canvas
            self.ax.relim()
            self.ax.autoscale_view()
            self.figure.tight_layout()
            self.canvas.draw()
            
        except queue.Empty:
            pass # Fila vazia, espera pela próxima chamada

        finally:
            self.after(1000, self.update_plot)

    def toggle_pause(self):
        """Pausa ou retoma a simulação."""
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.configure(text="Retomar")
            self.simulation_paused.clear() # Bloqueia a thread
        else:
            self.btn_pause.configure(text="Pausar")
            self.simulation_paused.set() # Desbloqueia a thread
# ----------------------------------------------------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
