import customtkinter as ctk
import pandas as pd
import random
import numpy as np
import time
import threading
import queue

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ========================================================================================
# PARTE 1: LÓGICA DO ALGORITMO (Adaptada para receber parâmetros da GUI)
# ========================================================================================

# A assinatura da função foi alterada para receber os parâmetros
def run_full_simulation(POPULATION_SIZE, NUM_GENERATIONS, MUTATION_RATE, ELITISM_COUNT):
    """
    Esta função encapsula toda a lógica da simulação e retorna os resultados.
    """
    # --- Configuração ---
    try:
        df = pd.read_csv('Quality of Service 5G.csv', sep=';')
    except FileNotFoundError:
        return {'error': "Arquivo 'Quality of Service 5G.csv' não encontrado."}
    
    NUM_USUARIOS = len(df)
    requests = df[['User_ID', 'Application_Type']].to_dict('records')
    RBS_MEDIO_POR_USUARIO = 8
    TOTAL_RBS = NUM_USUARIOS * RBS_MEDIO_POR_USUARIO

    # Os parâmetros do AG agora vêm da GUI, não são mais fixos aqui.
    TOURNAMENT_SIZE = 5

    SERVICE_REQUIREMENTS = {
        'Video_Streaming':   {'required_rbs': 25, 'min_rbs': 15, 'category': 'eMBB'},
        'Streaming':         {'required_rbs': 20, 'min_rbs': 10, 'category': 'eMBB'},
        'File_Download':     {'required_rbs': 18, 'min_rbs': 8,  'category': 'eMBB'},
        'Background_Download':{'required_rbs': 15, 'min_rbs': 5,  'category': 'eMBB'},
        'Web_Browse':      {'required_rbs': 10, 'min_rbs': 4,  'category': 'eMBB'},
        'Online_Gaming':     {'required_rbs': 15, 'min_rbs': 12, 'category': 'URLLC'},
        'Emergency_Service': {'required_rbs': 18, 'min_rbs': 15, 'category': 'URLLC'},
        'Voice_Call':        {'required_rbs': 10, 'min_rbs': 8,  'category': 'URLLC'},
        'VoIP_Call':         {'required_rbs': 10, 'min_rbs': 8,  'category': 'URLLC'},
        'Video_Call':        {'required_rbs': 18, 'min_rbs': 14, 'category': 'URLLC'},
        'IoT_Temperature':   {'required_rbs': 5, 'min_rbs': 2, 'category': 'mMTC'}
    }
    
    # --- Funções do AG (sem alterações na lógica interna) ---
    def create_individual():
        individual = [0] * NUM_USUARIOS
        remaining_rbs = TOTAL_RBS
        user_indices = list(range(NUM_USUARIOS))
        random.shuffle(user_indices)
        for i in user_indices:
            if remaining_rbs > 0:
                allocation = random.randint(0, int(RBS_MEDIO_POR_USUARIO * 2))
                allocation = min(allocation, remaining_rbs)
                individual[i] = allocation
                remaining_rbs -= allocation
        return individual

    def calculate_fitness(individual, requests):
        if sum(individual) > TOTAL_RBS: return -1000
        SATISFACTION_THRESHOLD = 0.50
        PENALTY_FACTOR = 1.0
        total_satisfaction = 0
        penalty_count = 0
        for i in range(NUM_USUARIOS):
            app_type = requests[i]['Application_Type']
            if app_type not in SERVICE_REQUIREMENTS:
                requirements = {'required_rbs': 10, 'min_rbs': 5, 'category': 'eMBB'}
            else:
                requirements = SERVICE_REQUIREMENTS[app_type]
            allocated_rbs = individual[i]
            satisfaction_score = 0.0
            if requirements['category'] == 'URLLC':
                if allocated_rbs >= requirements['min_rbs']: satisfaction_score = 1.0
            else:
                satisfaction_score = min(1.0, allocated_rbs / requirements['required_rbs'])
            if satisfaction_score < SATISFACTION_THRESHOLD:
                penalty_count += 1
            total_satisfaction += satisfaction_score
        final_fitness = total_satisfaction - (PENALTY_FACTOR * penalty_count)
        return final_fitness

    def selection(population, fitnesses):
        tournament = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
        return max(tournament, key=lambda x: x[1])[0]

    def crossover(parent1, parent2):
        crossover_point = random.randint(1, NUM_USUARIOS - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        child_sum = sum(child)
        if child_sum > TOTAL_RBS and child_sum > 0:
            correction_factor = TOTAL_RBS / child_sum
            child = [int(gene * correction_factor) for gene in child]
        return child

    def mutate(individual):
        if random.random() < MUTATION_RATE:
            user1, user2 = random.sample(range(NUM_USUARIOS), 2)
            if individual[user1] > 0:
                mutation_amount = random.randint(1, individual[user1])
                individual[user1] -= mutation_amount
                individual[user2] += mutation_amount
        return individual
    
    # --- Execução do AG ---
    start_time = time.time()
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    best_fitness_history, avg_fitness_history = [], []
    for generation in range(NUM_GENERATIONS):
        fitnesses = [calculate_fitness(ind, requests) for ind in population]
        best_fitness_history.append(max(fitnesses))
        avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
        new_population.extend(sorted_population[:ELITISM_COUNT])
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population
    
    end_time = time.time()
    ga_duration = end_time - start_time
    final_fitnesses = [calculate_fitness(ind, requests) for ind in population]
    ga_solution = population[final_fitnesses.index(max(final_fitnesses))]

    # ... (O restante da lógica de benchmark e análise permanece o mesmo) ...
    def schedule_round_robin(total_rbs, num_users):
        if num_users == 0: return []
        rbs_per_user = total_rbs // num_users
        return [rbs_per_user] * num_users

    def schedule_greedy(total_rbs, requests):
        indexed_requests = []
        for i, req in enumerate(requests):
            app_type = req['Application_Type']
            req['needed'] = SERVICE_REQUIREMENTS.get(app_type, {'required_rbs': 10})['required_rbs']
            req['original_index'] = i
            indexed_requests.append(req)
        sorted_requests = sorted(indexed_requests, key=lambda x: x['needed'])
        allocation = [0] * len(requests)
        remaining_rbs = total_rbs
        for req in sorted_requests:
            if remaining_rbs >= req['needed']:
                allocation[req['original_index']] = req['needed']
                remaining_rbs -= req['needed']
        return allocation
        
    round_robin_solution = schedule_round_robin(TOTAL_RBS, NUM_USUARIOS)
    greedy_solution = schedule_greedy(TOTAL_RBS, requests)

    def analyze_solution(allocation, requests):
        satisfactions = []
        for i in range(len(requests)):
            app_type = requests[i]['Application_Type']
            reqs = SERVICE_REQUIREMENTS.get(app_type, {'required_rbs': 10, 'min_rbs': 5, 'category': 'eMBB'})
            score = 0.0
            if reqs['category'] == 'URLLC':
                if allocation[i] >= reqs['min_rbs']: score = 1.0
            else:
                score = min(1.0, allocation[i] / reqs['required_rbs'])
            satisfactions.append(score)
        avg_satisfaction = (sum(satisfactions) / len(satisfactions)) * 100 if satisfactions else 0
        fairness = (1.0 - np.std(satisfactions)) * 100
        return {'satisfaction': avg_satisfaction, 'fairness': fairness}

    ga_results = analyze_solution(ga_solution, requests)
    rr_results = analyze_solution(round_robin_solution, requests)
    greedy_results = analyze_solution(greedy_solution, requests)

    report_text = f"Simulação concluída em {ga_duration:.2f} segundos para {NUM_USUARIOS} usuários.\n\n"
    report_text += "="*40 + "\nANÁLISE COMPARATIVA DOS ALGORITMOS\n" + "="*40 + "\n\n"
    results_data = {"Algoritmo Genético": ga_results, "Round Robin": rr_results, "Greedy (Ganancioso)": greedy_results}
    report_text += f"{'Algoritmo':<22} | {'Satisfação Média':<20} | {'Índice de Justiça':<20}\n" + "-"*70 + "\n"
    for name, result in results_data.items():
        report_text += f"{name:<22} | {result['satisfaction']:>19.2f}% | {result['fairness']:>19.2f}%\n"
    
    plt.style.use('seaborn-v0_8-darkgrid')
    fig_comp = Figure(figsize=(8, 4), dpi=100)
    ax1 = fig_comp.add_subplot(111)
    algorithms = list(results_data.keys())
    satisfactions = [r['satisfaction'] for r in results_data.values()]
    fairness_scores = [r['fairness'] for r in results_data.values()]
    x = np.arange(len(algorithms))
    width = 0.35
    rects1 = ax1.bar(x - width/2, satisfactions, width, label='Satisfação Média (%)', color='#4A90E2')
    rects2 = ax1.bar(x + width/2, fairness_scores, width, label='Justiça (%)', color='#50E3C2')
    ax1.set_ylabel('Pontuação (%)')
    ax1.set_title('Comparação de Desempenho dos Algoritmos')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms); ax1.legend()
    ax1.bar_label(rects1, padding=3, fmt='%.1f'); ax1.bar_label(rects2, padding=3, fmt='%.1f')
    fig_comp.tight_layout()

    fig_evo = Figure(figsize=(8, 4), dpi=100)
    ax2 = fig_evo.add_subplot(111)
    ax2.plot(best_fitness_history, label='Melhor Aptidão por Geração', color='#D0021B')
    ax2.plot(avg_fitness_history, label='Aptidão Média por Geração', linestyle='--', color='#4A90E2')
    ax2.set_title('Evolução da Aptidão do Algoritmo Genético'); ax2.set_xlabel('Geração')
    ax2.set_ylabel('Aptidão'); ax2.legend(); fig_evo.tight_layout()

    return {'text': report_text, 'fig_comp': fig_comp, 'fig_evo': fig_evo, 'error': None}


# ========================================================================================
# PARTE 2: APLICAÇÃO GRÁFICA (CustomTkinter)
# ========================================================================================

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Otimizador de Rede 5G com Algoritmo Genético")
        self.geometry("1200x750")
        ctk.set_appearance_mode("dark")

        self.results_queue = queue.Queue()
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Painel de Controle (Esquerda) ---
        self.control_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.control_frame.grid(row=0, column=0, rowspan=2, sticky="nswe")
        self.control_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.control_frame, text="Controles", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.start_button = ctk.CTkButton(self.control_frame, text="Iniciar Simulação", command=self.start_simulation_thread)
        self.start_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        self.reset_button = ctk.CTkButton(self.control_frame, text="Resetar", command=self.reset_ui, state="disabled")
        self.reset_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.status_label = ctk.CTkLabel(self.control_frame, text="Pronto para iniciar.", wraplength=200, justify="left")
        self.status_label.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        # --- NOVO: Frame para os parâmetros editáveis ---
        self.params_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        self.params_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")
        self.params_frame.grid_columnconfigure(1, weight=1)

        self.params_label = ctk.CTkLabel(self.params_frame, text="Parâmetros do AG:", font=ctk.CTkFont(weight="bold"))
        self.params_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")
        
        # Campo: Tamanho da População
        self.pop_size_label = ctk.CTkLabel(self.params_frame, text="Tam. População")
        self.pop_size_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.pop_size_entry = ctk.CTkEntry(self.params_frame)
        self.pop_size_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Campo: Número de Gerações
        self.num_gen_label = ctk.CTkLabel(self.params_frame, text="Nº de Gerações")
        self.num_gen_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.num_gen_entry = ctk.CTkEntry(self.params_frame)
        self.num_gen_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Campo: Taxa de Mutação
        self.mutation_rate_label = ctk.CTkLabel(self.params_frame, text="Taxa de Mutação")
        self.mutation_rate_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.mutation_rate_entry = ctk.CTkEntry(self.params_frame)
        self.mutation_rate_entry.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        # Campo: Contagem de Elitismo
        self.elitism_count_label = ctk.CTkLabel(self.params_frame, text="Elitismo")
        self.elitism_count_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.elitism_count_entry = ctk.CTkEntry(self.params_frame)
        self.elitism_count_entry.grid(row=4, column=1, padx=10, pady=5, sticky="ew")

        self.set_default_params() # Preenche os campos com valores padrão

        # --- Painel de Resultados (Direita) ---
        self.results_frame = ctk.CTkFrame(self, corner_radius=10)
        self.results_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nswe")
        self.results_frame.grid_columnconfigure(0, weight=1); self.results_frame.grid_columnconfigure(1, weight=1)
        self.results_frame.grid_rowconfigure(1, weight=1)

        self.textbox = ctk.CTkTextbox(self.results_frame, height=150, activate_scrollbars=True)
        self.textbox.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        self.textbox.insert("0.0", "Os resultados da simulação aparecerão aqui...")
        self.textbox.configure(state="disabled")

        self.graph_frame1 = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        self.graph_frame1.grid(row=1, column=0, padx=10, pady=10, sticky="nswe")
        
        self.graph_frame2 = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        self.graph_frame2.grid(row=1, column=1, padx=10, pady=10, sticky="nswe")
        
    def set_default_params(self):
        """Preenche os campos de entrada com valores padrão."""
        self.pop_size_entry.delete(0, "end"); self.pop_size_entry.insert(0, "100")
        self.num_gen_entry.delete(0, "end"); self.num_gen_entry.insert(0, "150")
        self.mutation_rate_entry.delete(0, "end"); self.mutation_rate_entry.insert(0, "0.1")
        self.elitism_count_entry.delete(0, "end"); self.elitism_count_entry.insert(0, "2")
        
    def start_simulation_thread(self):
        """Inicia a simulação em uma nova thread para não travar a GUI."""
        self.reset_ui(reset_params=False)
        
        # --- NOVO: Validação e Leitura dos Parâmetros ---
        try:
            pop_size = int(self.pop_size_entry.get())
            num_gen = int(self.num_gen_entry.get())
            mut_rate = float(self.mutation_rate_entry.get())
            elitism = int(self.elitism_count_entry.get())
            
            if not (0 <= mut_rate <= 1 and pop_size > 0 and num_gen > 0 and elitism >= 0):
                raise ValueError("Parâmetro fora do intervalo válido.")

        except ValueError:
            self.status_label.configure(text="Erro: Parâmetros inválidos. Verifique os valores e tente novamente.")
            return # Para a execução se houver erro

        self.status_label.configure(text="Executando simulação... Isso pode levar alguns minutos.")
        self.start_button.configure(state="disabled"); self.reset_button.configure(state="disabled")

        # Passa os parâmetros para a thread
        thread = threading.Thread(target=self.run_simulation_in_thread, args=(pop_size, num_gen, mut_rate, elitism))
        thread.daemon = True
        thread.start()
        
        self.after(100, self.process_queue)

    def run_simulation_in_thread(self, pop_size, num_gen, mut_rate, elitism):
        """Função que será executada na thread. Chama a lógica principal e põe o resultado na fila."""
        results = run_full_simulation(pop_size, num_gen, mut_rate, elitism)
        self.results_queue.put(results)

    def process_queue(self):
        try:
            results = self.results_queue.get_nowait()
            if results.get('error'):
                self.status_label.configure(text=f"Erro: {results['error']}")
                self.start_button.configure(state="normal")
            else:
                self.display_results(results)
                self.status_label.configure(text="Simulação Concluída!")
                self.start_button.configure(state="normal")
                self.reset_button.configure(state="normal")
        except queue.Empty:
            self.after(100, self.process_queue)
            
    def display_results(self, results):
        self.textbox.configure(state="normal")
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", results['text'])
        self.textbox.configure(state="disabled")
        self.embed_matplotlib_figure(results['fig_comp'], self.graph_frame1)
        self.embed_matplotlib_figure(results['fig_evo'], self.graph_frame2)

    def embed_matplotlib_figure(self, fig, frame):
        for widget in frame.winfo_children(): widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(side=ctk.TOP, fill=ctk.BOTH, expand=True)
        return canvas

    def reset_ui(self, reset_params=True):
        """Limpa a área de resultados para uma nova simulação."""
        self.status_label.configure(text="Pronto para iniciar.")
        self.reset_button.configure(state="disabled")
        if reset_params: self.set_default_params()
        self.textbox.configure(state="normal")
        self.textbox.delete("0.0", "end")
        self.textbox.insert("0.0", "Os resultados da simulação aparecerão aqui...")
        self.textbox.configure(state="disabled")
        for frame in [self.graph_frame1, self.graph_frame2]:
            for widget in frame.winfo_children():
                widget.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()