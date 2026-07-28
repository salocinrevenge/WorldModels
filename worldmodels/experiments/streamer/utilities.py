from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
import plotly
import os
import folium
from datetime import datetime
from scipy.cluster.hierarchy import fcluster
try:# execute as module:
    from src.algorithms import linkage_adjacent_ward, fcluster_custom
except:
    from algorithms import linkage_adjacent_ward, fcluster_custom

from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import json

def dendrograma(Z):
    dendrogram(Z)
    plt.gca().get_xaxis().set_visible(False)
    plt.show()

def plotar_completo(X_embedded_latent, labels, title = 'T-SNE dado completo', show = True, names = None, colors = None, ncols = 3, legend = True):

    # imita o funcionamento da funcao acima
    intervalo_x = [np.min(X_embedded_latent[:,0]), np.max(X_embedded_latent[:,0])]
    intervalo_y = [np.min(X_embedded_latent[:,1]), np.max(X_embedded_latent[:,1])]

    if names is None:
        names = {0: 'WALKING', 1: 'WALKING UPSTAIRS', 2: 'WALKING DOWNSTAIRS', 3: 'SITTING', 4: 'STANDING', 5: 'LAYING', 6: 'RUN', 7: 'Bicycle', 8: 'Table', 9: 'Plato'} #, 6: 'RUNNING'}
    # colors = {0: 'darkorange', 1: 'magenta', 2: 'purple', 3: 'blue', 4: 'green', 5: 'brown'}#, 6: 'red'}
    if colors is None:
        colors = {0: '#f08c19', 1: '#f39be7', 2: '#aa1fd1', 3: '#2fe9e6', 4: '#07a627', 5: '#963603', 6: '#ff0000', 7: '#f7fb28', 8: '#a8a8a8', 9: '#f9d2fe'}

    labels = np.array([colors[labels[i]] for i in range(len(labels))])

    # apenas a borda
    plt.scatter(X_embedded_latent[:,0], X_embedded_latent[:,1], c=labels, alpha=0.5, s=5)# s =2
    for i in range(len(names)):
        plt.scatter([], [], c=colors[i], label=names[i])
    
    if legend:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=ncols)
    plt.xlim(intervalo_x)
    plt.ylim(intervalo_y)
    if title:
        plt.title(title)
    if show:
        plt.show()

def plot_with_time(X_embedded_latent, title='', colors=None, show_colorbar=True):
    if colors is None:
        # Criando um gradiente de cores baseado no número de pontos
        num_points = len(X_embedded_latent)
        cmap = plt.cm.rainbow  # Escolhendo um colormap
        colors = cmap(np.linspace(0, 1, num_points))  # Criando a escala de cores

    scatter = plt.scatter(X_embedded_latent[:, 0], X_embedded_latent[:, 1], c=np.arange(len(X_embedded_latent)), cmap=plt.cm.rainbow, alpha=0.5, s=10)
    plt.title(title)

    # Criando a colorbar
    if show_colorbar:
        cbar = plt.colorbar(scatter)
        cbar.set_label("Índice do Ponto")

    plt.show()


def only_plot(X_embedded_latent, labels, title = None, show = True, palette = 'viridis', gamma = 0.5):
    plt.scatter(X_embedded_latent[:,0], X_embedded_latent[:,1], c=labels, cmap=palette, alpha=gamma)
    if title:
        plt.title(title)
    if show:
        plt.show()



class ScanerTree:
    def __init__(self, labels, Z) -> None:
        self.labels = labels
        self.Z = Z
        self.n_folhas_unitarias = int(Z[-1, 3])

    def recursive(self, x):
        if x < self.n_folhas_unitarias:
            self.dict_counter[self.labels[x]] += 1
            self.counter += 1
            return
        else:
            x = int(x)
            left = int(self.Z[x-self.n_folhas_unitarias, 0])
            right = int(self.Z[x-self.n_folhas_unitarias, 1])
            self.recursive(left)
            self.recursive(right)

    def __call__(self, x):
        self.dict_counter = {label: 0 for label in set(self.labels)}
        self.counter = 0
        self.recursive(x)

class TextLabel(ScanerTree):
    def __init__(self, labels, Z, names = None, remove_zero = False, show_ID = False):
        super().__init__(labels, Z)
        self.names = names
        self.remove_zero = remove_zero
        self.show_ID = show_ID

    def format(self, counter, dict_counter, id):
        if self.names:
            # translate the keys of dict_counter to the names
            dict_counter = {self.names[key]: value for key, value in self.dict_counter.items()}

            if self.remove_zero:
                # remove the keys and values that are zero
                dict_counter = {key: value for key, value in dict_counter.items() if value != 0}

        if self.show_ID:
            texto = f"ID: {id} \n {counter} \n {dict_counter}"
        else:
            texto = f"{counter} \n {dict_counter}"
        texto = texto.split(",")
        return "\n".join(texto).replace("{", "").replace("}", "").replace("'", "")

    def __call__(self, x):
        super().__call__(x)
        return self.format(self.counter, self.dict_counter, x)

class ColorLabel(ScanerTree):
    def __init__(self, labels, Z, colors = None):
        super().__init__(labels, Z)
        self.colors = colors

    def __call__(self, x):
        super().__call__(x)
        if max(self.dict_counter.values()) - sorted(self.dict_counter.values())[-2] > 0.2*max(self.dict_counter.values()):
            return self.colors[max(self.dict_counter, key=self.dict_counter.get)]
        else:
            return 'gray'

def plot_dendro_groups(Z, Y, num_groups, names = None, colors = None, size = (30, 10), remove_zero = False, show_ID = False):
    texter = TextLabel(Y, Z, names, remove_zero, show_ID)
    colorer = ColorLabel(Y, Z, colors)
    plt.figure(figsize=size)
    if colors:
        dendrogram(Z, color_threshold=0, truncate_mode='lastp', p=num_groups, labels=Y, leaf_label_func=texter, link_color_func=colorer)
        # color legend
        for i in range(len(names)):
            plt.scatter([], [], c=colors[i], label=names[i])
        plt.legend(loc='upper center', bbox_to_anchor=(0.85, 1.0), fancybox=True, shadow=True, ncol=3)
    else:
        dendrogram(Z, color_threshold=0, truncate_mode='lastp', p=num_groups, labels=Y, leaf_label_func=texter)
    plt.show()



import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots



class GenericScannerTree:
    def __init__(self, Z, leaf_function = None, start_call_function = None, stem_function = None) -> None:
        self.Z = Z
        self.n_folhas_unitarias = int(Z[-1, 3])
        self.leaf_function = leaf_function
        self.start_call_function = start_call_function
        self.stem_function = stem_function

    def recursive(self, x):
        if x < self.n_folhas_unitarias:
            if self.leaf_function:
                self.leaf_function(x)
            self.counter += 1
            return
        else:
            x = int(x)
            if self.stem_function:
                self.stem_function(x)
            left = int(self.Z[x-self.n_folhas_unitarias, 0])
            right = int(self.Z[x-self.n_folhas_unitarias, 1])
            self.recursive(left)
            self.recursive(right)

    def __call__(self, x):
        if self.start_call_function:
            self.start_call_function(x)
        self.counter = 0
        self.recursive(x)

class Anotator:
    def __init__(self, impedir_repeticao = False) -> None:
        if impedir_repeticao:
            self.caderno = set()
        else:
            self.caderno = []

    def __call__(self, x):
        if type(self.caderno) == set:
            self.caderno.add(x)
        else:
            self.caderno.append(x)
        
def get_n_ids(Z, n):
    n_leafs = int(Z[-1, 3])

    ids = set()

    idx_max = n_leafs + len(Z)
    idx_atual = idx_max
    i = len(Z) - 1
    while len(ids) < n:
        if len(ids) == n:
            break
        if idx_atual in ids:
            ids.remove(idx_atual)
        if Z[i,0]<= idx_max - n:
            ids.add(int(Z[i, 0]))
        if Z[i,1]<= idx_max - n:
            ids.add(int(Z[i, 1]))
        idx_atual -= 1
        i -=1
    return ids

def mostrar_series_presents(series, atividades, presentes = None, names = None, cores = None, cores_bg = None, atividades_nome = None, cores_presentes = None, title = "Visualização Interativa da Série Temporal", opacity = 0.2, maximo = 60, minimo = -60, turn_off_black_lines = False, plot_red_rects = False):
    """
    Mostra as series temporais passadas como parametro,
    Mostra também o fundo colorido de acordo com as atividades de cada janela,
    mostra a legenda com as atividades,
    e mostra uma faixa vermelha para destacar as amostras presentes
    """

    fig = go.Figure()

    times_stamps = np.arange(len(series[0]))

    if names is None:
        names = range(len(series))
    if cores is None:
        cores = [f"rgba(99, 110, 250, {opacity})", f"rgba(239, 85, 59, {opacity})", f"rgba(0, 204, 150, {opacity})", f"rgba(171, 99, 250, {opacity})", f"rgba(255, 161, 90, {opacity})", f"rgba(25, 211, 243, {opacity})", f"rgba(255, 102, 146, {opacity})", f"rgba(182, 232, 128, {opacity})", f"rgba(255, 151, 255, {opacity})", f"rgba(254, 203, 82, {opacity})"]
    if cores_presentes is None:
        cores_presentes = plotly.colors.qualitative.Plotly
        
    if cores_bg is None:
        # {0: '#f08c19', 1: '#f39be7', 2: '#aa1fd1', 3: '#2fe9e6', 4: '#07a627', 5: '#963603', 6: '#ff0000', 7: '#f7fb28', 8: '#a8a8a8', 9: '#f9d2fe'}
        # cores_bg = ["rgba(255, 150, 50, 0.6)", "rgba(225, 150, 250, 0.6)", "rgba(75, 0, 150, 0.6)", "rgba(50, 150, 225, 0.6)", "rgba(50, 120, 30, 0.6)", "rgba(150, 50, 25, 0.6)"]
        cores_bg = ["rgba(240, 140, 25, 0.6)", "rgba(243, 155, 231, 0.6)", "rgba(170, 31, 209, 0.6)", "rgba(47, 233, 230, 0.6)", "rgba(7, 166, 39, 0.6)", "rgba(150, 54, 3, 0.6)", "rgba(255, 0, 0, 0.6)", "rgba(247, 251, 40, 0.6)", "rgba(168, 168, 168, 0.6)", "rgba(249, 210, 254, 0.6)"]
    if atividades_nome is None:
        atividades_nome = {0: 'WALKING', 1: 'WALKING UPSTAIRS', 2: 'WALKING DOWNSTAIRS', 3: 'SITTING', 4: 'STANDING', 5: 'LAYING', 6: 'RUN', 7: 'Bicycle', 8: 'Table', 9: 'Plato'}

    for i in range(len(series)):
        fig.add_trace(go.Scatter(x=times_stamps, y=series[i], mode='lines', name=names[i], line=dict(color=cores[i])))

    # build background regions
    regions = []
    anterior = int(atividades[0])
    i_inicio = 0
    for i in range(0, len(atividades)):
        if anterior != atividades[i]:
            # print(f"{i_inicio=} {i=} {anterior=}")
            # print(f"{cores_bg[anterior]=}")
            regions.append({"start": i_inicio, "end": i, "color": cores_bg[anterior]})
            i_inicio = i
            anterior = int(atividades[i])
    regions.append({"start": i_inicio, "end": len(atividades), "color": cores_bg[anterior]})

    for region in regions:
        fig.add_shape( type="rect", x0=region["start"], x1=region["end"], y0= minimo, y1=maximo, fillcolor=region["color"], opacity=0.5, line_width=0)

    # add subtitle
    for i in atividades_nome.keys():
        fig.add_trace(go.Scatter( x=[None], y=[None], mode='markers', marker=dict(size=10, color=cores_bg[i]), name=atividades_nome[i] ))
    
    # add present samples
    if presentes is not None:
        sequencias_continuas = []
        sequencia_atual = []
        for i in range(0, len(presentes)-1):
            sequencia_atual.append(presentes[i])
            if presentes[i] != presentes[i+1] - 1:
                sequencias_continuas.append(sequencia_atual)
                sequencia_atual = []
        sequencia_atual.append(presentes[-1])
        sequencias_continuas.append(sequencia_atual)

    if presentes is not None:
        # cria retangulos vermelhos com base em sequencias_continuas
        regions = []
        for sequencia in sequencias_continuas:
            for i in range(len(series)):
                # desenha uma reta preta vertical no começo e no fim da sequencia
                if not turn_off_black_lines:
                    fig.add_trace(go.Scatter(x=[sequencia[0], sequencia[0]], y=[-1.5, 1.5], mode='lines', line=dict(color='black'), showlegend=False))
                # sequencia
                fig.add_trace(go.Scatter(x=times_stamps[sequencia], y=series[i][sequencia], mode='lines', line=dict(color=cores_presentes[i]), showlegend=False))
                # traço vertical no final da sequencia
                if not turn_off_black_lines:
                    fig.add_trace(go.Scatter(x=[sequencia[-1], sequencia[-1]], y=[-1.5, 1.5], mode='lines', line=dict(color='black'), showlegend=False))

        if plot_red_rects:
            # cria retangulos vermelhos com base em sequencias_continuas
            regions = []
            for sequencia in sequencias_continuas:
                regions.append({"start": sequencia[0], "end": sequencia[-1], "color": "rgba(255, 0, 0, 1)"})

            for region in regions:
                fig.add_shape( type="rect", x0=region["start"], x1=region["end"], y0= maximo, y1=maximo+(maximo-minimo)/60, fillcolor=region["color"], opacity=1, line_width=0)
        
    fig.update_layout(title= title, xaxis_title="Tempo", yaxis_title="Valores", template="plotly_white", height=600, showlegend=True)
    fig.show()

def mostrar_series_presents_plt(series, atividades, time_stamps, presentes = None, names = None, cores = None, cores_bg = None, atividades_nome = None, cores_presentes = None, title = "Visualização Interativa da Série Temporal", opacity = 0.2, maximo = 60, minimo = -60, turn_off_black_lines = False, plot_red_rects = False, utc = -3, episodes = None):
    """
    Mostra as series temporais passadas como parametro,
    Mostra também o fundo colorido de acordo com as atividades de cada janela,
    mostra a legenda com as atividades,
    e mostra uma faixa vermelha para destacar as amostras presentes
    Mostra também o tipo de episodio em cada momento se fornecido em episodes
    """

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.suptitle(title)
    times_stamps = np.arange(len(series[0]))

    if names is None:
        names = range(len(series))
    if len(series) == 3:
        names = ['X', 'Y', 'Z']
    if cores is None:
        cores = [(99/255, 110/255, 250/255, opacity), (239/255, 85/255, 59/255, opacity), (0/255, 204/255, 150/255, opacity), (171/255, 99/255, 250/255, opacity), (255/255, 161/255, 90/255, opacity), (25/255, 211/255, 243/255, opacity), (255/255, 102/255, 146/255, opacity), (182/255, 232/255, 128/255, opacity), (255/255, 151/255, 255/255, opacity), (254/255, 203/255, 82/255, opacity)]
    if cores_presentes is None:
        cores_presentes = plotly.colors.qualitative.Plotly
        
    if cores_bg is None:
        cores_bg = [(240/255, 140/255, 25/255, 0.6), (243/255, 155/255, 231/255, 0.6), (170/255, 31/255, 209/255, 0.6), (47/255, 233/255, 230/255, 0.6), (7/255, 166/255, 39/255, 0.6), (150/255, 54/255, 3/255, 0.6), (255/255, 0/255, 0/255, 0.6), (247/255, 251/255, 40/255, 0.6), (168/255, 168/255, 168/255, 0.6), (249/255, 210/255, 254/255, 0.6)]
    if atividades_nome is None:
        atividades_nome = {0: 'WALKING', 1: 'WALKING UPSTAIRS', 2: 'WALKING DOWNSTAIRS', 3: 'SITTING', 4: 'STANDING', 5: 'LAYING', 6: 'RUN', 7: 'BICYCLE', 8: 'TABLE', 9: 'PLATO'}

    for i in range(len(series)):
        ax.plot(times_stamps, series[i], label=names[i], color=cores[i])

    # build background regions
    regions = []
    anterior = int(atividades[0])
    i_inicio = 0
    for i in range(0, len(atividades)):
        if anterior != atividades[i]:
            regions.append({"start": i_inicio, "end": i, "color": cores_bg[anterior]})
            i_inicio = i
            anterior = int(atividades[i])
    regions.append({"start": i_inicio, "end": len(atividades), "color": cores_bg[anterior]})
    for region in regions:
        ax.axvspan(region["start"], region["end"], color=region["color"], alpha=0.5)

    # add subtitle
    for i in atividades_nome.keys():
        ax.scatter([], [], color=cores_bg[i], label=atividades_nome[i])

    # add present samples
    if presentes is not None:
        sequencias_continuas = []
        sequencia_atual = []
        for i in range(0, len(presentes)-1):
            sequencia_atual.append(presentes[i])
            if presentes[i] != presentes[i+1] - 1:
                sequencias_continuas.append(sequencia_atual)
                sequencia_atual = []
        sequencia_atual.append(presentes[-1])
        sequencias_continuas.append(sequencia_atual)

    if presentes is not None:
        # cria retangulos vermelhos com base em sequencias_continuas
        regions = []
        for sequencia in sequencias_continuas:
            # desenha uma reta preta vertical no começo e no fim da sequencia
            if not turn_off_black_lines:
                ax.axvline(sequencia[0], color='black')
            # sequencia
            for i in range(len(series)):
                ax.plot(times_stamps[sequencia], series[i][sequencia], color=cores_presentes[i])
            # traço vertical no final da sequencia
            if not turn_off_black_lines:
                ax.axvline(sequencia[-1], color='black')

        if plot_red_rects:
            # cria retangulos vermelhos com base em sequencias_continuas
            regions = []
            for sequencia in sequencias_continuas:
                regions.append({"start": sequencia[0], "end": sequencia[-1], "color": (1, 0, 0, 1)})

            for region in regions:
                ax.axvspan(region["start"], region["end"], ymin=0.99, ymax=1, color=(1, 0, 0, 0.6))
    
    if episodes is not None:
        for i in range(len(episodes)):
            # [{'label': 'breakfast', 'id': '0', 'start': 1732791357004000000, 'end': 1732793380133771776, 'color': '#ffdd0033'}, {'label': 'go_uni', 'id': '1', 'start': 1732793380133771776, 'end': 1732794416665211904, 'color': '#ff000033'}, {'label': 'meeting', 'id': '2', 'start': 1732794416665211904, 'end': 1732802894876557056, 'color': '#00ff0033'}, {'label': 'lunch', 'id': '3', 'start': 1732802894876557056, 'end': 1732813001618146048, 'color': '#eeff0033'}, {'label': 'class', 'id': '4', 'start': 1732813001618146048, 'end': 1732819465886000128, 'color': '#00ff1133'}, {'label': 'friends', 'id': '5', 'start': 1732819467814000128, 'end': 1732830620836465920, 'color': '#004cff33'}, {'label': 'dinner', 'id': '6', 'start': 1732830620836465920, 'end': 1732832657250881024, 'color': '#ff000033'}, {'label': 'friends', 'id': '5', 'start': 1732832657250881024, 'end': 1732849683041006080, 'color': '#0040ff33'}, {'label': 'go_home', 'id': '7', 'start': 1732849683041006080, 'end': 1732851014331000064, 'color': '#ff00ae33'}]
            # plota a label do episodio para aparecer na legenda
            ax.scatter([], [], color=episodes[i]["color"], label=episodes[i]["label"])
            # plota a faixa colorida do episodio
            # encontra os index start e end atraves dos timestamps mais proximos do start e end
            start = np.argmin(np.abs(time_stamps - episodes[i]["start"]//1e6))
            end = np.argmin(np.abs(time_stamps - episodes[i]["end"]//1e6))
            ax.axvspan(start, end, ymin=0.98, ymax=0.99, color=episodes[i]["color"], alpha=1)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=8)
    ax.set_ylim(minimo, maximo)
    ax.set_xlabel("Horas do Dia")
    ax.set_ylabel("Valores")
    ax.set_xticks(np.arange(0, len(time_stamps), step=len(time_stamps)//20))
    ax.set_xticklabels([f"{int((ts/3600000+utc)%24)}h:{int((ts/60000)%60)}m" for ts in time_stamps[::len(time_stamps)//20]])
    plt.show()
    plt.close()


def show_leaf(id, dataset, train, Z_latent, episodes = None):
    caderno = Anotator(impedir_repeticao=True)
    scanner = GenericScannerTree(Z_latent, leaf_function=caderno)
    scanner(id)

    presentes = []
    for idx in caderno.caderno:
        inicio = train.get_index((idx, 0))
        presentes += list(range(inicio, inicio+train.window_size))

    mostrar_series_presents_plt(dataset.T[7:10], dataset.T[-1], dataset.T[0], presentes, turn_off_black_lines=True, plot_red_rects=True, title=f"Visualização da Série Temporal referente ao grupo de ID {id}", episodes=episodes)
    


def show_episodes(data, title = "Episodios", modes = None, vertical = None, heights = None, show_y_ticks = None, milis_to_time = False, use_labels_as_title = None, path_to_save = None):
    """
    data should be a list of list of dict [[{}]]. Each dict must have the keys: start, end, label
    title can be a string or a list of string
    utc is the timezone of data
    """
    
    if isinstance(data[0], list):
        fig, axs = plt.subplots(len(data), 1, figsize=(20, sum(heights)), height_ratios=heights, sharex=True, gridspec_kw={'hspace': 0})
        if len(data) == 1:
            axs = [axs]
        for i, episode_list in enumerate(data):
            axs[i].set_ylabel(f"{title[i] if isinstance(title, list) else title}", rotation=-90, labelpad=10)
            axs[i].yaxis.set_label_position("right")

            if vertical:
                for v in vertical[i]:
                    axs[i].axvline(v, color='gray', linestyle='--')

            if modes and modes[i] == "line":
                x = []
                y = []
                if type(episode_list) == np.ndarray:
                    axs[i].plot(episode_list, color="black")
                else:
                    for episode in episode_list:
                        start = episode['start']
                        end = episode['end']
                        x.append(start)
                        y.append(episode['label'])
                        x.append(end)
                        y.append(episode['label'])
                    
                    
                    for j in range(0, len(x), 2):
                        axs[i].plot(x[j:j+2], y[j:j+2], color="black", linewidth=3)
                axs[i].grid(axis='y')
            elif modes and modes[i] == "error":
                center = []
                error = []
                for e, episode in enumerate(episode_list):
                    start = episode['start']
                    end = episode['end']
                    center.append((start + end) / 2)
                    error.append((end - start) / 2)
                    identifier = [chr(i) for i in range(65, 91)]
                    if use_labels_as_title and use_labels_as_title[i]:
                        axs[i].text((start+end)/2, 0, episode['label'], ha='center', va='center', fontsize=8, color='black', bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle,pad=0.3'))
                    else:
                        axs[i].text((start+end)/2, 0, identifier[e], ha='center', va='center', fontsize=8, color='black', bbox=dict(facecolor='white', edgecolor='none', boxstyle='circle,pad=0.3'))
                
                axs[i].errorbar(center, np.zeros(len(center)), xerr=error, fmt='.', color='black', capthick=2, capsize=10)
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['left'].set_visible(False)
                axs[i].spines['bottom'].set_visible(False)
                axs[i].set_ylim(-2, 1)
                
                if use_labels_as_title and use_labels_as_title[i]:
                    pass
                else:
                    # Add legend
                    legend_labels = {chr(65 + e): episode['label'] for e, episode in enumerate(episode_list)}
                    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label=f"{key}: {value}") for key, value in legend_labels.items()]
                    axs[i].legend(handles=handles, loc='lower center', ncol=len(legend_labels))
            elif modes and modes[i] == "time_series":
                axs[i].plot(episode_list[0], episode_list[1], color="black")

            if milis_to_time:
                if modes and modes[i] == "time_series":
                    maximus = max(episode_list[0])
                    minimun = min(episode_list[0])
                    axs[i].set_xticks(np.arange(minimun, maximus, step=(maximus-minimun)//19))
                    utc = -3
                    axs[i].set_xticklabels([f"{int((ts/3600000+utc)%24)}h:{int((ts/60000)%60)}m" for ts in np.arange(minimun, maximus, step=(maximus-minimun)//19)])
                else:
                    maximus = max([episode['end'] for episode in episode_list])
                    minimun = min([episode['start'] for episode in episode_list])
                    axs[i].set_xticks(np.arange(minimun, maximus, step=(maximus-minimun)//19))
                    utc = -3
                    axs[i].set_xticklabels([f"{int((ts/3600000+utc)%24)}h:{int((ts/60000)%60)}m" for ts in np.arange(minimun, maximus, step=(maximus-minimun)//19)])
            if show_y_ticks and show_y_ticks[i] == False:
                axs[i].set_yticks([])

    else:
        raise ValueError("Data must be a list of list of dict or a list of dict")
    
    if path_to_save:
        plt.savefig(path_to_save, bbox_inches='tight')
    else:
        plt.show()

def extract_vertical(episodes):
    vertical = []
    for episode in episodes:
        start = episode['start']
        end = episode['end']
        vertical.append(start)
        vertical.append(end)
    return vertical

# example: show_episodes([episodes,episodes, episodes], title=["Episodes\nlabels", "Infered\nEpisodes", "Activities"], modes = ["error", "error", "line"], vertical = [extract_vertical(episodes),extract_vertical(episodes), extract_vertical(episodes)], heights = [1,1,5], show_y_ticks = [False, False, True] )

# get the activities
def har_episode_format(vector):
    har_sequences = []
    atividades_nome = {0: 'WALKING', 1: 'WALKING UPSTAIRS', 2: 'WALKING DOWNSTAIRS', 3: 'SITTING', 4: 'STANDING', 5: 'LAYING', 6: 'RUN', 7: 'BICYCLE', 8: 'TABLE', 9: 'PLATO'}
    atividades = vector.T[-1]
    anterior = int(atividades[0])
    i_inicio = 0
    for i in range(0, len(atividades)):
        if anterior != atividades[i]:
            har_sequences.append({"start": int(vector[i_inicio][0]), "end": int(vector[i][0]), "label": atividades_nome[anterior]})
            i_inicio = i
            anterior = int(atividades[i])
    har_sequences.append({"start": int(vector[i_inicio][0]), "end": int(vector[len(atividades)-1][0]), "label": atividades_nome[anterior]})
    return har_sequences

def episode_to_ms(episodes, vector, dataset):
    for episode in episodes:
        episode["start"] = vector[dataset.get_index((episode["start"],0))][0]
        episode["end"] = vector[dataset.get_index((episode["end"],0))][0]
    return episodes

def get_start_end_label(vector):
    sequences = []
    last = int(vector[0])
    i_start = 0
    id = 1
    for i in range(0, len(vector)):
        if last != vector[i]:
            sequences.append({"start": i_start, "end": i, "label": id})
            i_start = i
            last = int(vector[i])
            id += 1
    sequences.append({"start": i_start, "end": len(vector)-1, "label": id})
    return sequences


def generate_map(predicted_ajdacent_ward, vector, dataset_path, lat_long_pos):
    groups = []

    for episode in predicted_ajdacent_ward:
        start = episode["start"]
        end = episode["end"]
        points_group = vector[(vector[:, 0] >= start) & (vector[:, 0] <= end)]
        
        points_group = points_group[:, [0, lat_long_pos[0], lat_long_pos[1]]]
        groups.append(points_group)

    # Define a lista de cores para os grupos
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'black', 'yellow', 'darkblue', 'darkgreen', 'cadetblue', 'grey', 'pink', 'lightblue', 'lightgreen']

    # Cria o mapa centrado no primeiro ponto do primeiro grupo
    mapa = folium.Map(location=[groups[0][0, 1], groups[0][0, 2]], zoom_start=16)

    # Adiciona um grupo de camadas para controle
    layer_control = folium.map.LayerControl(collapsed=False)

    # Plota cada grupo com uma cor diferente
    for group_index, points_group in enumerate(groups):
        color = colors[group_index % len(colors)]
        start_time = datetime.fromtimestamp(int(points_group[0, 0] / 1000))
        end_time = datetime.fromtimestamp(int(points_group[-1, 0] / 1000))
        group = folium.FeatureGroup(name=f'Grupo {group_index + 1} ({start_time.strftime("%H:%M")}-{end_time.strftime("%H:%M")})', show=False)
        for i in range(0, len(points_group), 100):
            if group._children:  # Check if there are already points in the group
                last_point = list(group._children.values())[-1].location
                distance = np.sqrt((points_group[i, 2] - last_point[0])**2 + (points_group[i, 1] - last_point[1])**2)
                if distance < 0.00001:  # Skip if the point is too close to the last one
                    continue
            timestamp = points_group[i, 0]
            time_str = (datetime.fromtimestamp(int(timestamp / 1000))).strftime('%H:%M:%S')
            folium.CircleMarker(
                location=[points_group[i, 1], points_group[i, 2]],  # Coordinates
                radius=5,  # Marker size
                color=color,  # Marker color
                fill=True,  # Fill the marker
                fill_color=color,  # Fill color
                fill_opacity=1,  # Marker opacity
                weight=0.5,  # Border weight
                tooltip=time_str  # Tooltip with the time
            ).add_to(group)
        group.add_to(mapa)

    # Adiciona o controle de camadas ao mapa
    layer_control.add_to(mapa)

    # Salva o mapa em um arquivo HTML
    mapa.save(dataset_path+'groups_map.html')

def show_sensors(vector, predicted_ajdacent_ward, path, pos_in_vector = [[1,2,3], [4,5,6], [10,11,12]], names = ["Accelerometers", "Gyroscopes", "GPS"]):

    # generalize a linha abaixo para qualquer valor na entrada acima, com o n sendo variavel para len do primeiro elemento de pos_in_vector e seus indices tbm
    for i, sensor_indices in enumerate(pos_in_vector):
        sensor_avg = np.mean(vector[:, sensor_indices], axis=1)
        show_episodes(
            [predicted_ajdacent_ward, [vector[:, 0], sensor_avg]],
            title=[f"Infered\nEpisodes\nadjacent", f"Average \n of \n {names[i]}"],
            modes=["error", "time_series"],
            vertical=[extract_vertical(predicted_ajdacent_ward), extract_vertical(predicted_ajdacent_ward)],
            heights=[1, 3],
            show_y_ticks=[False, False],
            milis_to_time=True,
            use_labels_as_title=[True, True],
            path_to_save=f"{path}/{names[i]}.png"
        )

def show_sensors_plotly(vector, predicted_ajdacent_ward, path, pos_in_vector = [[1,2,3], [4,5,6], [10,11,12]], names = ["Accelerometers", "Gyroscopes", "GPS"], together = False, normalization = None, common_ts_labels = None):
    vector = vector.copy()
    if normalization:
        if normalization == "minmax":
            vector[:, 1:] = (vector[:, 1:] - np.min(vector[:, 1:], axis=0)) / (np.max(vector[:, 1:], axis=0) - np.min(vector[:, 1:], axis=0))
        elif normalization == "standard":
            vector[:, 1:] = (vector[:, 1:] - np.mean(vector[:, 1:], axis=0)) / np.std(vector[:, 1:], axis=0)
        else:
            raise ValueError("Normalization must be 'minmax' or 'standard'")

    if together:

        extra = 0 # Add an extra row for the common_ts_labels subplot
        if common_ts_labels is not None:
            extra += 1

        # Create subplots with shared x-axis
        n_sensors = len(pos_in_vector)
        fig = make_subplots(
            rows=n_sensors + 1 + extra, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.02,
            row_heights=[1] + [3] * (n_sensors + extra),
            subplot_titles=["Episodes"] + [f"Average of {name}" for name in names] + (["Groups of a simple hierarchical cluster"] if common_ts_labels is not None else [])
        )

        # Add episodes subplot (first row)
        episode_list = predicted_ajdacent_ward
        center = []
        error = []
        texts = []
        for e, episode in enumerate(episode_list):
            start = episode['start']
            end = episode['end']
            center.append((start + end) / 2)
            error.append((end - start) / 2)
            texts.append(episode['label'])

        fig.add_trace(go.Scatter(
            x=center, y=[0]*len(center), mode='markers+text',
            marker=dict(color='black', size=8),
            text=texts, textposition='top center',
            showlegend=False
        ), row=1, col=1)

        # Add error bars for episodes
        for i, (c, e) in enumerate(zip(center, error)):
            fig.add_shape(
                type="line",
                x0=c-e, y0=0, x1=c+e, y1=0,
                line=dict(color="black", width=3),
                row=1, col=1
            )

        fig.update_yaxes(range=[-1, 1], showticklabels=False, row=1, col=1)

        # Add sensor data subplots
        for i, (sensor_indices, name) in enumerate(zip(pos_in_vector, names)):
            row = i + 2
            sensor_avg = np.mean(vector[:, sensor_indices], axis=1)
            # Use numpy vectorized operations for faster conversion
            ts_seconds = vector[:, 0] / 1000

            # Convert timestamps to hours, minutes, seconds
            ts_seconds = ts_seconds[:len(sensor_avg)]
            hours = ((ts_seconds // 3600 - 3) % 24).astype(int)
            minutes = ((ts_seconds // 60) % 60).astype(int)
            seconds = (ts_seconds % 60).astype(int)
            time_labels = np.array([f"{h:02d}:{m:02d}:{s:02d}" for h, m, s in zip(hours, minutes, seconds)])
            
            fig.add_trace(go.Scatter(
                x=vector[:, 0], 
                y=sensor_avg, 
                mode='lines', 
                line=dict(color='black'),
                name=name,
                showlegend=False,
                hovertemplate='Time: %{customdata}<br>Value: %{y}<extra></extra>',
                customdata=time_labels
            ), row=row, col=1)

        if common_ts_labels is not None:
            # Add common time series as points on a separate subplot

            fig.add_trace(go.Scatter(
                x=vector[:, 0], 
                y=common_ts_labels, 
                mode='lines', 
                marker=dict(color='black'),
                name='Common Time Series',
                showlegend=False,
                hovertemplate='Time: %{customdata}<br>Value: %{y}<extra></extra>',
                customdata=time_labels
            ), row=n_sensors + 2, col=1)

        # Add vertical lines for episode boundaries on all subplots
        vertical_lines = extract_vertical(predicted_ajdacent_ward)
        for i in range(0,len(vertical_lines),2):
            for row in range(1, n_sensors + 2 +extra):  # Add to all rows
                fig.add_vline(x=vertical_lines[i], line_dash="dash", line_color="gray", row=row, col=1)
        fig.add_vline(x=vertical_lines[-1], line_dash="dash", line_color="gray", row=row, col=1)

        # Configure time axis for all subplots
        minimun = min(vector[:, 0])
        maximus = max(vector[:, 0])
        ticks = np.arange(minimun, maximus, step=(maximus-minimun)//19)
        labels = [f"{int((ts/3600000-3)%24)}h:{int((ts/60000)%60)}m" for ts in ticks]
        fig.update_xaxes(tickvals=ticks, ticktext=labels)

        fig.update_layout(
            height=300 * (n_sensors + 1), 
            showlegend=False,
            title_text="Episodes and Sensor Data"
        )
        if path:
            save_path = f"{path}combined_sensors.html"
            fig.write_html(save_path)
        else:
            fig.show()    
        
    else:
        for i, sensor_indices in enumerate(pos_in_vector):
            sensor_avg = np.mean(vector[:, sensor_indices], axis=1)
            data = [predicted_ajdacent_ward, [vector[:, 0], sensor_avg]]
            title = [f"Infered\nEpisodes\nadjacent", f"Average \n of \n {names[i]}"]
            modes = ["error", "time_series"]
            vertical = [extract_vertical(predicted_ajdacent_ward), extract_vertical(predicted_ajdacent_ward)]
            heights = [1, 3]
            show_y_ticks = [False, False]
            milis_to_time = True
            use_labels_as_title = [True, True]
            path_to_save = f"{path}/{names[i]}.html"

            if not isinstance(data[0], list):
                raise ValueError("Data must be a list of list of dict or a list of [x, y] for time_series")

            n = len(data)
            if heights is None:
                heights = [1] * n
            if modes is None:
                modes = ["error"] * n
            if show_y_ticks is None:
                show_y_ticks = [True] * n
            if use_labels_as_title is None:
                use_labels_as_title = [False] * n
            if isinstance(title, str):
                title = [title] * n

            fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights, subplot_titles=title)

            for i, episode_list in enumerate(data):
                row = i + 1
                mode = modes[i]
                if mode == "line":
                    x = []
                    y = []
                    if isinstance(episode_list, np.ndarray):
                        fig.add_trace(go.Scatter(x=np.arange(len(episode_list)), y=episode_list, mode='lines', line=dict(color='black')), row=row, col=1)
                    else:
                        for episode in episode_list:
                            start = episode['start']
                            end = episode['end']
                            x += [start, end]
                            y += [episode['label'], episode['label']]
                        for j in range(0, len(x), 2):
                            fig.add_trace(go.Scatter(x=x[j:j+2], y=y[j:j+2], mode='lines', line=dict(color='black', width=3)), row=row, col=1)
                elif mode == "error":
                    center = []
                    error = []
                    texts = []
                    for e, episode in enumerate(episode_list):
                        start = episode['start']
                        end = episode['end']
                        center.append((start + end) / 2)
                        error.append((end - start) / 2)
                        label = episode['label']
                        if use_labels_as_title[i]:
                            texts.append(str(label))
                        else:
                            texts.append(chr(65 + e))
                    fig.add_trace(go.Scatter(
                        x=center, y=[0]*len(center), mode='markers+text',
                        marker=dict(color='black', size=8),
                        text=texts, textposition='top center',
                        showlegend=False
                    ), row=row, col=1)
                    fig.add_trace(go.Scatter(
                        x=[c for c in center for _ in (0, 1)],
                        y=[0]*2*len(center),
                        error_x=dict(
                            type='data',
                            symmetric=False,
                            array=[e for e in error],
                            arrayminus=[e for e in error],
                            thickness=2,
                            width=10
                        ),
                        mode='markers',
                        marker=dict(color='rgba(0,0,0,0)'),
                        showlegend=False
                    ), row=row, col=1)
                    fig.update_yaxes(range=[-2, 1], row=row, col=1)
                    if not use_labels_as_title[i]:
                        legend_labels = {chr(65 + e): episode['label'] for e, episode in enumerate(episode_list)}
                        for key, value in legend_labels.items():
                            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='black'), name=f"{key}: {value}"), row=row, col=1)
                elif mode == "time_series":
                    x, y = episode_list
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='black')), row=row, col=1)

                if vertical and vertical[i]:
                    for v in vertical[i]:
                        fig.add_vline(x=v, line_dash="dash", line_color="gray", row=row, col=1)

                if not show_y_ticks[i]:
                    fig.update_yaxes(showticklabels=False, row=row, col=1)

                if milis_to_time:
                    if mode == "time_series":
                        minimun = min(x)
                        maximus = max(x)
                        ticks = np.arange(minimun, maximus, step=(maximus-minimun)//19)
                        labels = [f"{int((ts/3600000-3)%24)}h:{int((ts/60000)%60)}m" for ts in ticks]
                        fig.update_xaxes(tickvals=ticks, ticktext=labels, row=row, col=1)
                    else:
                        minimun = min([episode['start'] for episode in episode_list])
                        maximus = max([episode['end'] for episode in episode_list])
                        ticks = np.arange(minimun, maximus, step=(maximus-minimun)//19)
                        labels = [f"{int((ts/3600000-3)%24)}h:{int((ts/60000)%60)}m" for ts in ticks]
                        fig.update_xaxes(tickvals=ticks, ticktext=labels, row=row, col=1)

            fig.update_layout(height=300*sum(heights), showlegend=True, title_text=None)
            if path_to_save:
                fig.write_html(path_to_save)
            else:
                fig.show()
    
def group_of_ts(labels, dataset):
    """
    Create a new array of groups based on the labels and the dataset's step size.

    Parameters:
    labels (np.ndarray): An array of labels indicating the group for each time step in the windowed latent space.
    dataset (Dataset): The dataset object containing the step size and the original dataset with time series data.

    Returns:
    np.ndarray: An array where each label is repeated according to the dataset's step size,
                ensuring that the length of the array matches the number of columns in the dataset.

    """
    new_groups = np.repeat(labels, dataset.step_size)
    # Ensure the new_groups array has the same length as the dataset's number of columns coping the last value if necessary
    if new_groups.shape[0] < dataset.dataset.shape[1]:
        last_value = new_groups[-1]
        n_missing = dataset.dataset.shape[1] - new_groups.shape[0]
        new_groups = np.concatenate([new_groups, np.full(n_missing, last_value, dtype=new_groups.dtype)])

    return new_groups

def compute_episodes(Z_adjacent, number_episodes, dataset_path, dataset, vector, lat_long_pos = (11,10), pos_in_vector = [[1,2,3], [4,5,6], [10,11,12]], names = ["Accelerometers", "Gyroscopes", "GPS"], normalization = None, common_HC = None, need_fcluster = True):
    # Define the path to save the episodes
    save_path = os.path.join(dataset_path, f"episodes/{number_episodes}/")
    os.makedirs(save_path, exist_ok=True)

    if need_fcluster:
        labels = fcluster_custom(Z_adjacent, number_episodes)   # standard path, cut Z_matrix
    else:
        labels = Z_adjacent # already cut Z_matrix
    predicted_ajdacent_ward = episode_to_ms(get_start_end_label(labels), vector, dataset)

    common_ts_labels = None
    if common_HC is not None:
        common_ts_labels = group_of_ts(fcluster(common_HC, number_episodes, criterion='maxclust'), dataset)

    # show_sensors(vector, predicted_ajdacent_ward, save_path, pos_in_vector= pos_in_vector, names=names)
    show_sensors_plotly(vector, predicted_ajdacent_ward, save_path, pos_in_vector= pos_in_vector, names=names, together=True, normalization=normalization, common_ts_labels=common_ts_labels)

    generate_map(predicted_ajdacent_ward, vector, save_path, lat_long_pos = lat_long_pos)

    return labels


def compute_map_of_all_episodes(Z_adjacent, numbers_episodes, dataset_path, dataset, vector, lat_long_pos = (11, 10)):
    """
    Performs a procedure similar to generate_map, but instead of each group being one of the episodes, 
    each group contains all the episodes of one of the cuts in numbers_episodes, with each episode separated by colors.
    """
    groups_by_cut = {}

    # Group episodes by each cut in numbers_episodes
    for number_episodes in numbers_episodes:
        predicted_ajdacent_ward = episode_to_ms(get_start_end_label(fcluster_custom(Z_adjacent, number_episodes)), vector, dataset)
        groups_by_cut[number_episodes] = predicted_ajdacent_ward

    # Define a list of colors for the episodes
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'yellow', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen']

    # Create a map centered on the first point of the dataset
    mapa = folium.Map(location=[vector[0, lat_long_pos[0]], vector[0, lat_long_pos[1]]], zoom_start=16)

    # Add a layer control for better visualization
    layer_control = folium.map.LayerControl(collapsed=False)

    # Iterate over each cut and its episodes
    for cut_index, (cut, episodes) in enumerate(groups_by_cut.items()):
        cut_group = folium.FeatureGroup(name=f'<a href="./{cut}/groups_map.html" target="_blank">Cut {cut} episodes</a>', show=False)
        for episode_index, episode in enumerate(episodes):
            start = episode["start"]
            end = episode["end"]
            points_group = vector[(vector[:, 0] >= start) & (vector[:, 0] <= end)]

            points_group = points_group[:, [0, lat_long_pos[0], lat_long_pos[1]]]
            color = colors[episode_index % len(colors)]
            for i in range(0, len(points_group), 100):
                if cut_group._children:  # Check if there are already points in the group
                    last_point = list(cut_group._children.values())[-1].location
                    distance = np.sqrt((points_group[i, 2] - last_point[0])**2 + (points_group[i, 1] - last_point[1])**2)
                    if distance < 0.00001:  # Skip if the point is too close to the last one
                        continue
                timestamp = points_group[i, 0]
                time_str = (datetime.fromtimestamp(int(timestamp / 1000))).strftime('%H:%M:%S')
                folium.CircleMarker(
                    location=[points_group[i, 1], points_group[i, 2]],  # Coordinates
                    radius=5,  # Marker size
                    color=color,  # Marker color
                    fill=True,  # Fill the marker
                    fill_color=color,  # Fill color
                    fill_opacity=1,  # Marker opacity
                    weight=0.5,  # Border weight
                    tooltip=f"Cut {cut} - Episode {episode_index + 1} - {time_str}"  # Tooltip with the time
                ).add_to(cut_group)
        cut_group.add_to(mapa)

    # Add the layer control to the map
    layer_control.add_to(mapa)

    # Save the map to an HTML file
    mapa.save(dataset_path + 'episodes/cuts_map.html')

def caching(func, args, cache_path):
    if os.path.exists(cache_path):
        result = np.load(cache_path, allow_pickle=True)
    else:
        result = func(*args)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, result)
    return result

def compute_linkage_adjacent_table(latent, dataset_path):

    # Define the path to save the Z_adjacent file
    save_path = os.path.join(dataset_path, "episodes/Z_adjacent.npy")

    # Check if the file already exists
    if os.path.exists(save_path):
        # Load and return the precomputed Z_adjacent
        Z_adjacent = np.load(save_path)
    else:
        # Compute Z_adjacent
        Z_adjacent = linkage_adjacent_ward(latent)  # takes 4 min

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the computed Z_adjacent
        np.save(save_path, Z_adjacent)

    return Z_adjacent

def show_dist_number_plot(d, path_to_save="", version = None, Z = None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d[::-1,0], y=d[::-1, 1], mode='markers'))
    fig.update_layout(
        title="Distance vs Number Plot",
        xaxis_title="Number of clusters",
        yaxis_title="Total distance ward inter clusters",
        template="plotly_white"
    )

    if version is not None:
        assert Z is not None, "If version is provided, Z must also be provided"
        inicial_y = Z[-1, 2]
        if version == "linear":
            step = - (inicial_y / Z.shape[0])
            x = np.linspace(0, Z.shape[0] - 1, Z.shape[0])
            y = inicial_y + step * x
            x = x + 1
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Linear Fit'
            ))
        elif version == "exponential":
            step = 1 / (inicial_y ** (1 / Z.shape[0]))
            x = np.linspace(0, Z.shape[0] - 1, Z.shape[0])
            y = inicial_y * (step ** x)
            x = x + 1
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Exponential Fit'
            ))
    fig.write_html(f"{path_to_save}dist_number_{version}.html")






def estimate_cost_matrix(gt_labels, cluster_labels):
    # Make sure the lengths of the inputs match:
    if len(gt_labels) != len(cluster_labels):
        print('The dimensions of the gt_labls and the pred_labels do not match')
        return -1
    L_gt = np.unique(gt_labels)
    L_pred = np.unique(cluster_labels)
    nClass_pred = len(L_pred)
    dim_1 = max(nClass_pred, np.max(L_gt) + 1)
    profit_mat = np.zeros((nClass_pred, int(dim_1)))
    for i in L_pred:
        idx = np.where(cluster_labels == i)
        gt_selected = gt_labels[idx]
        for j in L_gt:
            try:
                profit_mat[int(i)][int(j)] = np.count_nonzero(gt_selected == j)
            except Exception as e:
                print(f"{profit_mat=}, {i=}, {j=}, {gt_selected=}")
                print("Maybe turn on metrics( correct_labels = True ) to fix this error.")
                raise e
    return -profit_mat

def translate_labels(gt_labels, cluster_labels):
        """
        get cluster_labels and translate for gt_labels in order to get the best accuracy as possible,
        changing the numbers of cluster_labels to the best of gt_labels, that is used just once
        """
        cost_matrix = estimate_cost_matrix(gt_labels, cluster_labels)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        y_pred = col_ind[cluster_labels]
        return y_pred

import warnings

def metricas(gt_labels, cluster_labels, path_to_save_fig = None, correct_labels = False, save_data_as_npy = False):
    """
    Correct labels will change the cluster_labels to be sequential from 0 to n_clusters-1, intead of jumping numbers.
    This is useful when the clustering algorithm does not assign labels in a sequential manner. Standard is False.
    """
    if correct_labels:
        cluster_labels = groups_to_episodes(cluster_labels, inplace = False)
        if len(np.unique(cluster_labels)) < np.max(cluster_labels) + 1 or np.min(cluster_labels) < 0:
            new_labels = np.zeros_like(cluster_labels)
            new_label = -1
            current_label = None
            for i in range(len(cluster_labels)):
                if cluster_labels[i] != current_label:
                    current_label = cluster_labels[i]
                    new_label += 1
                new_labels[i] = new_label
            cluster_labels = new_labels
    y_pred = translate_labels(gt_labels, cluster_labels)
    # Compute and return evaluation metrics (e.g., accuracy, precision, recall)
    if path_to_save_fig is not None:
        os.makedirs(path_to_save_fig, exist_ok=True)
        plot_sequence(y_pred, gt_labels, path_to_save_fig)
        if save_data_as_npy:
            np.save(path_to_save_fig + "y_pred.npy", y_pred)
            np.save(path_to_save_fig + "gt_labels.npy", gt_labels)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        n_clusters = len(np.unique(cluster_labels))
        cur_acc = metrics.accuracy_score(gt_labels, y_pred)
        f1_macro = metrics.f1_score(gt_labels, y_pred, average='macro') # F1-Score
        iou = np.sum(metrics.jaccard_score(gt_labels, y_pred, average=None)) / n_clusters
    if path_to_save_fig is not None:
        # save values in a txt
        with open(path_to_save_fig + "metrics.txt", "w") as f:
            f.write("Accuracy: {}\n".format(cur_acc))
            f.write("F1 Macro: {}\n".format(f1_macro))
            f.write("IoU: {}\n".format(iou))
    return {
        "accuracy": cur_acc,
        "f1_macro": f1_macro,
        "iou": iou
    }

from repro.streamer.streamer.experiments.hlr import AnnotationsAsGraph
import copy

def streamer_metrics(duration, h_boundaries, ga,  gt_idx=0):
    ea = dict(file="filename",
                             fileType = "video/mp4",
                             cursor = 0,
                             duration=duration,
                             zoom=1,
                             layers=[])

    layer_template_general =dict(name = "layer 0",
                order = 0,
                annots = []
                )

    annot_template_general = dict(start=0.0,
                                end=0.5,
                                action="N/A",
                                colour =f'rgb{0,0,0}',
                                representation="null")

    for layer_num in range(len(h_boundaries)):

        layer_template = copy.deepcopy(layer_template_general)
        layer_template["name"] = f'layer {layer_num}'
        layer_template["order"] = layer_num

        boundaries = h_boundaries[layer_num]
        for i in range(len(boundaries)-1):
            annot_template = copy.deepcopy(annot_template_general)
            annot_template["start"] = boundaries[i]
            annot_template["end"] = boundaries[i+1]
            annot_template["action"] = "N/A"
            layer_template["annots"].append(annot_template)

        ea["layers"].append(layer_template)


    gta = ga['layers'][gt_idx]
    ea['layers'].append(gta)

    _, (iou, mof) = AnnotationsAsGraph.run_file(ea, False)


    return iou, mof



    
    


def add_legacy_metrics(current_metrics):
    """
    Adds legacy metrics to the current metrics dictionary for backward compatibility.
    
    Args:
        current_metrics (dict): Dictionary containing current metrics with keys 'accuracy', 'f1_macro', and 'iou'.
    """
    current_metrics['streamer_mof'] = current_metrics['accuracy']
    current_metrics['streamer_iou'] = current_metrics['f1_macro']

def iou_streamer(gt_labels, cluster_labels):
    pass

def mof_streamer(gt_labels, cluster_labels):
    pass


def plot_sequence(y_pred, gt_labels, path_to_save_fig=None):
    """
    Plota duas sequências categóricas (predição e ground truth) como blocos coloridos.
    
    Args:
        y_pred (list[int]): Sequência predita.
        gt_labels (list[int]): Sequência ground truth.
        path_to_save_fig (str): Caminho para salvar a figura (opcional).
    """
    # Paleta de cores automática (usa 'Plotly' que tem muitas cores distintas)
    base_palette = pc.qualitative.Plotly
    n_colors = len(base_palette)

    # Mapeia cada classe a uma cor da paleta (se passar do tamanho, repete)
    unique_labels = sorted(set(y_pred) | set(gt_labels))
    color_map = {label: base_palette[i % n_colors] for i, label in enumerate(unique_labels)}
    
    def compress_sequence(seq):
        """Transforma [0,0,1,1,2] em [(valor, start, end)]"""
        segments = []
        start = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]:
                segments.append((seq[i-1], start, i))
                start = i
        segments.append((seq[-1], start, len(seq)))
        return segments
    
    # Compacta sequências em intervalos
    pred_segments = compress_sequence(y_pred)
    gt_segments = compress_sequence(gt_labels)
    
    fig = go.Figure()
    
    # Sequência predita (linha superior em y=1)
    for val, start, end in pred_segments:
        fig.add_trace(go.Scatter(
            x=[start, end, end, start],
            y=[1.5, 1.5, 2, 2],  # faixa para "Ours"
            fill="toself",
            mode="lines",
            line=dict(color=color_map[val]),
            fillcolor=color_map[val],
            name=f"Pred {val}",
            showlegend=False
        ))
    
    # Sequência ground truth (linha inferior em y=0)
    for val, start, end in gt_segments:
        fig.add_trace(go.Scatter(
            x=[start, end, end, start],
            y=[0, 0, 0.5, 0.5],  # faixa para "GT"
            fill="toself",
            mode="lines",
            line=dict(color=color_map[val]),
            fillcolor=color_map[val],
            name=f"GT {val}",
            showlegend=False
        ))
    
    # Layout
    fig.update_layout(
        title="Episodes Visualization",
        xaxis=dict(title="Time (frames)", showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=True, tickvals=[0.25, 1.75], ticktext=["GT", "Prediction"],
                   showgrid=False, zeroline=False, range=[-0.5, 2.5]),
        plot_bgcolor="white",
        height=300
    )
    
    if path_to_save_fig:
        fig.write_html(path_to_save_fig + "episodes.html")

def plot_sequences(y_preds, gt_labels, path_to_save_fig=None, pred_labels=None, best_assignment = False):
    """
    Plot multiple predicted sequences and one ground truth sequence as colored blocks.
    
    Args:
        y_preds (list[list[int]]): List of predicted sequences.
        gt_labels (list[int]): Ground truth sequence.
        path_to_save_fig (str): Path to save the figure (optional).
        pred_labels (list[str]): Labels for each prediction sequence (optional).
    """
    
    # Color palette
    base_palette = pc.qualitative.Plotly
    n_colors = len(base_palette)

    for i in range(len(y_preds)):
        if best_assignment:
            y_preds[i] = translate_labels(gt_labels, y_preds[i])


    # Get unique labels from all sequences
    unique_labels = set(gt_labels)
    for pred in y_preds:
        unique_labels.update(pred)
    unique_labels = sorted(unique_labels)
    
    # Create color map
    color_map = {label: base_palette[i % n_colors] for i, label in enumerate(unique_labels)}
    
    def compress_sequence(seq):
        """Convert [0,0,1,1,2] to [(value, start, end)]"""
        segments = []
        start = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]:
                segments.append((seq[i-1], start, i))
                start = i
        segments.append((seq[-1], start, len(seq)))
        return segments
    
    fig = go.Figure()
    
    # Plot each prediction sequence
    n_preds = len(y_preds)
    for pred_idx, pred in enumerate(y_preds):
        pred_segments = compress_sequence(pred)
        y_base = (n_preds - pred_idx) * 1.0
        
        for val, start, end in pred_segments:
            fig.add_trace(go.Scatter(
                x=[start, end, end, start],
                y=[y_base + 0.5, y_base + 0.5, y_base + 1, y_base + 1],
                fill="toself",
                mode="lines",
                line=dict(color=color_map[val]),
                fillcolor=color_map[val],
                name=f"Pred {val}",
                showlegend=False
            ))
    
    # Plot ground truth at the bottom
    gt_segments = compress_sequence(gt_labels)
    for val, start, end in gt_segments:
        fig.add_trace(go.Scatter(
            x=[start, end, end, start],
            y=[0, 0, 0.5, 0.5],
            fill="toself",
            mode="lines",
            line=dict(color=color_map[val]),
            fillcolor=color_map[val],
            name=f"GT {val}",
            showlegend=False
        ))
    
    # Create y-axis labels
    tickvals = [0.25]  # GT position
    ticktext = ["GT"]
    for i in range(n_preds):
        tickvals.append((n_preds - i) + 0.75)
        ticktext.append(f"Pred {i+1}" if pred_labels is None else pred_labels[i])
    
    fig.update_layout(
        title="Episodes Visualization",
        xaxis=dict(title="Time (frames)", showgrid=False, zeroline=False),
        yaxis=dict(
            showticklabels=True,
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=False,
            zeroline=False,
            range=[-0.5, n_preds + 1.5]
        ),
        plot_bgcolor="white",
        height=100 + 100*n_preds
    )
    
    if path_to_save_fig:
        os.makedirs(path_to_save_fig, exist_ok=True)
        fig.write_html(path_to_save_fig + "episodes_hierar.html")
    else:
        fig.show()

def groups_to_episodes(vector, inplace = False):
    """
    This function translate a unidimensional vector of non continuos groups to different continuos episodes
    if inplace is True, is not created another vector.

    Param: a unidimensional vector of ints representing id of group for each time, not necessary continuos. Example:
    [0,0,0,0,1,1,1,0,0,2,2,2] (note that the group 0 is not continuos and every index has a group assigned)

    return: a unidimensional vector of new ints from 0 to n-1 with n continuos episodes. Example:
    [0,0,0,0,1,1,1,2,2,3,3,3] (every index has a episode assigned, but now they are continuos and the group 0 is not repeated. Also, the numbers are sequencial from 0 to n-1)
    """

    if inplace:
        ans = vector
    else:
        ans = np.zeros(len(vector))
    last = None
    id = -1
    for i in range(len(vector)):
        if last != vector[i]:
            id+=1
            last = vector[i]
        ans[i] = id
    ans = ans.astype(int)
    return ans

def get_starts_from_episodes(episodes):
    """
    This function takes a contiguous list of episodes and returns their start indices.

    Parameters:
    A unidimensional vector of ints from 0 to n-1 with n continuos episodes.

    Returns:
    list[int]: A list of start indices for each episode.
    """
    old_episode = -1
    starts = []
    for i in range(len(episodes)):
        if episodes[i] != old_episode:
            starts.append(i)
            old_episode = episodes[i]
    return starts

def get_connections(fine_episodes, coarse_episodes, new_index=0):
    """
    This function takes two contiguous lists of episodes (coarse and fine) of same length and returns a list of connections between them.

    Here we assume that every coarse border is also a fine border, but not the opposite, so every fine episode is contained in only a single
     coarse episode. Also, we assume that the episodes are ordered and contiguous, so if two consecutive indices have different episode id, it means that there is a border between them.

    Parameters:
    fine_episodes (list[int]): A unidimensional vector of ints from 0 to n-1 with n continuos episodes (fine).
    coarse_episodes (list[int]): A unidimensional vector of ints from 0 to n-1 with n continuos episodes (coarse).

    Returns:
    list[list[int]]: A list of lists where each internal list contains the indices of the fine episodes that are connected in the same coarse episode.
    """
    assert fine_episodes.shape == coarse_episodes.shape, f"The fine and coarse episodes must have the same shape: {fine_episodes.shape} vs {coarse_episodes.shape}"
    connections = []
    last_coarse = -1
    last_fine = -1
    last_update_index = -1
    connecting = []
    for i in range(len(fine_episodes)):
        if coarse_episodes[i] != last_coarse:   # if changed in high layer, need to close a group
            if len(connecting)>1:
                connections.append(connecting)
                for j in range(last_update_index, i):
                    coarse_episodes[j] = new_index
                new_index+=1
            elif len(connecting)==1:
                current_fine_idx = fine_episodes[i-1]
                for j in range(last_update_index, i):
                    coarse_episodes[j] = current_fine_idx
            last_update_index = i
            connecting = []
            last_coarse = coarse_episodes[i]
        if fine_episodes[i] != last_fine:   # if changed in low level, need to add in connecting groups 
            connecting.append(fine_episodes[i])
            last_fine = fine_episodes[i]
    if len(connecting)>1:
        connections.append(connecting)
        for j in range(last_update_index, i):
            coarse_episodes[j] = new_index
        new_index+=1
    if len(connecting)==1:
        current_fine_idx = fine_episodes[i-1]
        for j in range(last_update_index, i):
            coarse_episodes[j] = current_fine_idx
    return connections, new_index

def default_converter_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def common_hierarchical_cluster_representation(hierarchy, save_json = None, hierarchy_is_linkage_table = False):
    common_rep = {}

    if hierarchy_is_linkage_table:
        common_rep["number_steps"] = hierarchy.shape[0]+1
        common_rep["first_layer_starts"] = list(range(common_rep["number_steps"]))
        common_rep["above_layers_groups"] = []
        Z = hierarchy.copy()
        for i in range(Z.shape[0]):
            common_rep["above_layers_groups"].append([int(Z[i, 0]), int(Z[i, 1])])

    else:
        common_rep["number_steps"] = len(hierarchy[0])

        common_rep["first_layer_starts"] = get_starts_from_episodes(groups_to_episodes(hierarchy[0]))
        common_rep["above_layers_groups"] = []
        coarse_episodes = groups_to_episodes(hierarchy[0])
        idx_shift = len(np.unique(coarse_episodes))
        for i in range(1, len(hierarchy)):
            fine_episodes = coarse_episodes
            coarse_episodes = groups_to_episodes(hierarchy[i])
            connections, idx_shift = get_connections(fine_episodes, coarse_episodes,new_index=idx_shift)
            common_rep["above_layers_groups"].append(connections)
        
    if save_json:
        os.makedirs(os.path.dirname(save_json), exist_ok=True)
        with open(save_json, "w") as f:
            json.dump(common_rep, f, default=default_converter_for_json)
    return common_rep

def sequence_to_start(sequence):
    starts = []
    last = None
    for i in range(len(sequence)):
        if sequence[i]!= last:
            last = sequence[i]
            starts.append(i)
    return starts



def common_to_layers_sequences(common_rep):
    """Converts a common representation dictionary to a list of layers of sequences."""
    layers = []
    # First layer is just the first_layer_starts
    first_layer = np.zeros(common_rep["number_steps"], dtype=int)
    current_episode = 0
    current_pos = 0
    for i_start in range(1, len(common_rep["first_layer_starts"])):
        start = common_rep["first_layer_starts"][i_start]
        first_layer[current_pos:start] = current_episode
        current_pos = start
        current_episode += 1
    first_layer[current_pos:] = current_episode
    current_episode += 1
    layers.append(first_layer)

    # Above layers are constructed from the above_layers_groups
    for layer_groups in common_rep["above_layers_groups"]:
        layer = layers[-1].copy()
        assert type(layer_groups) == list, "layer_groups is not a list"
        if len(layer_groups) == 0:
            layers.append(layer)
            continue
        if type(layer_groups[0]) == list:
            for group in layer_groups:
                # if type(group) == list

                for fine_episode in group:
                    layer[layer == fine_episode] = current_episode
                current_episode += 1
            layers.append(layer)
        else:
            for fine_episode in layer_groups:
                layer[layer == fine_episode] = current_episode
            current_episode += 1
            layers.append(layer)


    return layers