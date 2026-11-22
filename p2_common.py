import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

import pagerank_lib

CARPETA_SALIDA = "resultados/p2"
ALPHAS = [0.5, 0.85, 0.95, 0.99, 0.999]
TIPOS_GRAFO = ["er", "er_sink", "ba", "ba_sink"]
MODOS = ["LM", "SM"]
SEMILLA = 0

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def tipos_de_grafo(ruta_snap):
    if ruta_snap:
        return TIPOS_GRAFO + ["snap"]
    return TIPOS_GRAFO


# Arma el grafo con el build_graph de Persona 1.
def construir_grafo(tipo, n, ruta_snap=None):
    if tipo == "snap":
        return pagerank_lib.load_snap_graph(ruta_snap, n)
    familia = tipo.split("_")[0]
    con_nodos_sin_salida = tipo.endswith("_sink")
    return pagerank_lib.build_graph(familia, n, SEMILLA, dangling=con_nodos_sin_salida)


def construir_S_densa(grafo):
    return pagerank_lib.build_S(grafo, sparse=False)


def construir_M_densa(S, alpha):
    return pagerank_lib.build_M(S, alpha)


# Normaliza el vector en L1.
def normalizar_l1(vector):
    norma = np.abs(vector).sum()
    if norma == 0:
        return vector
    if vector.sum() >= 0:
        return vector / norma
    return -vector / norma


# Calcula el pi exacto resolviendo el sistema.
def pi_referencia(S, alpha):
    n = S.shape[0]
    lado_derecho = np.full(n, (1 - alpha) / n)
    x = np.linalg.solve(np.eye(n) - alpha * S, lado_derecho)
    return normalizar_l1(x)


# Redondea para que los empates no se rompan por ruido.
def redondear_para_ranking(vector):
    return np.round(vector / np.abs(vector).max(), 10)


def error_l1(pi_real, pi_estimado):
    return float(np.abs(pi_real - pi_estimado).sum())


# Kendall tau tratando los empates de pi_real como empates.
def kendall_tau(pi_real, pi_estimado):
    _, grupo_de_cada_nodo = np.unique(redondear_para_ranking(pi_real), return_inverse=True)
    suma_por_grupo = np.bincount(grupo_de_cada_nodo, weights=pi_estimado)
    promedio_por_grupo = suma_por_grupo / np.bincount(grupo_de_cada_nodo)
    pi_estimado_sin_ruido = promedio_por_grupo[grupo_de_cada_nodo]
    return float(kendalltau(redondear_para_ranking(pi_real), redondear_para_ranking(pi_estimado_sin_ruido))[0])


# Fraccion de los top-m reales que salen en el top-m estimado.
def precision_en_m(pi_real, pi_estimado, m):
    top_real = set(np.argsort(-redondear_para_ranking(pi_real), kind="stable")[:m])
    top_estimado = set(np.argsort(-redondear_para_ranking(pi_estimado), kind="stable")[:m])
    return len(top_real & top_estimado) / m


# Calcula el Gini de los grados.
def gini_de_grados(grados):
    grados_ordenados = np.sort(np.asarray(grados, dtype=float))
    n = len(grados_ordenados)
    if grados_ordenados.sum() == 0:
        return 0.0
    posiciones = 2 * np.arange(1, n + 1) - n - 1
    return float(posiciones @ grados_ordenados / (n * grados_ordenados.sum()))


# Hace la SVD de M y deja listos los pi truncados.
def calcular_svd(M, alpha):
    n = M.shape[0]
    U, valores_singulares, Vt = np.linalg.svd(M)
    lado_derecho = np.full(n, (1 - alpha) / n)
    aporte_de_cada_vector = Vt.T * ((U.T @ lado_derecho) / valores_singulares)
    return {"n": n, "valores_singulares": valores_singulares, "Vt": Vt,
            "LM": np.cumsum(aporte_de_cada_vector, axis=1),
            "SM": np.cumsum(aporte_de_cada_vector[:, ::-1], axis=1)}


# Saca el pi truncado con rango k.
def pi_truncado(svd, k, modo):
    return normalizar_l1(svd[modo][:, k - 1])


# Lista de k en escala log.
def malla_de_k(n, cantidad_de_puntos=60):
    return sorted(set(np.geomspace(1, n, cantidad_de_puntos).astype(int).tolist()))


# Busca el menor k con tau mayor al umbral.
def menor_k_con_tau(svd, pi_real, modo, umbral=0.95):
    k_anterior = 0
    for k in malla_de_k(svd["n"], 80):
        if kendall_tau(pi_real, pi_truncado(svd, k, modo)) > umbral:
            for k_fino in range(k_anterior + 1, k + 1):
                if kendall_tau(pi_real, pi_truncado(svd, k_fino, modo)) > umbral:
                    return k_fino
        k_anterior = k
    return None


def guardar_figura(figura, nombre):
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    figura.tight_layout()
    figura.savefig(f"{CARPETA_SALIDA}/{nombre}.png", dpi=200)
    plt.close(figura)
