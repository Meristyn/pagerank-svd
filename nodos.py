import argparse
from scipy.stats import spearmanr
from p2_common import *


# Peso de cada nodo en los ultimos vectores singulares.
def peso_en_ultimos_vectores(svd, cantidad_de_vectores):
    return np.sqrt((svd["Vt"][-cantidad_de_vectores:, :] ** 2).sum(axis=0))


# Posicion de cada nodo en el ranking.
def posiciones_en_ranking(pi):
    return np.argsort(np.argsort(-redondear_para_ranking(pi), kind="stable")) + 1


# Clasifica los nodos en categorias.
def categorias_de_nodos(grafo, grado_entrada, grado_salida):
    grado_total = grado_entrada + grado_salida
    grafo_no_dirigido = nx.Graph(grafo.to_undirected())
    grafo_no_dirigido.remove_edges_from(nx.selfloop_edges(grafo_no_dirigido))
    nodos_en_puentes = {nodo for arista in nx.bridges(grafo_no_dirigido) for nodo in arista}
    return {"sin salida": grado_salida == 0,
            "bajo grado": grado_total <= np.percentile(grado_total, 25),
            "puente": np.isin(np.arange(len(grado_total)), list(nodos_en_puentes)),
            "hub (top 5% grado entrada)": grado_entrada >= np.percentile(grado_entrada, 95)}


# Dibuja el grafo con tamano segun pi y color segun cambio de posicion.
def dibujar_grafo(grafo, pi_real, cambio_de_posicion, titulo):
    grafo_no_dirigido = nx.Graph(grafo.to_undirected())
    n = len(pi_real)
    coordenadas = nx.spring_layout(grafo_no_dirigido, seed=SEMILLA, k=1.5 / np.sqrt(n))
    limite_de_color = max(1, np.abs(cambio_de_posicion).max())
    figura, eje = plt.subplots(figsize=(9, 8))
    nx.draw_networkx_edges(grafo, coordenadas, ax=eje, alpha=0.12, arrows=False, width=0.4)
    nodos_dibujados = nx.draw_networkx_nodes(grafo, coordenadas, ax=eje, node_size=40 + 900 * pi_real / pi_real.max(),
                                             node_color=cambio_de_posicion, cmap="RdBu_r", vmin=-limite_de_color,
                                             vmax=limite_de_color, edgecolors="k", linewidths=0.3)
    plt.colorbar(nodos_dibujados, ax=eje, shrink=0.7, label="cambio de posicion (positivo = baja, negativo = sube)")
    eje.set_title(titulo)
    eje.axis("off")
    eje.grid(False)
    guardar_figura(figura, "e_grafo")


# Grafica el peso medio por categoria.
def graficar_peso_por_categoria(resumen, cantidad_de_vectores):
    figura, eje = plt.subplots(figsize=(6.5, 3.8))
    eje.bar(resumen.index, resumen.peso_medio_relativo, color="#4c78a8")
    eje.axhline(1, color="r", ls="--")
    eje.set_ylabel("peso medio / peso global")
    eje.set_title(f"Peso en los ultimos {cantidad_de_vectores} vectores singulares")
    plt.setp(eje.get_xticklabels(), rotation=20)
    guardar_figura(figura, "e_peso_por_categoria")


# Hace el analisis por nodo.
def analizar_nodos(tipo, n, alpha, k, modo, cantidad_de_vectores, ruta_snap):
    grafo = construir_grafo(tipo, n, ruta_snap)
    S = construir_S_densa(grafo)
    n = S.shape[0]
    pi_real = pi_referencia(S, alpha)
    svd = calcular_svd(construir_M_densa(S, alpha), alpha)
    k = k or max(1, n // 10)
    pi_aproximado = pi_truncado(svd, k, modo)
    grado_entrada = np.array([grafo.in_degree(nodo) for nodo in range(n)])
    grado_salida = np.array([grafo.out_degree(nodo) for nodo in range(n)])
    peso = peso_en_ultimos_vectores(svd, cantidad_de_vectores)
    posicion_real = posiciones_en_ranking(pi_real)
    posicion_aproximada = posiciones_en_ranking(pi_aproximado)
    categorias = categorias_de_nodos(grafo, grado_entrada, grado_salida)
    tabla_nodos = pd.DataFrame({"nodo": range(n), "pi_real": pi_real, "pi_aproximado": pi_aproximado,
                                "error_absoluto": np.abs(pi_real - pi_aproximado),
                                "posicion_real": posicion_real, "posicion_aproximada": posicion_aproximada,
                                "cambio_de_posicion": posicion_aproximada - posicion_real,
                                "peso_ultimos_vectores": peso, "grado_entrada": grado_entrada, "grado_salida": grado_salida})
    for nombre_categoria, pertenece in categorias.items():
        tabla_nodos[nombre_categoria] = pertenece
    tabla_nodos.to_csv(f"{CARPETA_SALIDA}/e_nodos.csv", index=False)
    resumen = pd.DataFrame({nombre: {"cantidad_de_nodos": int(pertenece.sum()),
                                     "peso_medio_relativo": peso[pertenece].mean() / peso.mean(),
                                     "error_absoluto_medio": tabla_nodos.error_absoluto[pertenece].mean(),
                                     "cambio_de_posicion_medio": np.abs(tabla_nodos.cambio_de_posicion[pertenece]).mean()}
                            for nombre, pertenece in categorias.items()}).T
    resumen.to_csv(f"{CARPETA_SALIDA}/e_resumen_categorias.csv")
    k_minimo = menor_k_con_tau(svd, pi_real, modo)
    print(f"{tipo} n={n} alpha={alpha} modo={modo} k_min(tau>0.95)={k_minimo} k usado={k} tau={kendall_tau(pi_real, pi_aproximado):.3f}")
    print(resumen.round(4).to_string())
    for nombre, valores in [("grado_entrada", grado_entrada), ("grado_salida", grado_salida),
                            ("error_absoluto", tabla_nodos.error_absoluto), ("|cambio_de_posicion|", np.abs(tabla_nodos.cambio_de_posicion))]:
        print(f"Spearman(peso_ultimos_vectores, {nombre}) = {spearmanr(peso, valores)[0]:.3f}")
    graficar_peso_por_categoria(resumen, cantidad_de_vectores)
    dibujar_grafo(grafo, pi_real, tabla_nodos.cambio_de_posicion.values, f"{tipo}, n={n}, alpha={alpha}, k={k}, {modo} (tamano proporcional a pi)")


if __name__ == "__main__":
    lector = argparse.ArgumentParser()
    lector.add_argument("--kind", default="ba_sink")
    lector.add_argument("--n", type=int, default=500)
    lector.add_argument("--alpha", type=float, default=0.99)
    lector.add_argument("--k", type=int)
    lector.add_argument("--which", default="LM", choices=["LM", "SM"])
    lector.add_argument("--vectors", type=int, default=20)
    lector.add_argument("--snap")
    argumentos = lector.parse_args()
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    analizar_nodos(argumentos.kind, argumentos.n, argumentos.alpha, argumentos.k, argumentos.which, argumentos.vectors, argumentos.snap)
