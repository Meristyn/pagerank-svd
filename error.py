import argparse
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
from p2_common import *

VALORES_DE_M = [10, 50, 100]


# Calcula error L1, tau y precision@m.
def metricas_de_ranking(pi_real, pi_estimado, n):
    metricas = {"error_l1": error_l1(pi_real, pi_estimado), "tau": kendall_tau(pi_real, pi_estimado)}
    for m in VALORES_DE_M:
        metricas[f"precision@{m}"] = precision_en_m(pi_real, pi_estimado, min(m, n))
    return metricas


# Resuelve con svds.
def pi_con_svds(M, alpha, k):
    n = M.shape[0]
    U, valores_singulares, Vt = sp_linalg.svds(sp.csr_matrix(M), k=k, which="LM")
    lado_derecho = np.full(n, (1 - alpha) / n)
    return normalizar_l1(Vt.T @ ((U.T @ lado_derecho) / valores_singulares))


# Grafica el error vs k para LM, SM y svds.
def graficar_curvas_vs_k(n, alpha, ruta_snap):
    S = construir_S_densa(construir_grafo("ba_sink", n, ruta_snap))
    M = construir_M_densa(S, alpha)
    pi_real = pi_referencia(S, alpha)
    svd = calcular_svd(M, alpha)
    filas_curva = []
    for k in malla_de_k(n):
        for modo in MODOS:
            fila = {"k": k, "modo": modo}
            fila.update(metricas_de_ranking(pi_real, pi_truncado(svd, k, modo), n))
            filas_curva.append(fila)
    curvas = pd.DataFrame(filas_curva)
    curvas.to_csv(f"{CARPETA_SALIDA}/d_curvas_k.csv", index=False)
    filas_svds = []
    for k in [k for k in malla_de_k(n, 12) if k < n - 1]:
        fila = {"k": k}
        fila.update(metricas_de_ranking(pi_real, pi_con_svds(M, alpha, k), n))
        filas_svds.append(fila)
    curvas_svds = pd.DataFrame(filas_svds)
    estilo_por_modo = {"LM": "-", "SM": "--"}
    color_por_m = {10: "#4c78a8", 50: "#f58518", 100: "#54a24b"}
    figura, ejes = plt.subplots(1, 3, figsize=(15, 4))
    for modo in MODOS:
        datos_modo = curvas[curvas.modo == modo]
        ejes[0].loglog(datos_modo.k, datos_modo.error_l1, estilo_por_modo[modo], marker="o", ms=3, label=f"trunc. {modo}")
        ejes[1].semilogx(datos_modo.k, datos_modo.tau, estilo_por_modo[modo], marker="o", ms=3, label=f"trunc. {modo}")
        for m in VALORES_DE_M:
            ejes[2].semilogx(datos_modo.k, datos_modo[f"precision@{m}"], estilo_por_modo[modo], color=color_por_m[m], label=f"{modo}, m={m}")
    ejes[0].loglog(curvas_svds.k, curvas_svds.error_l1, "s:", color="gray", label="svds (LM)")
    ejes[1].semilogx(curvas_svds.k, curvas_svds.tau, "s:", color="gray", label="svds (LM)")
    ejes[1].axhline(0.95, color="r", ls="--", lw=1)
    ejes[0].set(xlabel="k", ylabel="error L1", title=f"Error L1 (ba_sink, n={n}, alpha={alpha})")
    ejes[1].set(xlabel="k", ylabel="Kendall tau", title="Kendall tau vs k")
    ejes[2].set(xlabel="k", ylabel="precision@m", title="Precision@m vs k")
    for eje in ejes:
        eje.legend(fontsize=7)
    guardar_figura(figura, "d_error_vs_k")


# Busca el menor k con tau > 0.95 para cada caso.
def tabla_de_k_minimo(tamanios, ruta_snap):
    filas_tabla = []
    for tipo in tipos_de_grafo(ruta_snap):
        for n in tamanios:
            grafo = construir_grafo(tipo, n, ruta_snap)
            S = construir_S_densa(grafo)
            n_real = S.shape[0]
            grados_entrada = np.array([grafo.in_degree(nodo) for nodo in range(n_real)])
            grados_salida = np.array([grafo.out_degree(nodo) for nodo in range(n_real)])
            for alpha in ALPHAS:
                pi_real = pi_referencia(S, alpha)
                svd = calcular_svd(construir_M_densa(S, alpha), alpha)
                fila = {"tipo": tipo, "n": n_real, "alpha": alpha,
                        "kappa2": svd["valores_singulares"][0] / svd["valores_singulares"][-1],
                        "fraccion_sin_salida": float(np.mean(grados_salida == 0)),
                        "grado_entrada_max": int(grados_entrada.max()),
                        "gini_grado_entrada": gini_de_grados(grados_entrada)}
                for modo in MODOS:
                    k_minimo = menor_k_con_tau(svd, pi_real, modo)
                    fila[f"k_min_{modo}"] = k_minimo if k_minimo else np.nan
                    fila[f"k_min_{modo}_sobre_n"] = k_minimo / n_real if k_minimo else np.nan
                    for m in VALORES_DE_M:
                        fila[f"precision@{m}_en_k_min_{modo}"] = precision_en_m(pi_real, pi_truncado(svd, k_minimo, modo), min(m, n_real)) if k_minimo else np.nan
                filas_tabla.append(fila)
                print(tipo, n_real, alpha, "k_min LM/SM =", fila["k_min_LM"], fila["k_min_SM"])
    return pd.DataFrame(filas_tabla)


# Grafica k minimo vs alpha y vs n.
def graficar_k_minimo(tabla):
    n_maximo = tabla.n.max()
    estilo_por_modo = {"LM": "--", "SM": "-"}
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
    for indice, tipo in enumerate(tabla.tipo.unique()):
        color = plt.cm.tab10(indice)
        for modo in MODOS:
            datos_alpha = tabla[(tabla.tipo == tipo) & (tabla.n == n_maximo)]
            datos_n = tabla[(tabla.tipo == tipo) & (tabla.alpha == 0.99)]
            ejes[0].plot(datos_alpha.alpha, datos_alpha[f"k_min_{modo}_sobre_n"], estilo_por_modo[modo], marker="o", color=color, label=f"{tipo} {modo}")
            ejes[1].loglog(datos_n.n, datos_n[f"k_min_{modo}"], estilo_por_modo[modo], marker="o", color=color, label=f"{tipo} {modo}")
    ejes[0].set(xlabel="alpha", ylabel="k_min / n", title=f"k minimo (tau>0.95) vs alpha, n={n_maximo}")
    ejes[1].set(xlabel="n", ylabel="k_min", title="k minimo vs n (alpha=0.99)")
    ejes[0].legend(fontsize=7, ncol=2)
    guardar_figura(figura, "d_kmin_escala")


if __name__ == "__main__":
    lector = argparse.ArgumentParser()
    lector.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000, 2000])
    lector.add_argument("--snap")
    argumentos = lector.parse_args()
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    graficar_curvas_vs_k(min(1000, max(argumentos.sizes)), 0.99, argumentos.snap)
    tabla_k_minimo = tabla_de_k_minimo(argumentos.sizes, argumentos.snap)
    tabla_k_minimo.to_csv(f"{CARPETA_SALIDA}/d_kmin_tabla.csv", index=False)
    graficar_k_minimo(tabla_k_minimo)
    print(tabla_k_minimo.round(4).to_string())
