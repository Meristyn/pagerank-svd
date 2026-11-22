import argparse
from p2_common import *

TOLERANCIAS = [1e-4, 1e-8, 1e-12]


# Cuenta los sigma mayores a tolerancia por sigma_max.
def rango_efectivo(valores_singulares, tolerancia):
    return int(np.sum(valores_singulares > tolerancia * valores_singulares[0]))


# Arma la fila de la tabla para un tipo y alpha.
def fila_de_espectro(tipo, n, alpha, valores_singulares):
    fila = {"tipo": tipo, "n": n, "alpha": alpha,
            "sigma_max": valores_singulares[0], "sigma_min": valores_singulares[-1],
            "kappa2": valores_singulares[0] / valores_singulares[-1]}
    for tolerancia in TOLERANCIAS:
        fila[f"rango_tol_{tolerancia:g}"] = rango_efectivo(valores_singulares, tolerancia)
    return fila


# Grafica el espectro y kappa_2 y guarda la tabla.
def analizar_espectro(n, ruta_snap):
    tipos = tipos_de_grafo(ruta_snap)
    columnas = 2
    filas_de_ejes = (len(tipos) + 1) // 2
    figura_sigma, ejes = plt.subplots(filas_de_ejes, columnas, figsize=(10, 3.5 * filas_de_ejes), sharey=True, squeeze=False)
    filas_tabla = []
    kappas_por_tipo = {}
    for eje, tipo in zip(ejes.ravel(), tipos):
        S = construir_S_densa(construir_grafo(tipo, n, ruta_snap))
        kappas_por_tipo[tipo] = []
        for alpha in ALPHAS:
            valores_singulares = np.linalg.svd(construir_M_densa(S, alpha), compute_uv=False)
            eje.semilogy(np.arange(1, len(valores_singulares) + 1), valores_singulares, label=f"alpha={alpha}")
            kappas_por_tipo[tipo].append(valores_singulares[0] / valores_singulares[-1])
            filas_tabla.append(fila_de_espectro(tipo, len(S), alpha, valores_singulares))
        eje.set(title=f"{tipo} (n={len(S)})", xlabel="i", ylabel="sigma_i")
    ejes[0, 0].legend(fontsize=8)
    guardar_figura(figura_sigma, "c_espectro")
    figura_kappa, eje_kappa = plt.subplots(figsize=(6, 4))
    for tipo in tipos:
        eje_kappa.semilogy(ALPHAS, kappas_por_tipo[tipo], "o-", label=tipo)
    eje_kappa.set(xlabel="alpha", ylabel="kappa_2(M_alpha)", title="Condicionamiento vs alpha")
    eje_kappa.legend()
    guardar_figura(figura_kappa, "c_kappa_vs_alpha")
    tabla = pd.DataFrame(filas_tabla)
    tabla.to_csv(f"{CARPETA_SALIDA}/c_tabla.csv", index=False)
    print(tabla.round(4).to_string())


if __name__ == "__main__":
    lector = argparse.ArgumentParser()
    lector.add_argument("--n", type=int, default=500)
    lector.add_argument("--snap")
    argumentos = lector.parse_args()
    analizar_espectro(argumentos.n, argumentos.snap)
