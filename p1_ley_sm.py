"""Ley de decaimiento del error de la SVD truncada SM (k = 1) cuando alpha -> 1.

Para cada familia de grafos (ER, BA), tamano n y semilla, mide
    e(alpha) = || pi_SM(k=1) - pi_ref ||_1
con alpha en {0.85, 0.95, 0.99, 0.999} y ajusta la pendiente p de
    log e = p * log(1 - alpha) + c.
Guarda los resultados en resultados/ley_sm.csv y un resumen por consola.

Uso:
    python p1_ley_sm.py
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np

import pagerank_lib as pr

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
FAMILIAS = ("er", "ba")
TAMANIOS = (500, 1000)
SEMILLAS = tuple(range(10))
ALPHAS = (0.85, 0.95, 0.99, 0.999)
TOL_REF = 1e-14
MAX_ITER = 300000
DANGLING_FRAC = 0.05


def error_sm_k1(S_sp, S_dn, alpha: float) -> float:
    """Error L1 de la SVD truncada SM (k=1) frente a la potencia con tol=1e-14."""
    pi_ref = pr.solve_power(S_sp, alpha, tol=TOL_REF, max_iter=MAX_ITER)
    pi_sm = pr.solve_svd_trunc(pr.build_M(S_dn, alpha), alpha, k=1, which="SM")
    return float(np.linalg.norm(pi_sm - pi_ref, ord=1))


def main() -> None:
    RESULTADOS_DIR.mkdir(exist_ok=True)
    filas: list[dict] = []
    pendientes: dict[tuple[str, int], list[float]] = {}

    for familia in FAMILIAS:
        for n in TAMANIOS:
            for semilla in SEMILLAS:
                G = pr.build_graph(familia, n=n, seed=semilla, dangling=True,
                                   dangling_frac=DANGLING_FRAC)
                S_sp = pr.build_S(G, sparse=True)
                S_dn = pr.build_S(G, sparse=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    errores = [error_sm_k1(S_sp, S_dn, a) for a in ALPHAS]
                pendiente = float(np.polyfit(np.log([1 - a for a in ALPHAS]),
                                             np.log(errores), 1)[0])
                pendientes.setdefault((familia, n), []).append(pendiente)
                for a, e in zip(ALPHAS, errores):
                    filas.append({"graph_kind": familia, "n": n, "seed": semilla,
                                  "alpha": a, "l1_error_sm_k1": e,
                                  "pendiente_semilla": pendiente})

    with open(RESULTADOS_DIR / "ley_sm.csv", "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    print("Pendiente p de log(error) vs log(1-alpha), alpha en", ALPHAS)
    print(f"{'grafo':<6} {'n':>5} {'p media':>9} {'p std':>8}   (10 semillas)")
    for (familia, n), ps in pendientes.items():
        print(f"{familia:<6} {n:>5d} {np.mean(ps):>9.3f} {np.std(ps):>8.3f}")


if __name__ == "__main__":
    main()
