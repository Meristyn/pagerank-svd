"""Diagnostico espectral de M_alpha en ER, BA y (opcionalmente) SNAP.

Para cada grafo y alpha calcula, con la SVD densa de M_alpha:
  * sigma_min y su cociente con (1 - alpha);
  * la prediccion 1 / (sqrt(n) * ||pi||_2), que viene de la cota
        sigma_min <= ||M pi|| / ||pi|| = (1 - alpha) / (sqrt(n) * ||pi||_2) <= 1 - alpha
    (||M pi||_2 = ||b||_2 = (1 - alpha)/sqrt(n) y ||pi||_2 >= 1/sqrt(n));
  * kappa_2 = sigma_max / sigma_min;
  * la alineacion de u_n con e/sqrt(n) y de v_n con pi;
  * el error L1 de la SVD truncada SM para k = 1, 2, 3 y 5.

Salida: resultados/diagnostico_espectral.csv y una tabla por consola.

Uso:
    python p1_diagnostico_espectral.py
    python p1_diagnostico_espectral.py --snap-path data/web-Stanford.txt.gz --n 2000
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np

import pagerank_lib as pr

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
ALPHAS = (0.85, 0.99, 0.999)
KS = (1, 2, 3, 5)
TOL_REF = 1e-14
MAX_ITER = 300000


def diagnosticar(nombre: str, G, alphas=ALPHAS) -> list[dict]:
    """Devuelve una fila de diagnostico por cada alpha para el grafo G."""
    S_sp = pr.build_S(G, sparse=True)
    S_dn = pr.build_S(G, sparse=False)
    n = S_dn.shape[0]
    e_norm = np.ones(n) / np.sqrt(n)
    filas = []
    for alpha in alphas:
        M = pr.build_M(S_dn, alpha)
        U, s, Vt = np.linalg.svd(M)
        pi = pr.solve_power(S_sp, alpha, tol=TOL_REF, max_iter=MAX_ITER)
        prediccion = 1.0 / (np.sqrt(n) * np.linalg.norm(pi))
        fila = {
            "graph": nombre, "n": n, "alpha": alpha,
            "sigma_min": float(s[-1]),
            "sigma_min_sobre_1ma": float(s[-1] / (1 - alpha)),
            "prediccion_1_sobre_sqrtn_norma2": float(prediccion),
            "sigma_segundo_menor": float(s[-2]),
            "kappa2": float(s[0] / s[-1]),
            "align_un_e": float(abs(U[:, -1] @ e_norm)),
            "align_vn_pi": float(abs(Vt[-1] @ pi) / np.linalg.norm(pi)),
        }
        for k in KS:
            pi_k = pr.solve_svd_trunc(M, alpha, k, "SM")
            fila[f"err_SM_k{k}"] = float(np.abs(pi_k - pi).sum())
        filas.append(fila)
    return filas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=2000, help="tamano de los grafos")
    parser.add_argument("--snap-path", default=None, help="archivo web-Stanford (.txt o .txt.gz)")
    args = parser.parse_args()

    grafos = [
        ("er", pr.build_graph("er", args.n, 0, True)),
        ("ba", pr.build_graph("ba", args.n, 0, True)),
    ]
    if args.snap_path:
        grafos.append(("snap", pr.load_snap_graph(args.snap_path, args.n)))

    filas: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nombre, G in grafos:
            filas.extend(diagnosticar(nombre, G))

    RESULTADOS_DIR.mkdir(exist_ok=True)
    with open(RESULTADOS_DIR / "diagnostico_espectral.csv", "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"{'grafo':<5} {'n':>5} {'alpha':>6} {'smin/(1-a)':>11} {'predic.':>8} {'kappa2':>9} "
          f"{'|un.e|':>7} {'|vn.pi|':>8}  " + " ".join(f"{'SM k=' + str(k):>9}" for k in KS))
    for f in filas:
        print(f"{f['graph']:<5} {f['n']:>5d} {f['alpha']:>6.3f} {f['sigma_min_sobre_1ma']:>11.3f} "
              f"{f['prediccion_1_sobre_sqrtn_norma2']:>8.3f} {f['kappa2']:>9.0f} "
              f"{f['align_un_e']:>7.4f} {f['align_vn_pi']:>8.4f}  "
              + " ".join(f"{f[f'err_SM_k{k}']:>9.1e}" for k in KS))


if __name__ == "__main__":
    main()
