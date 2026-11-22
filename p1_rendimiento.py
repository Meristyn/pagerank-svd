"""Experimentos de rendimiento (f) y robustez (g).

Uso:
    python p1_rendimiento.py
    python p1_rendimiento.py --full
    python p1_rendimiento.py --snap-path data/web-Stanford.txt
    python p1_rendimiento.py --only f

Genera tablas CSV y gráficas en la carpeta ``resultados/``.
"""

# Nota pedagógica: scipy.sparse no forma parte de las librerías vistas en clase; se utiliza como optimización estándar para grafos dispersos, encapsulando la lógica matricial vista en teoría.
# Nota de implementación: la medición empírica con módulos estándar de Python complementa el análisis teórico de flops del curso.

import argparse
from collections import defaultdict
import csv
import gc
import math
from pathlib import Path
import time
import tracemalloc
from typing import Any, Callable
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla

import pagerank_lib as pr


RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"

MAX_N_DENSE = 2000
NS_F = [100, 500, 1000, 2000, 5000]
ALPHAS_F = [0.85, 0.99, 0.999]
REPEATS_F = 5

NS_G = [100, 500, 1000]
ALPHAS_G = [0.85, 0.99]
SEEDS_G = list(range(10))

K_LM = 20
K_SM = 1
TOL_EVAL = 1e-12
TOL_REF = 1e-14
MAX_ITER = 100000
DANGLING_FRAC = 0.05

CSV_COLUMNS = [
    "graph_kind",
    "n",
    "alpha",
    "method",
    "k",
    "seed",
    "perturbation",
    "time_sec",
    "peak_mem_bytes",
    "l1_diff_ref",
    "iters",
    "iters_teoricas",
    "dangling",
    "matrix_bytes",
    "status",
    "ref",
]


def calc_iters_teoricas(alpha: float, tol: float) -> int:
    """Calcula la cota superior de peor caso ceil(log(tol) / log(alpha)).

    Esta cota corresponde al caso |lambda_2(S)| = 1; en grafos sintéticos
    dispersos (ER, BA) las iteraciones observadas suelen ser mucho menores.
    """
    return math.ceil(math.log(tol) / math.log(alpha))


def measure_solver(fn: Callable[[], Any], repeats: int) -> tuple[Any, float, int]:
    """Mide tiempo medio (perf_counter) y memoria pico (tracemalloc) solo del solver.

    Nota sobre ``tracemalloc``
    --------------------------
    ``tracemalloc`` rastrea únicamente las asignaciones gestionadas por el
    asignador de memoria de Python (incluidos los arreglos NumPy creados desde
    Python), pero **no ve el workspace interno** reservado directamente en
    C/Fortran por rutinas de LAPACK (en ``pinv``/``svd``) ni por ARPACK (en
    ``svds``). Por ello, ``peak_mem_bytes`` debe interpretarse como una cota
    inferior de la memoria temporal del solver.

    Parámetros
    ----------
    fn : Callable[[], Any]
        Función sin argumentos que invoca únicamente al solver sobre matrices
        preconstruidas.
    repeats : int
        Número de corridas cronometradas con tracemalloc desactivado.

    Retorna
    -------
    tuple[Any, float, int]
        ``(resultado, tiempo_medio_seg, memoria_pico_bytes)``.
    """
    times: list[float] = []
    result: Any = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Medición de memoria en corrida separada para no sesgar el tiempo.
    gc.collect()
    tracemalloc.start()
    result = fn()
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mean_time = float(np.mean(times))
    return result, mean_time, int(peak_mem)


def _make_row(
    graph_kind: str,
    n: int,
    alpha: float,
    method: str,
    k: int | str,
    seed: int,
    perturbation: str,
    time_sec: float | str,
    peak_mem_bytes: int | str,
    l1_diff_ref: float | str,
    iters: int | str,
    iters_teoricas: int | str,
    dangling: bool,
    matrix_bytes: int | str,
    status: str,
    ref: str,
) -> dict[str, Any]:
    """Construye una fila homogénea con las columnas exactas del CSV."""
    return {
        "graph_kind": graph_kind,
        "n": n,
        "alpha": alpha,
        "method": method,
        "k": k,
        "seed": seed,
        "perturbation": perturbation,
        "time_sec": f"{time_sec:.8g}" if isinstance(time_sec, float) else time_sec,
        "peak_mem_bytes": peak_mem_bytes,
        "l1_diff_ref": f"{l1_diff_ref:.8e}" if isinstance(l1_diff_ref, float) else l1_diff_ref,
        "iters": iters,
        "iters_teoricas": iters_teoricas,
        "dangling": dangling,
        "matrix_bytes": matrix_bytes,
        "status": status,
        "ref": ref,
    }


def _save_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Guarda una lista de diccionarios como CSV UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _run_svds_safe(
    M_sp: Any,
    alpha: float,
    k: int,
    which: str,
    seed: int,
    repeats: int,
) -> tuple[np.ndarray | None, float | str, int | str, str]:
    """Ejecuta solve_svds protegido contra errores de convergencia de ARPACK."""
    try:
        pi, mean_t, peak_m = measure_solver(
            lambda: pr.solve_svds(M_sp, alpha, k=k, which=which, seed=seed),
            repeats=repeats,
        )
        return pi, mean_t, peak_m, "ok"
    except (spla.ArpackNoConvergence, spla.ArpackError) as exc:
        warnings.warn(
            f"svds falló (n={M_sp.shape[0]}, alpha={alpha}, k={k}, which={which}): {exc}",
            RuntimeWarning,
        )
        return None, "", "", "fallo_arpack"


def _evaluate_graph_f(
    graph_kind: str,
    G: Any,
    seed: int,
    dangling: bool,
    full: bool,
    rows_f: list[dict[str, Any]],
    rows_pinv_cmp: list[dict[str, Any]],
) -> None:
    """Ejecuta las mediciones de la tarea (f) sobre un grafo dado."""
    n = G.number_of_nodes()
    run_dense = (n <= MAX_N_DENSE) or full
    dense_repeats = 1 if (n > MAX_N_DENSE and full) else REPEATS_F

    # Construcción de matrices fuera de la ventana de medición (Ajuste 2).
    A_sp, d = pr.graph_to_adjacency(G, sparse=True)
    S_sp = pr.build_S(G, sparse=True)
    S_dn = pr.build_S(G, sparse=False) if run_dense else None

    bytes_S_sp = pr.matrix_nbytes(S_sp)
    bytes_Ad = pr.matrix_nbytes(A_sp) + int(d.nbytes)

    for alpha in ALPHAS_F:
        print(f"  [f] {graph_kind} n={n} alpha={alpha} ...", flush=True)
        M_sp = pr.build_M(S_sp, alpha)
        M_dn = pr.build_M(S_dn, alpha) if S_dn is not None else None

        bytes_M_sp = pr.matrix_nbytes(M_sp)
        bytes_M_dn = pr.matrix_nbytes(M_dn) if M_dn is not None else ""

        # Referencia estricta: potencia con tol=1e-14 (error real <= alpha/(1-alpha)*tol).
        pi_ref = pr.solve_power(S_sp, alpha, tol=TOL_REF, max_iter=MAX_ITER)
        ref_label = "power_tol_1e-14"
        it_teo = calc_iters_teoricas(alpha, TOL_EVAL)

        # 1. Método de la potencia (S dispersa explícita).
        (pi_pow, it_pow), t_pow, m_pow = measure_solver(
            lambda: pr.solve_power(S_sp, alpha, tol=TOL_EVAL, max_iter=MAX_ITER, return_iters=True),
            repeats=REPEATS_F,
        )
        rows_f.append(
            _make_row(
                graph_kind, n, alpha, "power", "", seed, "none",
                t_pow, m_pow, float(np.linalg.norm(pi_pow - pi_ref, ord=1)),
                it_pow, it_teo, dangling, bytes_S_sp, "ok", ref_label,
            )
        )

        # 2. Variante potencia implícita (A y d) medida aparte para comparar memoria.
        (pi_imp, it_imp), t_imp, m_imp = measure_solver(
            lambda: pr.solve_power_implicit(
                A_sp, d, alpha, tol=TOL_EVAL, max_iter=MAX_ITER, return_iters=True
            ),
            repeats=REPEATS_F,
        )
        rows_f.append(
            _make_row(
                graph_kind, n, alpha, "power_implicit", "", seed, "none",
                t_imp, m_imp, float(np.linalg.norm(pi_imp - pi_ref, ord=1)),
                it_imp, it_teo, dangling, bytes_Ad, "ok", ref_label,
            )
        )

        # 3 y 4. Métodos densos: pinv y svd_trunc (LM, k=20).
        pi_pinv: np.ndarray | None = None
        pi_svd_lm: np.ndarray | None = None
        if run_dense and M_dn is not None:
            pi_pinv, t_pinv, m_pinv = measure_solver(
                lambda: pr.solve_pinv(M_dn, alpha),
                repeats=dense_repeats,
            )
            rows_f.append(
                _make_row(
                    graph_kind, n, alpha, "pinv", "", seed, "none",
                    t_pinv, m_pinv, float(np.linalg.norm(pi_pinv - pi_ref, ord=1)),
                    "", "", dangling, bytes_M_dn, "ok", ref_label,
                )
            )

            pi_svd_lm, t_svd, m_svd = measure_solver(
                lambda: pr.solve_svd_trunc(M_dn, alpha, k=K_LM, which="LM"),
                repeats=dense_repeats,
            )
            rows_f.append(
                _make_row(
                    graph_kind, n, alpha, "svd_trunc_lm", K_LM, seed, "none",
                    t_svd, m_svd, float(np.linalg.norm(pi_svd_lm - pi_ref, ord=1)),
                    "", "", dangling, bytes_M_dn, "ok", ref_label,
                )
            )
        else:
            rows_f.append(
                _make_row(
                    graph_kind, n, alpha, "pinv", "", seed, "none",
                    "", "", "", "", "", dangling, "", "omitido_n_grande", ref_label,
                )
            )
            rows_f.append(
                _make_row(
                    graph_kind, n, alpha, "svd_trunc_lm", K_LM, seed, "none",
                    "", "", "", "", "", dangling, "", "omitido_n_grande", ref_label,
                )
            )

        # 5. SVD truncada dispersa (svds, LM, k=20).
        pi_svds_lm, t_svds, m_svds, st_svds = _run_svds_safe(
            M_sp, alpha, k=K_LM, which="LM", seed=seed, repeats=REPEATS_F
        )
        diff_svds = (
            float(np.linalg.norm(pi_svds_lm - pi_ref, ord=1))
            if pi_svds_lm is not None
            else ""
        )
        rows_f.append(
            _make_row(
                graph_kind, n, alpha, "svds_lm", K_LM, seed, "none",
                t_svds, m_svds, diff_svds, "", "", dangling, bytes_M_sp, st_svds, ref_label,
            )
        )

        # 6. SVD truncada dispersa (svds, SM, k=1).
        pi_svds_sm, t_svds_sm, m_svds_sm, st_svds_sm = _run_svds_safe(
            M_sp, alpha, k=K_SM, which="SM", seed=seed, repeats=REPEATS_F
        )
        diff_svds_sm = (
            float(np.linalg.norm(pi_svds_sm - pi_ref, ord=1))
            if pi_svds_sm is not None
            else ""
        )
        rows_f.append(
            _make_row(
                graph_kind, n, alpha, "svds_sm", K_SM, seed, "none",
                t_svds_sm, m_svds_sm, diff_svds_sm, "", "", dangling, bytes_M_sp, st_svds_sm, ref_label,
            )
        )

        # Comparación adicional contra pinv cuando n <= MAX_N_DENSE.
        if n <= MAX_N_DENSE and pi_pinv is not None:
            for m_name, pi_vec, k_val in (
                ("power_ref_1e-14", pi_ref, ""),
                ("power", pi_pow, ""),
                ("power_implicit", pi_imp, ""),
                ("svd_trunc_lm", pi_svd_lm, K_LM),
                ("svds_lm", pi_svds_lm, K_LM),
                ("svds_sm", pi_svds_sm, K_SM),
            ):
                if pi_vec is None:
                    continue
                rows_pinv_cmp.append(
                    {
                        "graph_kind": graph_kind,
                        "n": n,
                        "alpha": alpha,
                        "method": m_name,
                        "k": k_val,
                        "seed": seed,
                        "l1_diff_pinv": f"{float(np.linalg.norm(pi_vec - pi_pinv, ord=1)):.8e}",
                        "ref": "pinv",
                    }
                )


def run_task_f(
    full: bool,
    snap_path: str | None,
    snap_max_nodes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ejecuta la tarea (f) de rendimiento en ER, BA y (opcionalmente) SNAP."""
    print("=== TAREA (f): RENDIMIENTO ===", flush=True)
    rows_f: list[dict[str, Any]] = []
    rows_pinv_cmp: list[dict[str, Any]] = []
    seed = 0

    for graph_kind in ("er", "ba"):
        for n in NS_F:
            G = pr.build_graph(
                graph_kind, n=n, seed=seed, dangling=True, dangling_frac=DANGLING_FRAC
            )
            _evaluate_graph_f(graph_kind, G, seed, True, full, rows_f, rows_pinv_cmp)

    if snap_path is not None:
        print(f"  [f] Cargando SNAP desde {snap_path} (max_nodes={snap_max_nodes}) ...", flush=True)
        G_snap = pr.load_snap_graph(snap_path, max_nodes=snap_max_nodes)
        has_dangling = any(G_snap.out_degree(u) == 0 for u in G_snap.nodes())
        _evaluate_graph_f("snap", G_snap, seed, has_dangling, full, rows_f, rows_pinv_cmp)

    return rows_f, rows_pinv_cmp


def run_task_g() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ejecuta la tarea (g) de robustez (10 semillas) y sensibilidad (perturbación 1%)."""
    print("\n=== TAREA (g): ROBUSTEZ Y SENSIBILIDAD (10 SEMILLAS) ===", flush=True)
    rows_g: list[dict[str, Any]] = []
    self_sens_rows: list[dict[str, Any]] = []

    for graph_kind in ("er", "ba"):
        for n in NS_G:
            print(f"  [g] {graph_kind} n={n} (semillas 0..9) ...", flush=True)
            for seed in SEEDS_G:
                G_orig = pr.build_graph(
                    graph_kind, n=n, seed=seed, dangling=True, dangling_frac=DANGLING_FRAC
                )
                G_pert = pr.perturb_graph(G_orig, frac=0.01, seed=seed)

                S_sp_orig = pr.build_S(G_orig, sparse=True)
                S_dn_orig = pr.build_S(G_orig, sparse=False)
                S_sp_pert = pr.build_S(G_pert, sparse=True)
                S_dn_pert = pr.build_S(G_pert, sparse=False)

                bytes_S_orig = pr.matrix_nbytes(S_sp_orig)
                bytes_S_pert = pr.matrix_nbytes(S_sp_pert)

                for alpha in ALPHAS_G:
                    M_sp_orig = pr.build_M(S_sp_orig, alpha)
                    M_dn_orig = pr.build_M(S_dn_orig, alpha)
                    M_sp_pert = pr.build_M(S_sp_pert, alpha)
                    M_dn_pert = pr.build_M(S_dn_pert, alpha)

                    pi_ref_orig = pr.solve_power(
                        S_sp_orig, alpha, tol=TOL_REF, max_iter=MAX_ITER
                    )
                    it_teo = calc_iters_teoricas(alpha, TOL_EVAL)

                    # Guardar soluciones sin perturbar por método para medir cambio propio.
                    orig_pis: dict[str, np.ndarray | None] = {}

                    for pert_label, S_sp, M_sp, M_dn, b_S, ref_label in (
                        ("none", S_sp_orig, M_sp_orig, M_dn_orig, bytes_S_orig, "power_tol_1e-14"),
                        (
                            "1pct",
                            S_sp_pert,
                            M_sp_pert,
                            M_dn_pert,
                            bytes_S_pert,
                            "power_tol_1e-14_sin_perturbar",
                        ),
                    ):
                        b_M_sp = pr.matrix_nbytes(M_sp)
                        b_M_dn = pr.matrix_nbytes(M_dn)

                        # 1. power
                        (pi_pow, it_pow), t_pow, m_pow = measure_solver(
                            lambda: pr.solve_power(
                                S_sp, alpha, tol=TOL_EVAL, max_iter=MAX_ITER, return_iters=True
                            ),
                            repeats=1,
                        )
                        rows_g.append(
                            _make_row(
                                graph_kind, n, alpha, "power", "", seed, pert_label,
                                t_pow, m_pow, float(np.linalg.norm(pi_pow - pi_ref_orig, ord=1)),
                                it_pow, it_teo, True, b_S, "ok", ref_label,
                            )
                        )

                        # 2. pinv
                        pi_pinv, t_pinv, m_pinv = measure_solver(
                            lambda: pr.solve_pinv(M_dn, alpha), repeats=1
                        )
                        rows_g.append(
                            _make_row(
                                graph_kind, n, alpha, "pinv", "", seed, pert_label,
                                t_pinv, m_pinv, float(np.linalg.norm(pi_pinv - pi_ref_orig, ord=1)),
                                "", "", True, b_M_dn, "ok", ref_label,
                            )
                        )

                        # 3. svd_trunc LM (k=20)
                        pi_svd_lm, t_slm, m_slm = measure_solver(
                            lambda: pr.solve_svd_trunc(M_dn, alpha, k=K_LM, which="LM"),
                            repeats=1,
                        )
                        rows_g.append(
                            _make_row(
                                graph_kind, n, alpha, "svd_trunc_lm", K_LM, seed, pert_label,
                                t_slm, m_slm, float(np.linalg.norm(pi_svd_lm - pi_ref_orig, ord=1)),
                                "", "", True, b_M_dn, "ok", ref_label,
                            )
                        )

                        # 4. svd_trunc SM (k=1)
                        pi_svd_sm, t_ssm, m_ssm = measure_solver(
                            lambda: pr.solve_svd_trunc(M_dn, alpha, k=K_SM, which="SM"),
                            repeats=1,
                        )
                        rows_g.append(
                            _make_row(
                                graph_kind, n, alpha, "svd_trunc_sm", K_SM, seed, pert_label,
                                t_ssm, m_ssm, float(np.linalg.norm(pi_svd_sm - pi_ref_orig, ord=1)),
                                "", "", True, b_M_dn, "ok", ref_label,
                            )
                        )

                        # 5. svds LM (k=20)
                        pi_svds_lm, t_vlm, m_vlm, st_vlm = _run_svds_safe(
                            M_sp, alpha, k=K_LM, which="LM", seed=seed, repeats=1
                        )
                        diff_vlm = (
                            float(np.linalg.norm(pi_svds_lm - pi_ref_orig, ord=1))
                            if pi_svds_lm is not None
                            else ""
                        )
                        rows_g.append(
                            _make_row(
                                graph_kind, n, alpha, "svds_lm", K_LM, seed, pert_label,
                                t_vlm, m_vlm, diff_vlm, "", "", True, b_M_sp, st_vlm, ref_label,
                            )
                        )

                        # 6. svds SM (k=1)
                        pi_svds_sm, t_vsm, m_vsm, st_vsm = _run_svds_safe(
                            M_sp, alpha, k=K_SM, which="SM", seed=seed, repeats=1
                        )
                        diff_vsm = (
                            float(np.linalg.norm(pi_svds_sm - pi_ref_orig, ord=1))
                            if pi_svds_sm is not None
                            else ""
                        )
                        rows_g.append(
                            _make_row(
                                graph_kind, n, alpha, "svds_sm", K_SM, seed, pert_label,
                                t_vsm, m_vsm, diff_vsm, "", "", True, b_M_sp, st_vsm, ref_label,
                            )
                        )

                        current_pis = {
                            "power": pi_pow,
                            "pinv": pi_pinv,
                            "svd_trunc_lm": pi_svd_lm,
                            "svd_trunc_sm": pi_svd_sm,
                            "svds_lm": pi_svds_lm,
                            "svds_sm": pi_svds_sm,
                        }
                        if pert_label == "none":
                            orig_pis = current_pis
                        else:
                            for m_name, pi_p in current_pis.items():
                                pi_o = orig_pis.get(m_name)
                                if pi_p is not None and pi_o is not None:
                                    self_sens_rows.append(
                                        {
                                            "graph_kind": graph_kind,
                                            "n": n,
                                            "alpha": alpha,
                                            "method": m_name,
                                            "seed": seed,
                                            "l1_change_1pct": f"{float(np.linalg.norm(pi_p - pi_o, ord=1)):.8e}",
                                        }
                                    )

    return rows_g, self_sens_rows


def fit_empirical_exponents(rows_f: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ajusta exponentes empíricos np.polyfit(log(n), log(t), 1) solo para n >= 500.

    Marca como ``"poco fiable"`` los ajustes basados en 3 o menos puntos y agrega,
    para los métodos densos (``pinv`` y ``svd_trunc_lm``), una fila resumen
    agrupada por ``graph_kind`` con el rango de exponentes observados entre los
    distintos valores de ``alpha``.
    """
    grouped: dict[tuple[str, float, str], list[tuple[int, float]]] = defaultdict(list)
    for r in rows_f:
        if r["status"] != "ok" or r["graph_kind"] == "snap":
            continue
        n_val = int(r["n"])
        if n_val < 500:
            continue
        grouped[(r["graph_kind"], float(r["alpha"]), r["method"])].append(
            (n_val, float(r["time_sec"]))
        )

    exponents: list[dict[str, Any]] = []
    for (gkind, alpha, method), pairs in sorted(grouped.items()):
        if len(pairs) < 2:
            continue
        pairs.sort(key=lambda x: x[0])
        ns = np.array([p[0] for p in pairs], dtype=np.float64)
        ts = np.array([p[1] for p in pairs], dtype=np.float64)
        slope, intercept = np.polyfit(np.log(ns), np.log(ts), 1)
        fiabilidad = "poco fiable" if len(pairs) <= 3 else "ok"
        exponents.append(
            {
                "graph_kind": gkind,
                "alpha": alpha,
                "method": method,
                "n_range": f"{int(ns.min())}..{int(ns.max())}",
                "n_points": len(pairs),
                "exponent": float(slope),
                "intercept": float(intercept),
                "rango_exponentes": "",
                "fiabilidad": fiabilidad,
            }
        )

    # Fila agrupada por graph_kind para métodos densos (pinv y svd_trunc_lm).
    dense_summary_rows: list[dict[str, Any]] = []
    for gkind in sorted({e["graph_kind"] for e in exponents}):
        for method in ("pinv", "svd_trunc_lm"):
            sub = [
                e for e in exponents
                if e["graph_kind"] == gkind and e["method"] == method
            ]
            if not sub:
                continue
            exp_vals = [float(e["exponent"]) for e in sub]
            min_pts = min(int(e["n_points"]) for e in sub)
            dense_summary_rows.append(
                {
                    "graph_kind": gkind,
                    "alpha": "agrupado",
                    "method": method,
                    "n_range": sub[0]["n_range"],
                    "n_points": min_pts,
                    "exponent": float(np.mean(exp_vals)),
                    "intercept": "",
                    "rango_exponentes": f"[{min(exp_vals):.3f}, {max(exp_vals):.3f}]",
                    "fiabilidad": "poco fiable" if min_pts <= 3 else "ok",
                }
            )

    exponents.extend(dense_summary_rows)
    return exponents


def plot_time_vs_n(
    rows_f: list[dict[str, Any]],
    exponents: list[dict[str, Any]],
    out_path: Path,
    full: bool,
) -> None:
    """Genera la gráfica tiempo vs. n en escala log-log."""
    exp_map = {
        (e["graph_kind"], e["alpha"], e["method"]): e["exponent"]
        for e in exponents
        if e["alpha"] != "agrupado"
    }
    methods = ["power", "power_implicit", "pinv", "svd_trunc_lm", "svds_lm", "svds_sm"]
    labels = {
        "power": "Potencia (S explícita)",
        "power_implicit": "Potencia implícita (A, d)",
        "pinv": "pinv (densa)",
        "svd_trunc_lm": f"SVD trunc. LM (k={K_LM})",
        "svds_lm": f"svds LM (k={K_LM})",
        "svds_sm": f"svds SM (k={K_SM})",
    }
    markers = {
        "power": "o",
        "power_implicit": "s",
        "pinv": "^",
        "svd_trunc_lm": "D",
        "svds_lm": "v",
        "svds_sm": "P",
    }

    fig, axes = plt.subplots(2, len(ALPHAS_F), figsize=(15, 9), sharex=True, sharey=True)
    for row_idx, gkind in enumerate(("er", "ba")):
        for col_idx, alpha in enumerate(ALPHAS_F):
            ax = axes[row_idx, col_idx]
            for method in methods:
                pts = [
                    (int(r["n"]), float(r["time_sec"]))
                    for r in rows_f
                    if r["graph_kind"] == gkind
                    and float(r["alpha"]) == alpha
                    and r["method"] == method
                    and r["status"] == "ok"
                ]
                if not pts:
                    continue
                pts.sort(key=lambda x: x[0])
                ns = [p[0] for p in pts]
                ts = [p[1] for p in pts]
                slope = exp_map.get((gkind, alpha, method))
                lbl = (
                    f"{labels[method]} (p={slope:.2f})"
                    if slope is not None
                    else labels[method]
                )
                ax.loglog(ns, ts, marker=markers[method], linewidth=1.8, label=lbl)

            # Línea de referencia teórica O(n^3) anclada al tiempo de svd_trunc_lm en n=500.
            ref_pts = [
                (int(r["n"]), float(r["time_sec"]))
                for r in rows_f
                if r["graph_kind"] == gkind
                and float(r["alpha"]) == alpha
                and r["method"] == "svd_trunc_lm"
                and r["status"] == "ok"
                and int(r["n"]) == 500
            ]
            if ref_pts:
                n0, t0 = ref_pts[0]
                n_ref = np.array([500.0, 2000.0, 5000.0])
                t_ref = t0 * (n_ref / n0) ** 3
                ax.loglog(
                    n_ref, t_ref, linestyle="--", color="gray", alpha=0.75, label="Ref. O(n³)"
                )

            ax.set_title(f"{gkind.upper()} — α = {alpha}")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            if row_idx == 1:
                ax.set_xlabel("Número de nodos n")
            if col_idx == 0:
                ax.set_ylabel("Tiempo medio del solver (s)")
            ax.legend(fontsize=7.5, loc="upper left")

    omision_nota = (
        "Incluye n=5000 denso (--full, 1 repetición)."
        if full
        else "Métodos densos omitidos en n=5000 (status=omitido_n_grande). Pendientes p ajustadas en n >= 500."
    )
    fig.suptitle(f"Rendimiento (f): Tiempo del solver vs. n (escala log-log)\n{omision_nota}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_memory_vs_n(rows_f: list[dict[str, Any]], out_path: Path, full: bool) -> None:
    """Genera la gráfica de memoria total (solver + matrix_bytes) vs. n en escala log-log."""
    alpha_plot = 0.99
    methods = ["power", "power_implicit", "pinv", "svd_trunc_lm", "svds_lm", "svds_sm"]
    labels = {
        "power": "Potencia explícita (solver + S)",
        "power_implicit": "Potencia implícita (solver + A, d)",
        "pinv": "pinv (solver + M densa)",
        "svd_trunc_lm": f"SVD trunc. LM k={K_LM} (solver + M densa)",
        "svds_lm": f"svds LM k={K_LM} (solver + M dispersa)",
        "svds_sm": f"svds SM k={K_SM} (solver + M dispersa)",
    }
    markers = {
        "power": "o",
        "power_implicit": "s",
        "pinv": "^",
        "svd_trunc_lm": "D",
        "svds_lm": "v",
        "svds_sm": "P",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for col_idx, gkind in enumerate(("er", "ba")):
        ax = axes[col_idx]
        for method in methods:
            pts = [
                (
                    int(r["n"]),
                    (int(r["peak_mem_bytes"]) + int(r["matrix_bytes"])) / (1024.0 * 1024.0),
                )
                for r in rows_f
                if r["graph_kind"] == gkind
                and float(r["alpha"]) == alpha_plot
                and r["method"] == method
                and r["status"] == "ok"
            ]
            if not pts:
                continue
            pts.sort(key=lambda x: x[0])
            ns = [p[0] for p in pts]
            mems_mb = [p[1] for p in pts]
            ax.loglog(ns, mems_mb, marker=markers[method], linewidth=1.8, label=labels[method])

        # Referencia teórica O(n^2) (matriz float64 de tamaño n x n).
        n_ref = np.array([100.0, 500.0, 1000.0, 2000.0, 5000.0])
        mem_ref = (8.0 * n_ref * n_ref) / (1024.0 * 1024.0)
        ax.loglog(n_ref, mem_ref, linestyle="--", color="gray", alpha=0.75, label="Ref. 8·n² bytes (1 matriz densa)")

        ax.set_title(f"{gkind.upper()} (α = {alpha_plot}, 5% nodos sin salida)")
        ax.set_xlabel("Número de nodos n")
        if col_idx == 0:
            ax.set_ylabel("Memoria total: pico solver + matrix_bytes (MB)")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8, loc="upper left")

    nota = (
        "Con --full (incluye n=5000 denso)."
        if full
        else "Métodos densos omitidos en n=5000 (status=omitido_n_grande)."
    )
    fig.suptitle(f"Rendimiento (f): Memoria total (solver + matriz) vs. n (escala log-log)\n{nota}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_robustness_g(
    rows_g: list[dict[str, Any]],
    self_sens_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Genera resumen gráfico de (g): fidelidad L1 en 10 semillas y sensibilidad al 1%."""
    methods_order = ["power", "pinv", "svd_trunc_lm", "svd_trunc_sm", "svds_lm", "svds_sm"]
    short_lbl = {
        "power": "power",
        "pinv": "pinv",
        "svd_trunc_lm": f"svd LM\n(k={K_LM})",
        "svd_trunc_sm": f"svd SM\n(k={K_SM})",
        "svds_lm": f"svds LM\n(k={K_LM})",
        "svds_sm": f"svds SM\n(k={K_SM})",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel izquierdo: error L1 medio +/- std frente a pi_ref en n=1000 (perturbation=none).
    ax0 = axes[0]
    x = np.arange(len(methods_order))
    width = 0.2
    combos = [("er", 0.85), ("er", 0.99), ("ba", 0.85), ("ba", 0.99)]
    for idx, (gkind, alpha) in enumerate(combos):
        means: list[float] = []
        stds: list[float] = []
        for m in methods_order:
            vals = [
                float(r["l1_diff_ref"])
                for r in rows_g
                if r["graph_kind"] == gkind
                and int(r["n"]) == 1000
                and float(r["alpha"]) == alpha
                and r["method"] == m
                and r["perturbation"] == "none"
                and r["status"] == "ok"
            ]
            means.append(float(np.mean(vals)) if vals else np.nan)
            stds.append(float(np.std(vals)) if vals else 0.0)
        offset = (idx - 1.5) * width
        ax0.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=2.5,
            label=f"{gkind.upper()} α={alpha}",
            alpha=0.85,
        )
    ax0.set_yscale("log")
    ax0.set_xticks(x)
    ax0.set_xticklabels([short_lbl[m] for m in methods_order], fontsize=8.5)
    ax0.set_ylabel("||π - π_ref||_1 (media ± desv. est., 10 semillas)")
    ax0.set_title("Error L1 respecto a π_ref (n = 1000, sin perturbación)")
    ax0.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    ax0.legend(fontsize=8)

    # Panel derecho: sensibilidad al 1% de perturbación (cambio ||π(G') - π(G)||_1 en power).
    ax1 = axes[1]
    for gkind, marker in (("er", "o"), ("ba", "s")):
        for alpha, ls in ((0.85, "-"), (0.99, "--")):
            means_s: list[float] = []
            stds_s: list[float] = []
            for n_val in NS_G:
                vals = [
                    float(r["l1_change_1pct"])
                    for r in self_sens_rows
                    if r["graph_kind"] == gkind
                    and int(r["n"]) == n_val
                    and float(r["alpha"]) == alpha
                    and r["method"] == "power"
                ]
                means_s.append(float(np.mean(vals)) if vals else np.nan)
                stds_s.append(float(np.std(vals)) if vals else 0.0)
            ax1.errorbar(
                NS_G,
                means_s,
                yerr=stds_s,
                marker=marker,
                linestyle=ls,
                capsize=4,
                linewidth=1.8,
                label=f"{gkind.upper()} α={alpha}",
            )
    ax1.set_xlabel("Número de nodos n")
    ax1.set_ylabel("||π(G') - π(G)||_1 (media ± desv. est., 10 semillas)")
    ax1.set_title("Sensibilidad de PageRank ante perturbación del 1% de aristas")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8.5)

    fig.suptitle("Robustez y sensibilidad (g) — 10 semillas fijas (0..9)", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def print_console_summary(
    rows_f: list[dict[str, Any]],
    exponents: list[dict[str, Any]],
    rows_pinv_cmp: list[dict[str, Any]],
    rows_g: list[dict[str, Any]],
    self_sens_rows: list[dict[str, Any]],
) -> None:
    """Imprime un resumen legible de resultados, memoria, iteraciones y exponentes."""
    if rows_f:
        omitted = [r for r in rows_f if r["status"] == "omitido_n_grande"]
        if omitted:
            print(
                f"\n[Reporte de omisiones] {len(omitted)} filas omitidas en n > {MAX_N_DENSE} "
                f"(status='omitido_n_grande') para métodos densos pinv y svd_trunc_lm. "
                f"Usa --full para ejecutarlos en n=5000 (1 repetición)."
            )

        arpack_fails_f = [r for r in rows_f if r["status"] == "fallo_arpack"]
        if arpack_fails_f:
            print(f"[ARPACK] Fallos en (f): {len(arpack_fails_f)} filas con status='fallo_arpack'.")

        print("\n=== ITERACIONES OBSERVADAS VS. COTA DE PEOR CASO (n=5000, tarea f) ===")
        print(f"{'grafo':<6} {'alpha':<7} {'metodo':<15} {'iters_obs':>10} {'cota_peor_caso':>15} {'tiempo(s)':>11}")
        for r in rows_f:
            if int(r["n"]) == 5000 and r["method"] in {"power", "power_implicit"} and r["status"] == "ok":
                print(
                    f"{r['graph_kind']:<6} {float(r['alpha']):<7.3f} {r['method']:<15} "
                    f"{int(r['iters']):>10d} {int(r['iters_teoricas']):>15d} {float(r['time_sec']):>11.5f}"
                )

        print("\n=== MEMORIA TOTAL (solver + matrix_bytes): S EXPLÍCITA VS. (A, d) IMPLÍCITA (alpha=0.99) ===")
        print(f"{'grafo':<6} {'n':>5} {'power_exp (MB)':>15} {'power_imp (MB)':>15} {'ahorro_MB':>11} {'factor':>8}")
        for gkind in ("er", "ba"):
            for n_val in NS_F:
                r_exp = next(
                    (
                        r for r in rows_f
                        if r["graph_kind"] == gkind and int(r["n"]) == n_val
                        and float(r["alpha"]) == 0.99 and r["method"] == "power"
                    ),
                    None,
                )
                r_imp = next(
                    (
                        r for r in rows_f
                        if r["graph_kind"] == gkind and int(r["n"]) == n_val
                        and float(r["alpha"]) == 0.99 and r["method"] == "power_implicit"
                    ),
                    None,
                )
                if r_exp and r_imp:
                    mb_exp = (int(r_exp["peak_mem_bytes"]) + int(r_exp["matrix_bytes"])) / (1024.0 ** 2)
                    mb_imp = (int(r_imp["peak_mem_bytes"]) + int(r_imp["matrix_bytes"])) / (1024.0 ** 2)
                    ratio = mb_exp / mb_imp if mb_imp > 0 else float("inf")
                    print(
                        f"{gkind:<6} {n_val:>5d} {mb_exp:>15.3f} {mb_imp:>15.3f} "
                        f"{(mb_exp - mb_imp):>11.3f} {ratio:>7.1f}x"
                    )

        if rows_pinv_cmp:
            diffs_ref = [
                float(r["l1_diff_pinv"])
                for r in rows_pinv_cmp
                if r["method"] == "power_ref_1e-14"
            ]
            if diffs_ref:
                print(
                    f"\n[Validación vs. pinv en n <= {MAX_N_DENSE}] "
                    f"max ||pi_ref(tol=1e-14) - pi_pinv||_1 = {max(diffs_ref):.3e}"
                )

        print("\n=== EXPONENTES EMPÍRICOS np.polyfit(log(n), log(t), 1) — SOLO n >= 500 ===")
        print(
            f"{'grafo':<6} {'alpha':<9} {'metodo':<15} {'rango_n':<10} {'pts':>3} "
            f"{'exponente':>10} {'vs_O(n^3)':>10} {'rango_exp':<16} {'fiabilidad':<12}"
        )
        for e in exponents:
            alpha_str = f"{e['alpha']:.3f}" if isinstance(e["alpha"], float) else str(e["alpha"])
            delta_n3 = float(e["exponent"]) - 3.0
            print(
                f"{e['graph_kind']:<6} {alpha_str:<9} {e['method']:<15} "
                f"{e['n_range']:<10} {e['n_points']:>3d} {float(e['exponent']):>10.3f} "
                f"{delta_n3:>+10.3f} {e['rango_exponentes']:<16} {e['fiabilidad']:<12}"
            )

    if rows_g:
        arpack_fails_g = [r for r in rows_g if r["status"] == "fallo_arpack"]
        print(f"\n=== RESUMEN TAREA (g): 10 SEMILLAS (n=1000, sin perturbación) | fallos ARPACK={len(arpack_fails_g)} ===")
        print(f"{'grafo':<6} {'alpha':<6} {'metodo':<14} {'tiempo_media±std (s)':>24} {'L1_ref_media±std':>24}")
        for gkind in ("er", "ba"):
            for alpha in ALPHAS_G:
                for method in ("power", "pinv", "svd_trunc_lm", "svd_trunc_sm", "svds_lm", "svds_sm"):
                    sub = [
                        r for r in rows_g
                        if r["graph_kind"] == gkind
                        and int(r["n"]) == 1000
                        and float(r["alpha"]) == alpha
                        and r["method"] == method
                        and r["perturbation"] == "none"
                        and r["status"] == "ok"
                    ]
                    if not sub:
                        continue
                    ts = [float(r["time_sec"]) for r in sub]
                    ds = [float(r["l1_diff_ref"]) for r in sub]
                    print(
                        f"{gkind:<6} {alpha:<6.2f} {method:<14} "
                        f"{np.mean(ts):>11.5f} ± {np.std(ts):<10.5f} "
                        f"{np.mean(ds):>11.3e} ± {np.std(ds):<10.3e}"
                    )

        print("\n=== SENSIBILIDAD ANTE PERTURBACIÓN DEL 1% (||pi(G') - pi(G)||_1 en power, 10 semillas) ===")
        print(f"{'grafo':<6} {'n':>5} {'alpha':<6} {'cambio_L1_media±std':>24}")
        for gkind in ("er", "ba"):
            for n_val in NS_G:
                for alpha in ALPHAS_G:
                    vals = [
                        float(r["l1_change_1pct"])
                        for r in self_sens_rows
                        if r["graph_kind"] == gkind
                        and int(r["n"]) == n_val
                        and float(r["alpha"]) == alpha
                        and r["method"] == "power"
                    ]
                    if vals:
                        print(
                            f"{gkind:<6} {n_val:>5d} {alpha:<6.2f} "
                            f"{np.mean(vals):>11.4e} ± {np.std(vals):<10.4e}"
                        )


def parse_args() -> argparse.Namespace:
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Experimentos de rendimiento (f) y robustez (g) — Persona 1."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ejecuta también pinv y svd_trunc densos en n=5000 (1 repetición).",
    )
    parser.add_argument(
        "--snap-path",
        type=str,
        default=None,
        help="Ruta opcional a archivo local de SNAP (ej. data/web-Stanford.txt).",
    )
    parser.add_argument(
        "--snap-max-nodes",
        type=int,
        default=5000,
        help="Tamaño máximo del subgrafo BFS de SNAP (por defecto 5000).",
    )
    parser.add_argument(
        "--only",
        choices=("all", "f", "g"),
        default="all",
        help="Ejecutar solo tarea (f), solo (g) o ambas (por defecto 'all').",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal de experimentos y generación de resultados."""
    args = parse_args()
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

    rows_f: list[dict[str, Any]] = []
    rows_pinv_cmp: list[dict[str, Any]] = []
    exponents: list[dict[str, Any]] = []
    rows_g: list[dict[str, Any]] = []
    self_sens_rows: list[dict[str, Any]] = []

    if args.only in ("all", "f"):
        rows_f, rows_pinv_cmp = run_task_f(
            full=args.full,
            snap_path=args.snap_path,
            snap_max_nodes=args.snap_max_nodes,
        )
        exponents = fit_empirical_exponents(rows_f)
        _save_csv(RESULTADOS_DIR / "rendimiento_f.csv", CSV_COLUMNS, rows_f)
        if rows_pinv_cmp:
            _save_csv(
                RESULTADOS_DIR / "comparacion_ref_pinv.csv",
                ["graph_kind", "n", "alpha", "method", "k", "seed", "l1_diff_pinv", "ref"],
                rows_pinv_cmp,
            )
        if exponents:
            _save_csv(
                RESULTADOS_DIR / "exponentes_empiricos_f.csv",
                [
                    "graph_kind",
                    "alpha",
                    "method",
                    "n_range",
                    "n_points",
                    "exponent",
                    "intercept",
                    "rango_exponentes",
                    "fiabilidad",
                ],
                exponents,
            )
        plot_time_vs_n(rows_f, exponents, RESULTADOS_DIR / "f_tiempo_vs_n.png", full=args.full)
        plot_memory_vs_n(rows_f, RESULTADOS_DIR / "f_memoria_vs_n.png", full=args.full)

    if args.only in ("all", "g"):
        rows_g, self_sens_rows = run_task_g()
        _save_csv(RESULTADOS_DIR / "robustez_g.csv", CSV_COLUMNS, rows_g)
        if self_sens_rows:
            _save_csv(
                RESULTADOS_DIR / "sensibilidad_g.csv",
                ["graph_kind", "n", "alpha", "method", "seed", "l1_change_1pct"],
                self_sens_rows,
            )
        plot_robustness_g(rows_g, self_sens_rows, RESULTADOS_DIR / "g_robustez_sensibilidad.png")

    print_console_summary(rows_f, exponents, rows_pinv_cmp, rows_g, self_sens_rows)
    print(f"\nResultados guardados en: {RESULTADOS_DIR}")


if __name__ == "__main__":
    main()
