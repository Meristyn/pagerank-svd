"""Módulo compartido para el proyecto PageRank vía SVD.

Contiene únicamente funciones reutilizables para:
- Generación y carga de grafos dirigidos (sintéticos y subgrafos SNAP).
- Construcción de las matrices matemáticas A, S y M_alpha = I - alpha * S.
- Resolución de PageRank mediante cuatro métodos: potencia, pseudoinversa
  exacta (pinv), SVD truncada densa (svd) y SVD truncada dispersa (svds).

Este módulo no ejecuta experimentos ni produce salidas al importarse.
"""

# Nota pedagógica: scipy.sparse no forma parte de las librerías vistas en clase; se utiliza como optimización estándar para grafos dispersos, encapsulando la lógica matricial vista en teoría.
# Nota pedagógica: networkx no fue cubierta en el curso; se utiliza solo como contenedor auxiliar de adyacencia del grafo.

from collections import deque
import gzip
import math
import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


_ZERO_NORM_TOL = 1e-12


def _validate_alpha(alpha: float) -> None:
    """Valida que el factor de amortiguamiento alpha esté en (0, 1)."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha debe satisfacer 0 < alpha < 1; se recibió {alpha}.")


def build_graph(
    kind: str,
    n: int,
    seed: int,
    dangling: bool = True,
    dangling_frac: float = 0.05,
    er_avg_deg: float = 6.0,
    ba_m: int = 3,
) -> nx.DiGraph:
    """Genera un grafo dirigido sintético Erdős–Rényi ('er') o Barabási–Albert ('ba').

    Parámetros
    ----------
    kind : str
        Familia del grafo: ``"er"`` o ``"ba"``.
    n : int
        Número de nodos (n >= 2).
    seed : int
        Semilla fija para reproducibilidad de la estructura y los nodos sin salida.
    dangling : bool, opcional
        Si es True, garantiza al menos ``ceil(dangling_frac * n)`` nodos con grado
        de salida 0. Si es False, agrega una arista saliente aleatoria a cada nodo
        sin salida. Por defecto True.
    dangling_frac : float, opcional
        Fracción mínima de nodos sin salida cuando ``dangling=True`` (en [0, 1]).
        Por defecto 0.05.
    er_avg_deg : float, opcional
        Grado de salida esperado en grafos ER (``p = min(1, er_avg_deg / n)``).
        Por defecto 6.0.
    ba_m : int, opcional
        Parámetro de enlaces por nodo nuevo en Barabási–Albert. Por defecto 3.

    Retorna
    -------
    nx.DiGraph
        Grafo dirigido con nodos etiquetados ``0, ..., n - 1``.

    Complejidad teórica
    -------------------
    O(n + |E|) en tiempo y memoria.
    """
    if kind not in {"er", "ba"}:
        raise ValueError(f"kind debe ser 'er' o 'ba'; se recibió {kind!r}.")
    if n < 2:
        raise ValueError(f"n debe ser >= 2; se recibió {n}.")
    if not (0.0 <= dangling_frac <= 1.0):
        raise ValueError(f"dangling_frac debe estar en [0, 1]; se recibió {dangling_frac}.")

    rng = np.random.default_rng(seed)

    if kind == "er":
        p = min(1.0, er_avg_deg / n)
        G = nx.fast_gnp_random_graph(n, p, seed=seed, directed=True)
    else:
        m = max(1, min(ba_m, n - 1))
        G_undir = nx.barabasi_albert_graph(n, m, seed=seed)
        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        edges = list(G_undir.edges())
        flips = rng.random(len(edges)) < 0.5
        for (u, v), forward in zip(edges, flips):
            if forward:
                G.add_edge(u, v)
            else:
                G.add_edge(v, u)

    # Ajuste controlado de nodos sin salida (dangling nodes).
    dangling_nodes = [u for u in range(n) if G.out_degree(u) == 0]
    if dangling:
        target = min(n, math.ceil(dangling_frac * n))
        if len(dangling_nodes) < target:
            non_dangling = [u for u in range(n) if G.out_degree(u) > 0]
            needed = target - len(dangling_nodes)
            chosen = rng.choice(non_dangling, size=needed, replace=False)
            for u in chosen:
                G.remove_edges_from(list(G.out_edges(int(u))))
    else:
        for u in dangling_nodes:
            v = int(rng.integers(0, n - 1))
            if v >= u:
                v += 1
            G.add_edge(u, v)

    return G


def _bfs_induced_subgraph(
    G_full: nx.DiGraph,
    start_node: int,
    max_nodes: int,
    nodes_by_degree: list[int],
) -> nx.DiGraph:
    """Extrae un subgrafo inducido de hasta max_nodes nodos mediante BFS no dirigido."""
    visited: set[int] = set()
    order: list[int] = []
    queue: deque[int] = deque([start_node])
    visited.add(start_node)
    deg_cursor = 0

    while len(order) < max_nodes:
        if not queue:
            # Si la componente conexa se agotó, continuar en otro nodo no visitado.
            while deg_cursor < len(nodes_by_degree) and nodes_by_degree[deg_cursor] in visited:
                deg_cursor += 1
            if deg_cursor >= len(nodes_by_degree):
                break
            nxt = nodes_by_degree[deg_cursor]
            visited.add(nxt)
            queue.append(nxt)

        u = queue.popleft()
        order.append(u)
        if len(order) >= max_nodes:
            break

        # Explorar vecinos salientes y entrantes en orden determinista.
        neighbors = sorted(set(G_full.successors(u)) | set(G_full.predecessors(u)))
        for v in neighbors:
            if v not in visited:
                visited.add(v)
                queue.append(v)

    return G_full.subgraph(order).copy()


def _graph_stats(G: nx.DiGraph) -> tuple[int, int, int, float]:
    """Devuelve (n, |E|, |D|, grado_medio) de un grafo dirigido."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dangling_count = sum(1 for u in G.nodes() if G.out_degree(u) == 0)
    avg_deg = m / n if n > 0 else 0.0
    return n, m, dangling_count, avg_deg


def load_snap_graph(path: str, max_nodes: int | None = None) -> nx.DiGraph:
    """Carga un grafo dirigido local en formato de lista de aristas de SNAP.

    Soporta archivos ``.txt`` y ``.txt.gz`` (ignorando líneas vacías y comentarios
    que comienzan con ``#``). Si ``max_nodes`` es menor que el tamaño del grafo
    completo, extrae un subgrafo inducido mediante BFS partiendo del nodo de mayor
    grado total. Si el subgrafo resulta excesivamente ralo (casi una estrella en
    comparación con el grafo completo), intenta BFS desde un nodo de grado mediano
    y, en última instancia, usa los primeros ``max_nodes`` IDs ordenados, emitiendo
    una advertencia explícita. Imprime estadísticas comparativas (n, |E|, |D| y
    grado medio) del subgrafo frente al grafo completo.

    Parámetros
    ----------
    path : str
        Ruta local al archivo de lista de aristas SNAP.
    max_nodes : int | None, opcional
        Número máximo de nodos del subgrafo a extraer. Si es None, usa el grafo
        completo. Por defecto None.

    Retorna
    -------
    nx.DiGraph
        Grafo dirigido con nodos reetiquetados a ``0, ..., n - 1``.

    Complejidad teórica
    -------------------
    O(|V| + |E|) para lectura y BFS.
    """
    if max_nodes is not None and max_nodes < 1:
        raise ValueError(f"max_nodes debe ser >= 1 o None; se recibió {max_nodes}.")

    opener = gzip.open if path.endswith(".gz") else open
    G_full = nx.DiGraph()
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                G_full.add_edge(int(parts[0]), int(parts[1]))

    if G_full.number_of_nodes() == 0:
        raise ValueError(f"No se encontraron aristas válidas en {path!r}.")

    n_full, e_full, d_full, avg_full = _graph_stats(G_full)

    if max_nodes is None or n_full <= max_nodes:
        sub = G_full
        strategy = "grafo_completo"
    else:
        nodes_by_deg = sorted(
            G_full.nodes(),
            key=lambda u: (G_full.in_degree(u) + G_full.out_degree(u), -u),
            reverse=True,
        )
        sub = _bfs_induced_subgraph(G_full, nodes_by_deg[0], max_nodes, nodes_by_deg)
        strategy = "bfs_grado_maximo"
        _, _, _, avg_sub = _graph_stats(sub)

        def _is_too_sparse(avg_deg_sub: float) -> bool:
            return (avg_full > 0.0 and avg_deg_sub < 0.5 * avg_full) or (
                avg_full >= 1.5 and avg_deg_sub < 1.5
            )

        if _is_too_sparse(avg_sub):
            # Alternativa 1: BFS desde un nodo cercano al grado total medio (grado >= 2).
            mean_total_deg = 2.0 * avg_full
            non_leaf = [
                u for u in nodes_by_deg if (G_full.in_degree(u) + G_full.out_degree(u)) >= 2
            ]
            pool = non_leaf if non_leaf else nodes_by_deg
            mean_deg_node = min(
                pool,
                key=lambda u: (abs((G_full.in_degree(u) + G_full.out_degree(u)) - mean_total_deg), u),
            )
            sub_mean = _bfs_induced_subgraph(G_full, mean_deg_node, max_nodes, nodes_by_deg)
            _, _, _, avg_mean = _graph_stats(sub_mean)

            # Alternativa 2: primeros max_nodes IDs ordenados.
            first_ids = sorted(G_full.nodes())[:max_nodes]
            sub_ids = G_full.subgraph(first_ids).copy()
            _, _, _, avg_ids = _graph_stats(sub_ids)

            # Elegir el subgrafo con mejor densidad relativa y avisar al usuario.
            candidates = [
                (avg_mean, "bfs_grado_medio", sub_mean),
                (avg_ids, "primeros_ids", sub_ids),
                (avg_sub, "bfs_grado_maximo", sub),
            ]
            best_avg, strategy, sub = max(candidates, key=lambda item: item[0])
            warnings.warn(
                f"load_snap_graph: el BFS desde el nodo de mayor grado resultó muy ralo "
                f"(grado medio={avg_sub:.2f} vs completo={avg_full:.2f}). "
                f"Se seleccionó la estrategia '{strategy}' (grado medio={best_avg:.2f}).",
                RuntimeWarning,
            )

    n_sub, e_sub, d_sub, avg_sub = _graph_stats(sub)
    print(
        f"[SNAP] Completo: n={n_full}, |E|={e_full}, |D|={d_full}, grado_medio={avg_full:.2f} | "
        f"Subgrafo ({strategy}): n={n_sub}, |E|={e_sub}, |D|={d_sub}, grado_medio={avg_sub:.2f}"
    )

    # Reetiquetar nodos en orden determinista a 0..n_sub-1.
    mapping = {old: idx for idx, old in enumerate(sorted(sub.nodes()))}
    return nx.relabel_nodes(sub, mapping, copy=True)


def perturb_graph(G: nx.DiGraph, frac: float = 0.01, seed: int = 0) -> nx.DiGraph:
    """Devuelve una copia de G perturbando una fracción de sus aristas al azar.

    Aproximadamente la mitad de las aristas perturbadas se eliminan de las
    existentes y la otra mitad se añaden como aristas dirigidas nuevas (u != v)
    no presentes en el grafo original. No modifica ``G``.

    Parámetros
    ----------
    G : nx.DiGraph
        Grafo dirigido original.
    frac : float, opcional
        Fracción de aristas a perturbar en [0, 1]. Por defecto 0.01 (1 %).
    seed : int, opcional
        Semilla fija para reproducibilidad. Por defecto 0.

    Retorna
    -------
    nx.DiGraph
        Copia perturbada de ``G``.

    Complejidad teórica
    -------------------
    O(|V| + |E|) en grafos dispersos.
    """
    if not (0.0 <= frac <= 1.0):
        raise ValueError(f"frac debe estar en [0, 1]; se recibió {frac}.")

    H = G.copy()
    m = H.number_of_edges()
    n = H.number_of_nodes()
    if m == 0 or frac == 0.0 or n < 2:
        return H

    rng = np.random.default_rng(seed)
    edges = sorted(H.edges())
    nodes = sorted(H.nodes())

    num_changes = max(1, round(frac * m))
    n_remove = min(num_changes // 2, m)
    max_possible_new = n * (n - 1) - len(edges)
    n_add = min(num_changes - n_remove, max(0, max_possible_new))

    if n_remove > 0:
        remove_idx = rng.choice(len(edges), size=n_remove, replace=False)
        for idx in remove_idx:
            H.remove_edge(*edges[int(idx)])

    if n_add > 0:
        # Evitar re-agregar aristas que ya existían en G o repetir nuevas.
        forbidden = set(edges)
        added = 0
        while added < n_add:
            u = nodes[int(rng.integers(0, n))]
            v = nodes[int(rng.integers(0, n))]
            if u != v and (u, v) not in forbidden:
                H.add_edge(u, v)
                forbidden.add((u, v))
                added += 1

    return H


def graph_to_adjacency(
    G: nx.DiGraph,
    sparse: bool = True,
) -> tuple[sp.csr_matrix | np.ndarray, np.ndarray]:
    """Construye la matriz de adyacencia A (A_ij = 1 si j -> i) y los grados de salida d.

    Parámetros
    ----------
    G : nx.DiGraph
        Grafo dirigido no vacío.
    sparse : bool, opcional
        Si es True devuelve A como ``scipy.sparse.csr_matrix``; si es False como
        ``np.ndarray`` denso. Por defecto True.

    Retorna
    -------
    tuple[sp.csr_matrix | np.ndarray, np.ndarray]
        Par ``(A, d)`` con ``A`` de forma ``(n, n)`` (float64) y ``d`` de forma
        ``(n,)`` conteniendo el grado de salida de cada nodo.

    Complejidad teórica
    -------------------
    O(n + |E|) en formato disperso; O(n^2) en formato denso.
    """
    n = G.number_of_nodes()
    if n == 0:
        raise ValueError("El grafo no puede estar vacío.")

    nodes = sorted(G.nodes())
    # NetworkX coloca u -> v en (u, v); transponemos para tener A_ij = 1 si j -> i.
    A_nx = nx.to_scipy_sparse_array(
        G, nodelist=nodes, dtype=np.float64, weight=None, format="csc"
    )
    A_sp = sp.csr_matrix(A_nx.T, dtype=np.float64)
    d = np.asarray(A_sp.sum(axis=0), dtype=np.float64).ravel()

    if sparse:
        return A_sp, d
    return A_sp.toarray(), d


def build_S(G: nx.DiGraph, sparse: bool = True) -> sp.csr_matrix | np.ndarray:
    """Construye la matriz estocástica por columnas S con columnas 1/n para nodos sin salida.

    Devuelve siempre la matriz matemática exacta:
        S_ij = A_ij / d_j   si d_j > 0
        S_ij = 1 / n        si d_j = 0
    tanto en formato disperso como denso.

    Parámetros
    ----------
    G : nx.DiGraph
        Grafo dirigido no vacío.
    sparse : bool, opcional
        Si es True retorna ``scipy.sparse.csr_matrix``; si es False retorna
        ``np.ndarray``. Por defecto True.

    Retorna
    -------
    sp.csr_matrix | np.ndarray
        Matriz estocástica ``S`` de tamaño ``(n, n)`` en ``float64``.

    Complejidad teórica
    -------------------
    O(|E| + |D| * n) en tiempo y memoria para formato disperso; O(n^2) para denso.
    """
    A_sp, d = graph_to_adjacency(G, sparse=True)
    n = A_sp.shape[0]

    mask = d > 0
    inv_d = np.zeros(n, dtype=np.float64)
    inv_d[mask] = 1.0 / d[mask]

    S_sp = (A_sp @ sp.diags(inv_d, format="csr")).tocsr()
    dangling_idx = np.flatnonzero(~mask)

    if sparse:
        if len(dangling_idx) > 0:
            rows = np.tile(np.arange(n, dtype=np.int32), len(dangling_idx))
            cols = np.repeat(dangling_idx.astype(np.int32), n)
            vals = np.full(rows.shape[0], 1.0 / n, dtype=np.float64)
            D_mat = sp.csc_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
            S_sp = (S_sp + D_mat).tocsr()
        return S_sp

    S_dense = S_sp.toarray()
    if len(dangling_idx) > 0:
        S_dense[:, dangling_idx] = 1.0 / n
    return S_dense


def build_M(S: sp.spmatrix | np.ndarray, alpha: float) -> sp.csr_matrix | np.ndarray:
    """Construye la matriz del sistema lineal M_alpha = I - alpha * S.

    Conserva exactamente el formato (disperso o denso) de ``S`` para garantizar
    que todos los métodos resuelvan la misma matriz matemática.

    Parámetros
    ----------
    S : sp.spmatrix | np.ndarray
        Matriz estocástica por columnas construida por ``build_S``.
    alpha : float
        Factor de amortiguamiento en (0, 1).

    Retorna
    -------
    sp.csr_matrix | np.ndarray
        Matriz ``M_alpha = I - alpha * S`` en el mismo formato que ``S``.

    Complejidad teórica
    -------------------
    O(nnz(S) + n) en formato disperso; O(n^2) en formato denso.
    """
    _validate_alpha(alpha)
    n = S.shape[0]
    if sp.issparse(S):
        return (sp.eye(n, format=S.format, dtype=np.float64) - alpha * S).asformat(S.format)
    return np.eye(n, dtype=np.float64) - alpha * np.asarray(S, dtype=np.float64)


def matvec_S_implicit(
    A: sp.spmatrix | np.ndarray,
    d: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Calcula el producto S @ x sin formar explícitamente las columnas 1/n.

    Aplica la identidad algebraica:
        S x = A (x / d_seguro) + (1 / n) * e * sum_{j in D} x_j
    evitando división por cero en los nodos con grado de salida 0.

    Parámetros
    ----------
    A : sp.spmatrix | np.ndarray
        Matriz de adyacencia (A_ij = 1 si j -> i).
    d : np.ndarray
        Vector de grados de salida de longitud n.
    x : np.ndarray
        Vector de entrada de longitud n.

    Retorna
    -------
    np.ndarray
        Vector resultante ``S @ x`` de longitud n en ``float64``.

    Complejidad teórica
    -------------------
    O(nnz(A) + n) en tiempo; O(n) en memoria auxiliar (sin almacenar |D| * n entradas).
    """
    x_arr = np.asarray(x, dtype=np.float64).ravel()
    d_arr = np.asarray(d, dtype=np.float64).ravel()
    n = x_arr.shape[0]

    mask = d_arr > 0
    y = np.zeros(n, dtype=np.float64)
    y[mask] = x_arr[mask] / d_arr[mask]

    dangling_sum = float(np.sum(x_arr[~mask]))
    return np.asarray(A @ y, dtype=np.float64).ravel() + (dangling_sum / n)


def matrix_nbytes(mat: sp.spmatrix | np.ndarray) -> int:
    """Calcula los bytes ocupados por una matriz densa o dispersa de SciPy.

    Parámetros
    ----------
    mat : sp.spmatrix | np.ndarray
        Matriz densa o dispersa (CSR, CSC o COO).

    Retorna
    -------
    int
        Tamaño total en bytes de los arreglos internos de la matriz.

    Complejidad teórica
    -------------------
    O(1).
    """
    if sp.issparse(mat):
        if isinstance(mat, (sp.csr_matrix, sp.csc_matrix, sp.csr_array, sp.csc_array)):
            return int(mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes)
        if isinstance(mat, (sp.coo_matrix, sp.coo_array)):
            return int(mat.data.nbytes + mat.row.nbytes + mat.col.nbytes)
        mat_csr = mat.tocsr()
        return int(mat_csr.data.nbytes + mat_csr.indices.nbytes + mat_csr.indptr.nbytes)
    return int(np.asarray(mat).nbytes)


def solve_power(
    S: sp.spmatrix | np.ndarray,
    alpha: float,
    tol: float = 1e-12,
    max_iter: int = 100000,
    return_iters: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Calcula el vector PageRank mediante el método de la potencia usando ``S @ x``.

    Itera ``x_{t+1} = alpha * (S @ x_t) + ((1 - alpha) / n) * e`` desde la
    distribución uniforme hasta que ``||x_{t+1} - x_t||_1 < tol``.

    Notas de convergencia
    ---------------------
    - El segundo eigenvalor de la matriz de Google satisface |lambda_2| <= alpha,
      por lo que ``ceil(log(tol) / log(alpha))`` actúa como **cota superior de
      peor caso** de iteraciones (~171 para alpha=0.85, ~2750 para alpha=0.99 y
      ~27618 para alpha=0.999 con tol=1e-12), alcanzada cuando |lambda_2(S)| ≈ 1.
      En grafos con buena mezcla (como ER o BA dispersos), el número de
      iteraciones observadas puede ser mucho menor que dicha cota.
    - Por contractividad en norma L1 con factor alpha, el criterio de parada
      ``||x_{t+1} - x_t||_1 < tol`` acota el error real respecto al punto fijo
      exacto ``pi*`` por ``||x_{t+1} - pi*||_1 <= (alpha / (1 - alpha)) * tol``.
      Por ello, para alpha -> 1 conviene usar una tolerancia más estricta al
      construir una solución de referencia.

    Parámetros
    ----------
    S : sp.spmatrix | np.ndarray
        Matriz estocástica por columnas.
    alpha : float
        Factor de amortiguamiento en (0, 1).
    tol : float, opcional
        Tolerancia en norma L1 entre iteraciones sucesivas. Por defecto 1e-12.
    max_iter : int, opcional
        Número máximo de iteraciones. Si se alcanza sin converger se emite una
        advertencia (``warnings.warn``). Por defecto 100000.
    return_iters : bool, opcional
        Si es True retorna ``(pi, iters)``. Por defecto False.

    Retorna
    -------
    np.ndarray | tuple[np.ndarray, int]
        Vector PageRank normalizado en L1, o tupla ``(pi, iters)``.

    Complejidad teórica
    -------------------
    O(T * nnz(S)) en formato disperso o O(T * n^2) en formato denso, con
    T ~ log(tol) / log(alpha).
    """
    _validate_alpha(alpha)
    if tol <= 0.0:
        raise ValueError(f"tol debe ser > 0; se recibió {tol}.")
    if max_iter < 1:
        raise ValueError(f"max_iter debe ser >= 1; se recibió {max_iter}.")

    n = S.shape[0]
    x = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - alpha) / n

    converged = False
    iters = 0
    diff = float("inf")

    for iters in range(1, max_iter + 1):
        x_new = alpha * np.asarray(S @ x, dtype=np.float64).ravel() + teleport
        diff = float(np.linalg.norm(x_new - x, ord=1))
        x = x_new
        if diff < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"solve_power no convergió en {max_iter} iteraciones "
            f"(||x_t - x_{{t-1}}||_1 = {diff:.3e} >= tol = {tol:.3e}).",
            RuntimeWarning,
        )

    pi = x / float(np.linalg.norm(x, ord=1))
    if return_iters:
        return pi, iters
    return pi


def solve_power_implicit(
    A: sp.spmatrix | np.ndarray,
    d: np.ndarray,
    alpha: float,
    tol: float = 1e-12,
    max_iter: int = 100000,
    return_iters: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Variante del método de la potencia que usa el producto implícito ``matvec_S_implicit``.

    Evita almacenar las columnas 1/n de los nodos sin salida. Mantiene el mismo
    criterio de parada (``||x_{t+1} - x_t||_1 < tol``), cota de error real
    ``~ (alpha / (1 - alpha)) * tol`` y número de iteraciones ``~ log(tol) / log(alpha)``
    que ``solve_power``.

    Parámetros
    ----------
    A : sp.spmatrix | np.ndarray
        Matriz de adyacencia (A_ij = 1 si j -> i).
    d : np.ndarray
        Vector de grados de salida de longitud n.
    alpha : float
        Factor de amortiguamiento en (0, 1).
    tol : float, opcional
        Tolerancia en norma L1. Por defecto 1e-12.
    max_iter : int, opcional
        Número máximo de iteraciones. Por defecto 100000.
    return_iters : bool, opcional
        Si es True retorna ``(pi, iters)``. Por defecto False.

    Retorna
    -------
    np.ndarray | tuple[np.ndarray, int]
        Vector PageRank normalizado en L1, o tupla ``(pi, iters)``.

    Complejidad teórica
    -------------------
    O(T * (nnz(A) + n)) en tiempo; O(nnz(A) + n) en memoria total.
    """
    _validate_alpha(alpha)
    if tol <= 0.0:
        raise ValueError(f"tol debe ser > 0; se recibió {tol}.")
    if max_iter < 1:
        raise ValueError(f"max_iter debe ser >= 1; se recibió {max_iter}.")

    n = A.shape[0]
    x = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - alpha) / n

    converged = False
    iters = 0
    diff = float("inf")

    for iters in range(1, max_iter + 1):
        x_new = alpha * matvec_S_implicit(A, d, x) + teleport
        diff = float(np.linalg.norm(x_new - x, ord=1))
        x = x_new
        if diff < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"solve_power_implicit no convergió en {max_iter} iteraciones "
            f"(||x_t - x_{{t-1}}||_1 = {diff:.3e} >= tol = {tol:.3e}).",
            RuntimeWarning,
        )

    pi = x / float(np.linalg.norm(x, ord=1))
    if return_iters:
        return pi, iters
    return pi


def solve_pinv(M: sp.spmatrix | np.ndarray, alpha: float) -> np.ndarray:
    """Resuelve M_alpha x = ((1 - alpha) / n) * e mediante ``numpy.linalg.pinv``.

    Nota sobre ``alpha``
    --------------------
    El parámetro ``alpha`` únicamente se utiliza para construir el vector derecho
    ``b = ((1 - alpha) / n) * e``; al normalizar ``pi = x / ||x||_1`` el factor
    escalar ``(1 - alpha) / n`` se cancela. Se conserva en la firma para mantener
    una interfaz uniforme entre los cuatro solvers.

    Parámetros
    ----------
    M : sp.spmatrix | np.ndarray
        Matriz ``M_alpha = I - alpha * S`` de tamaño ``(n, n)``.
    alpha : float
        Factor de amortiguamiento en (0, 1).

    Retorna
    -------
    np.ndarray
        Vector PageRank normalizado en L1 de longitud n.

    Complejidad teórica
    -------------------
    O(n^3) en tiempo (SVD densa interna) y O(n^2) en memoria.
    """
    _validate_alpha(alpha)
    M_dense = M.toarray() if sp.issparse(M) else np.asarray(M, dtype=np.float64)
    n = M_dense.shape[0]
    b = np.full(n, (1.0 - alpha) / n, dtype=np.float64)

    x = np.linalg.pinv(M_dense) @ b
    return x / float(np.linalg.norm(x, ord=1))


def solve_svd_trunc(
    M: sp.spmatrix | np.ndarray,
    alpha: float,
    k: int,
    which: str = "LM",
) -> np.ndarray:
    """Resuelve M_alpha x = b mediante pseudoinversa truncada de rango k desde ``numpy.linalg.svd``.

    Calcula ``x_k = sum_{i=1}^k sigma_i^{-1} v_i (u_i^T b)`` mediante el producto
    vectorial ``u_i^T b`` sin formar explícitamente la matriz pseudoinversa completa,
    y normaliza ``pi_k = x_k / ||x_k||_1``.

    Notas teóricas y numéricas
    --------------------------
    1. ``alpha`` solo se usa para formar ``b = ((1 - alpha) / n) * e`` y se
       cancela al normalizar en L1; se conserva por uniformidad de firmas.
    2. Para ``alpha -> 1``, como ``M_alpha^T e = (1 - alpha) e``, se cumple que
       ``u_n ≈ e / sqrt(n) ∝ b`` y ``v_n ≈ pi``, por lo que la componente
       dominante de ``x_k`` corresponde al valor singular mínimo ``sigma_min``
       (capturado por ``which="SM"`` incluso con k pequeño, mientras que
       ``which="LM"`` lo omite cuando k < n).
    3. Cuando ``k < n``, ``x_k`` puede contener entradas negativas y la suma de
       las entradas de ``pi_k`` puede ser distinta de 1 aunque ``||pi_k||_1 = 1``;
       esto refleja fielmente la fórmula truncada y no se corrige artificialmente.
    4. Si ``||x_k||_1 < 1e-12`` (por ejemplo, cancelación ortogonal exacta de
       ``u_i^T b`` en grafos regulares/ciclos con ``which="LM"`` y ``k < n``),
       se emite una advertencia antes de normalizar. En el caso degenerado en
       que ``||x_k||_1 == 0`` exactamente, se devuelve un vector de ceros para
       evitar ``NaN`` (téngase en cuenta que métricas de ranking como Kendall
       tau producirán ``NaN`` sobre un vector constante de ceros).

    Parámetros
    ----------
    M : sp.spmatrix | np.ndarray
        Matriz ``M_alpha = I - alpha * S`` de tamaño ``(n, n)``.
    alpha : float
        Factor de amortiguamiento en (0, 1).
    k : int
        Rango de truncamiento (1 <= k <= n).
    which : str, opcional
        ``"LM"`` toma los k mayores valores singulares (fórmula del enunciado);
        ``"SM"`` toma los k menores valores singulares (``Vt[-k:]``). Por defecto ``"LM"``.

    Retorna
    -------
    np.ndarray
        Vector ``pi_k = x_k / ||x_k||_1`` de longitud n.

    Complejidad teórica
    -------------------
    O(n^3) en tiempo (SVD densa completa) + O(n * k) para aplicar el truncamiento;
    O(n^2) en memoria.
    """
    _validate_alpha(alpha)
    if which not in {"LM", "SM"}:
        raise ValueError(f"which debe ser 'LM' o 'SM'; se recibió {which!r}.")

    M_dense = M.toarray() if sp.issparse(M) else np.asarray(M, dtype=np.float64)
    n = M_dense.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k debe satisfacer 1 <= k <= n ({n}); se recibió {k}.")

    U, s, Vt = np.linalg.svd(M_dense, full_matrices=False)
    if which == "LM":
        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]
    else:
        U_k = U[:, -k:]
        s_k = s[-k:]
        Vt_k = Vt[-k:, :]

    b = np.full(n, (1.0 - alpha) / n, dtype=np.float64)
    coeffs = U_k.T @ b
    x_k = Vt_k.T @ (coeffs / s_k)

    norm_l1 = float(np.linalg.norm(x_k, ord=1))
    if norm_l1 < _ZERO_NORM_TOL:
        warnings.warn(
            f"solve_svd_trunc: ||x_k||_1 = {norm_l1:.3e} < {_ZERO_NORM_TOL:.0e} "
            f"(posible cancelación ortogonal de u_i^T b con which={which!r}, k={k}).",
            RuntimeWarning,
        )
        if norm_l1 == 0.0:
            return np.zeros(n, dtype=np.float64)

    return x_k / norm_l1


def solve_svds(
    M: sp.spmatrix | np.ndarray,
    alpha: float,
    k: int,
    which: str = "LM",
    seed: int = 0,
) -> np.ndarray:
    """Resuelve M_alpha x = b mediante SVD truncada dispersa con ``scipy.sparse.linalg.svds``.

    Reordena explícitamente los valores singulares de mayor a menor (ya que
    ``svds`` los devuelve en orden ascendente), calcula ``x_k`` mediante
    ``u_i^T b`` sin formar la pseudoinversa completa y normaliza ``pi_k = x_k / ||x_k||_1``.

    Notas teóricas y numéricas
    --------------------------
    - ``alpha`` únicamente construye ``b = ((1 - alpha) / n) * e`` y se cancela
      al normalizar; se mantiene por uniformidad de firma.
    - Requiere ``0 < k < n`` (restricción de ARPACK en SciPy).
    - Cuando ``k < n``, ``x_k`` puede tener entradas negativas y suma distinta
      de 1 aunque ``||pi_k||_1 = 1``; se conserva sin corrección.
    - Si ``||x_k||_1 < 1e-12``, se emite una advertencia por posible cancelación
      ortogonal de ``u_i^T b``. Si ``||x_k||_1 == 0`` exactamente, se devuelve
      un vector de ceros para evitar ``NaN`` (métricas como Kendall tau darán
      ``NaN`` sobre un vector de ceros).

    Parámetros
    ----------
    M : sp.spmatrix | np.ndarray
        Matriz ``M_alpha = I - alpha * S`` de tamaño ``(n, n)``.
    alpha : float
        Factor de amortiguamiento en (0, 1).
    k : int
        Número de valores singulares a calcular (0 < k < n).
    which : str, opcional
        ``"LM"`` para los k mayores valores singulares o ``"SM"`` para los k
        menores. Por defecto ``"LM"``.
    seed : int, opcional
        Semilla para generar un vector inicial determinista ``v0`` en ARPACK.
        Por defecto 0.

    Retorna
    -------
    np.ndarray
        Vector ``pi_k = x_k / ||x_k||_1`` de longitud n.

    Complejidad teórica
    -------------------
    Iterativo tipo Lanczos/ARPACK: O(iter_arpack * (k * nnz(M) + k^2 * n)) en
    tiempo y O(nnz(M) + n * k) en memoria.
    """
    _validate_alpha(alpha)
    if which not in {"LM", "SM"}:
        raise ValueError(f"which debe ser 'LM' o 'SM'; se recibió {which!r}.")

    n = M.shape[0]
    if not (0 < k < n):
        raise ValueError(f"svds requiere 0 < k < n ({n}); se recibió k={k}.")

    v0 = np.random.default_rng(seed).standard_normal(n)
    U, s, Vt = spla.svds(M, k=k, which=which, v0=v0)

    # svds retorna valores singulares en orden ascendente: reordenar de mayor a menor.
    order = np.argsort(s)[::-1]
    U = U[:, order]
    s = s[order]
    Vt = Vt[order, :]

    b = np.full(n, (1.0 - alpha) / n, dtype=np.float64)
    coeffs = U.T @ b
    x_k = Vt.T @ (coeffs / s)

    norm_l1 = float(np.linalg.norm(x_k, ord=1))
    if norm_l1 < _ZERO_NORM_TOL:
        warnings.warn(
            f"solve_svds: ||x_k||_1 = {norm_l1:.3e} < {_ZERO_NORM_TOL:.0e} "
            f"(posible cancelación ortogonal de u_i^T b con which={which!r}, k={k}).",
            RuntimeWarning,
        )
        if norm_l1 == 0.0:
            return np.zeros(n, dtype=np.float64)

    return x_k / norm_l1
