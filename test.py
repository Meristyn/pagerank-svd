"""Suite de validaciones para las tareas a y b.

Ejecutable directamente con:
    python test.py
"""

import gzip
import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout

import networkx as nx
import numpy as np
import scipy.sparse as sp

import pagerank_lib as pr


def _build_handcrafted_graphs() -> dict[str, nx.DiGraph]:
    """Construye grafos pequeños a mano para probar casos borde."""
    graphs: dict[str, nx.DiGraph] = {}

    # 1. Nodo aislado (nodo 3 sin aristas entrantes ni salientes).
    g_iso = nx.DiGraph()
    g_iso.add_nodes_from([0, 1, 2, 3])
    g_iso.add_edges_from([(0, 1), (1, 2), (2, 0)])
    graphs["nodo_aislado"] = g_iso

    # 2. Múltiples nodos sin salida (nodos 3 y 4 son sumideros).
    g_dang = nx.DiGraph()
    g_dang.add_nodes_from([0, 1, 2, 3, 4])
    g_dang.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4)])
    graphs["nodos_sin_salida"] = g_dang

    # 3. Grafo con ciclo puro (solución analítica exacta: uniforme 1/4).
    g_cycle = nx.DiGraph()
    g_cycle.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    graphs["ciclo_puro"] = g_cycle

    # 4. Grafo con ciclo y cuerda asimétrica.
    g_chord = nx.DiGraph()
    g_chord.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    graphs["ciclo_con_cuerda"] = g_chord

    # 5. Grafo desconectado (dos componentes dirigidas separadas).
    g_disc = nx.DiGraph()
    g_disc.add_edges_from([(0, 1), (1, 0), (2, 3), (3, 4), (4, 2)])
    graphs["desconectado"] = g_disc

    return graphs


class TestPersona1(unittest.TestCase):
    """Pruebas unitarias y de validación matemática para pagerank_lib."""

    def test_import_sin_efectos_secundarios(self) -> None:
        """Criterio 5: pagerank_lib no imprime ni ejecuta nada al importarse."""
        proc = subprocess.run(
            [sys.executable, "-c", "import pagerank_lib"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(proc.stdout, "")
        self.assertEqual(proc.stderr, "")

    def test_casos_borde_grafos_a_mano(self) -> None:
        """Criterio 4: casos borde sin división por cero y concordancia de métodos."""
        alpha = 0.85
        rng = np.random.default_rng(42)

        for name, G in _build_handcrafted_graphs().items():
            with self.subTest(grafo=name):
                n = G.number_of_nodes()
                A_sp, d = pr.graph_to_adjacency(G, sparse=True)
                S_sp = pr.build_S(G, sparse=True)
                S_dn = pr.build_S(G, sparse=False)
                M_sp = pr.build_M(S_sp, alpha)
                M_dn = pr.build_M(S_dn, alpha)

                # Igualdad dispersa vs densa y columnas estocásticas.
                self.assertTrue(sp.issparse(S_sp))
                self.assertTrue(sp.issparse(M_sp))
                self.assertIsInstance(S_dn, np.ndarray)
                self.assertIsInstance(M_dn, np.ndarray)
                np.testing.assert_allclose(S_sp.toarray(), S_dn, atol=1e-14)
                np.testing.assert_allclose(S_dn.sum(axis=0), np.ones(n), atol=1e-14)
                np.testing.assert_allclose(M_sp.toarray(), M_dn, atol=1e-14)

                # Columnas de nodos sin salida iguales a 1/n.
                for j in np.flatnonzero(d == 0):
                    np.testing.assert_allclose(S_dn[:, j], np.full(n, 1.0 / n), atol=1e-14)

                # Producto implícito coincide con S @ x.
                x = rng.random(n)
                np.testing.assert_allclose(
                    pr.matvec_S_implicit(A_sp, d, x),
                    S_sp @ x,
                    atol=1e-14,
                )

                # Solvers exactos (k = n) deben coincidir con tol 1e-8 en norma L1.
                pi_pow = pr.solve_power(S_sp, alpha, tol=1e-12)
                pi_pow_imp = pr.solve_power_implicit(A_sp, d, alpha, tol=1e-12)
                pi_pinv = pr.solve_pinv(M_dn, alpha)
                pi_svd = pr.solve_svd_trunc(M_dn, alpha, k=n)

                for pi in (pi_pow, pi_pow_imp, pi_pinv, pi_svd):
                    self.assertTrue(np.all(np.isfinite(pi)))
                    self.assertTrue(np.all(pi >= 0.0))
                    self.assertAlmostEqual(float(np.sum(pi)), 1.0, delta=1e-12)

                self.assertLess(np.linalg.norm(pi_pow - pi_pow_imp, ord=1), 1e-12)
                self.assertLess(np.linalg.norm(pi_pow - pi_pinv, ord=1), 1e-8)
                self.assertLess(np.linalg.norm(pi_pow - pi_svd, ord=1), 1e-8)
                self.assertLess(np.linalg.norm(pi_pinv - pi_svd, ord=1), 1e-12)

                # En el ciclo puro, la solución analítica exacta es uniforme 1/n.
                if name == "ciclo_puro":
                    np.testing.assert_allclose(pi_pow, np.full(n, 1.0 / n), atol=1e-12)

    def test_advertencia_cancelacion_ortogonal_ciclo_puro(self) -> None:
        """Ajuste 1: ||x_k||_1 < 1e-12 dispara advertencia en ciclo puro con LM y k < n."""
        G = _build_handcrafted_graphs()["ciclo_puro"]
        S_dn = pr.build_S(G, sparse=False)
        M_dn = pr.build_M(S_dn, alpha=0.85)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pi_k = pr.solve_svd_trunc(M_dn, alpha=0.85, k=2, which="LM")

        self.assertTrue(any("||x_k||_1" in str(w.message) for w in caught))
        self.assertTrue(np.all(np.isfinite(pi_k)))

    def test_build_graph_y_dangling(self) -> None:
        """Verifica ER y BA con/sin nodos sin salida y reproducibilidad con semillas."""
        for kind in ("er", "ba"):
            for frac in (0.05, 0.20):
                G1 = pr.build_graph(kind, n=100, seed=7, dangling=True, dangling_frac=frac)
                G2 = pr.build_graph(kind, n=100, seed=7, dangling=True, dangling_frac=frac)
                G3 = pr.build_graph(kind, n=100, seed=8, dangling=True, dangling_frac=frac)

                self.assertEqual(G1.number_of_nodes(), 100)
                self.assertEqual(sorted(G1.edges()), sorted(G2.edges()))
                self.assertNotEqual(sorted(G1.edges()), sorted(G3.edges()))

                dangling_count = sum(1 for u in G1.nodes() if G1.out_degree(u) == 0)
                self.assertGreaterEqual(dangling_count, math.ceil(frac * 100))

            G_nodang = pr.build_graph(kind, n=100, seed=7, dangling=False)
            dangling_zero = sum(1 for u in G_nodang.nodes() if G_nodang.out_degree(u) == 0)
            self.assertEqual(dangling_zero, 0)

    def test_perturb_graph(self) -> None:
        """Verifica perturbación del 1% (50% quitar / 50% añadir) sin mutar G."""
        G = pr.build_graph("er", n=200, seed=1, dangling=True)
        orig_edges = set(G.edges())

        H1 = pr.perturb_graph(G, frac=0.01, seed=123)
        H2 = pr.perturb_graph(G, frac=0.01, seed=123)

        self.assertEqual(set(G.edges()), orig_edges)
        self.assertEqual(set(H1.edges()), set(H2.edges()))

        new_edges = set(H1.edges())
        removed = len(orig_edges - new_edges)
        added = len(new_edges - orig_edges)
        expected_changes = max(1, round(0.01 * len(orig_edges)))
        self.assertEqual(removed + added, expected_changes)
        self.assertEqual(removed, expected_changes // 2)
        self.assertEqual(added, expected_changes - removed)

    def test_validacion_metodos_grafos_sinteticos(self) -> None:
        """Criterios 1 y 3: validación k = n en ER y BA con tol 1e-8 en norma L1."""
        alpha = 0.85
        n = 100
        rng = np.random.default_rng(0)

        for kind in ("er", "ba"):
            for dangling in (True, False):
                with self.subTest(kind=kind, dangling=dangling):
                    G = pr.build_graph(kind, n=n, seed=0, dangling=dangling)
                    A_sp, d = pr.graph_to_adjacency(G, sparse=True)
                    S_sp = pr.build_S(G, sparse=True)
                    S_dn = pr.build_S(G, sparse=False)
                    M_sp = pr.build_M(S_sp, alpha)
                    M_dn = pr.build_M(S_dn, alpha)

                    # Verificar igualdad S dispersa/densa y producto implícito.
                    np.testing.assert_allclose(S_sp.toarray(), S_dn, atol=1e-14)
                    x = rng.random(n)
                    np.testing.assert_allclose(
                        pr.matvec_S_implicit(A_sp, d, x),
                        S_sp @ x,
                        atol=1e-14,
                    )

                    if dangling:
                        # Con nodos sin salida, (A, d) ocupa menos bytes que S explícita.
                        bytes_imp = pr.matrix_nbytes(A_sp) + int(d.nbytes)
                        bytes_exp = pr.matrix_nbytes(S_sp)
                        self.assertLess(bytes_imp, bytes_exp)

                    pi_pow, iters = pr.solve_power(S_sp, alpha, tol=1e-12, return_iters=True)
                    pi_imp = pr.solve_power_implicit(A_sp, d, alpha, tol=1e-12)
                    pi_pinv = pr.solve_pinv(M_dn, alpha)
                    pi_svd_full = pr.solve_svd_trunc(M_dn, alpha, k=n, which="LM")

                    self.assertGreater(iters, 10)
                    for pi in (pi_pow, pi_imp, pi_pinv, pi_svd_full):
                        self.assertTrue(np.all(np.isfinite(pi)))
                        self.assertTrue(np.all(pi >= 0.0))
                        self.assertAlmostEqual(float(np.sum(pi)), 1.0, delta=1e-12)

                    self.assertLess(np.linalg.norm(pi_pow - pi_imp, ord=1), 1e-12)
                    self.assertLess(np.linalg.norm(pi_pow - pi_pinv, ord=1), 1e-8)
                    self.assertLess(np.linalg.norm(pi_pow - pi_svd_full, ord=1), 1e-8)
                    self.assertLess(np.linalg.norm(pi_pinv - pi_svd_full, ord=1), 1e-11)

                    # Verificar k < n en solve_svd_trunc y solve_svds (LM y SM).
                    k = 20
                    pi_trunc_lm = pr.solve_svd_trunc(M_dn, alpha, k=k, which="LM")
                    pi_svds_lm = pr.solve_svds(M_sp, alpha, k=k, which="LM", seed=0)
                    pi_trunc_sm = pr.solve_svd_trunc(M_dn, alpha, k=1, which="SM")
                    pi_svds_sm = pr.solve_svds(M_sp, alpha, k=1, which="SM", seed=0)

                    for pi_k in (pi_trunc_lm, pi_svds_lm, pi_trunc_sm, pi_svds_sm):
                        self.assertTrue(np.all(np.isfinite(pi_k)))
                        self.assertAlmostEqual(float(np.linalg.norm(pi_k, ord=1)), 1.0, delta=1e-12)

                    # svds y svd_trunc deben coincidir para los mismos k valores singulares.
                    self.assertLess(np.linalg.norm(pi_trunc_lm - pi_svds_lm, ord=1), 1e-10)
                    self.assertLess(np.linalg.norm(pi_trunc_sm - pi_svds_sm, ord=1), 1e-10)

    def test_svds_restriccion_y_k_igual_n_menos_1(self) -> None:
        """Verifica restricción 0 < k < n y prueba explícitamente k = n - 1 en svds."""
        n = 30
        alpha = 0.85
        G = pr.build_graph("er", n=n, seed=3, dangling=True)
        S_sp = pr.build_S(G, sparse=True)
        M_sp = pr.build_M(S_sp, alpha)
        M_dn = M_sp.toarray()

        with self.assertRaises(ValueError):
            pr.solve_svds(M_sp, alpha, k=0)
        with self.assertRaises(ValueError):
            pr.solve_svds(M_sp, alpha, k=n)

        pi_svds_nm1 = pr.solve_svds(M_sp, alpha, k=n - 1, which="LM", seed=0)
        pi_svd_nm1 = pr.solve_svd_trunc(M_dn, alpha, k=n - 1, which="LM")

        self.assertTrue(np.all(np.isfinite(pi_svds_nm1)))
        self.assertAlmostEqual(float(np.linalg.norm(pi_svds_nm1, ord=1)), 1.0, delta=1e-12)
        self.assertLess(np.linalg.norm(pi_svds_nm1 - pi_svd_nm1, ord=1), 1e-10)

    def test_componente_dominante_sigma_min_alpha_cerca_de_1(self) -> None:
        """Verifica que para alpha=0.99, SM con k=1 aproxima mucho mejor que LM con k=20."""
        G = pr.build_graph("ba", n=100, seed=5, dangling=True)
        alpha = 0.99
        S_sp = pr.build_S(G, sparse=True)
        M_sp = pr.build_M(S_sp, alpha)
        M_dn = M_sp.toarray()

        pi_ref, iters_99 = pr.solve_power(S_sp, alpha, tol=1e-14, return_iters=True)
        _, iters_85 = pr.solve_power(S_sp, 0.85, tol=1e-14, return_iters=True)
        self.assertGreater(iters_99, iters_85)

        # En un ciclo dirigido con cola (0->1->2->0 y 3->0), |lambda_2(S)| = 1
        # y el número de iteraciones sigue exactamente ~ log(tol)/log(alpha).
        G_cyc_tail = nx.DiGraph([(0, 1), (1, 2), (2, 0), (3, 0)])
        S_ct = pr.build_S(G_cyc_tail, sparse=True)
        _, it_ct_85 = pr.solve_power(S_ct, 0.85, tol=1e-12, return_iters=True)
        _, it_ct_99 = pr.solve_power(S_ct, 0.99, tol=1e-12, return_iters=True)
        self.assertGreater(it_ct_99, 10 * it_ct_85)
        self.assertGreater(it_ct_99, 2000)

        pi_lm = pr.solve_svd_trunc(M_dn, alpha, k=20, which="LM")
        pi_sm = pr.solve_svd_trunc(M_dn, alpha, k=1, which="SM")
        pi_svds_sm = pr.solve_svds(M_sp, alpha, k=1, which="SM", seed=0)

        err_lm = float(np.linalg.norm(pi_lm - pi_ref, ord=1))
        err_sm = float(np.linalg.norm(pi_sm - pi_ref, ord=1))
        err_svds_sm = float(np.linalg.norm(pi_svds_sm - pi_ref, ord=1))

        self.assertLess(err_sm, err_lm)
        self.assertLess(err_sm, 0.05)
        self.assertLess(np.linalg.norm(pi_sm - pi_svds_sm, ord=1), 1e-9)
        self.assertLess(err_svds_sm, 0.05)

    def test_advertencia_no_convergencia_potencia(self) -> None:
        """Verifica que solve_power y solve_power_implicit advierten si no convergen."""
        G = pr.build_graph("er", n=50, seed=2, dangling=True)
        A_sp, d = pr.graph_to_adjacency(G, sparse=True)
        S_sp = pr.build_S(G, sparse=True)

        with warnings.catch_warnings(record=True) as caught_exp:
            warnings.simplefilter("always")
            pr.solve_power(S_sp, alpha=0.99, tol=1e-12, max_iter=5)
        self.assertTrue(any("no convergió" in str(w.message) for w in caught_exp))

        with warnings.catch_warnings(record=True) as caught_imp:
            warnings.simplefilter("always")
            pr.solve_power_implicit(A_sp, d, alpha=0.99, tol=1e-12, max_iter=5)
        self.assertTrue(any("no convergió" in str(w.message) for w in caught_imp))

    def test_load_snap_graph_sintetico_y_fallback_estrella(self) -> None:
        """Prueba load_snap_graph con archivos sintéticos (.txt y .txt.gz) y detección de estrella."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Caso 1: grafo sintético conexo y denso en .txt con comentarios e IDs no contiguos.
            G_syn = pr.build_graph("er", n=80, seed=10, dangling=True, er_avg_deg=8.0)
            txt_path = os.path.join(tmpdir, "snap_normal.txt")
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write("# Directed graph sintético para prueba\n")
                fh.write("# FromNodeId\tToNodeId\n")
                for u, v in G_syn.edges():
                    fh.write(f"{1000 + u * 7}\t{1000 + v * 7}\n")

            buf = io.StringIO()
            with redirect_stdout(buf):
                sub = pr.load_snap_graph(txt_path, max_nodes=40)

            self.assertEqual(sub.number_of_nodes(), 40)
            self.assertEqual(sorted(sub.nodes()), list(range(40)))
            self.assertIn("[SNAP]", buf.getvalue())

            # Caso 2: archivo .txt.gz donde el nodo de mayor grado forma una estrella pura
            # (grado medio ~1), mientras otra componente es mucho más densa -> dispara fallback.
            gz_path = os.path.join(tmpdir, "snap_estrella.txt.gz")
            with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
                fh.write("# Grafo con estrella gigante + bloque denso\n")
                # Estrella de 50 nodos (centro 0 conectado a hojas 1..49).
                for leaf in range(1, 50):
                    fh.write(f"0\t{leaf}\n")
                # Bloque denso en nodos 100..139 (cada nodo enlaza a 12 vecinos).
                for u in range(100, 140):
                    for step in range(1, 13):
                        v = 100 + ((u - 100 + step) % 40)
                        fh.write(f"{u}\t{v}\n")

            buf2 = io.StringIO()
            with warnings.catch_warnings(record=True) as caught, redirect_stdout(buf2):
                warnings.simplefilter("always")
                sub_fb = pr.load_snap_graph(gz_path, max_nodes=35)

            self.assertEqual(sub_fb.number_of_nodes(), 35)
            self.assertEqual(sorted(sub_fb.nodes()), list(range(35)))
            self.assertTrue(any("muy ralo" in str(w.message) for w in caught))
            self.assertGreater(sub_fb.number_of_edges() / sub_fb.number_of_nodes(), 2.0)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPersona1)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
