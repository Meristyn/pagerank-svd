# PageRank vía SVD

Comparación numérica y experimental de cuatro métodos para calcular el vector de **PageRank** como solución del sistema lineal $M_\alpha \mathbf{x} = \frac{1-\alpha}{n}\mathbf{e}$ (con $M_\alpha = I - \alpha S$ y $\boldsymbol{\pi} = \mathbf{x}/\Vert{}\mathbf{x}\Vert{}_1$), analizando su comportamiento cuando $\alpha \to 1$.

Proyecto desarrollado para el curso **Matemática Aplicada a la Computación**.

## Métodos evaluados

1. **Método de la potencia (`solve_power` / `solve_power_implicit`):** Iteración directa con $S$ dispersa explícita y variante implícita usando $(A, \mathbf{d})$ para evitar almacenar las columnas $\frac{1}{n}\mathbf{e}$ de los nodos sin salida (*dangling nodes*).
2. **Pseudoinversa exacta (`solve_pinv`):** Solución directa mediante `numpy.linalg.pinv`.
3. **Pseudoinversa truncada – SVD densa (`solve_svd_trunc`):** Factorización completa con `numpy.linalg.svd`, comparando la truncación clásica con los $k$ mayores valores singulares (`LM`) frente a los $k$ menores (`SM`).
4. **SVD truncada dispersa (`solve_svds`):** Aproximación iterativa con `scipy.sparse.linalg.svds` (ARPACK) en modos `LM` y `SM`.

## Hallazgos principales

- **Cota espectral:** Se demuestra y verifica empíricamente que $\sigma_{\min}(M_\alpha) \le \frac{1-\alpha}{\sqrt{n}\Vert{}\boldsymbol{\pi}\Vert{}_2} \le 1-\alpha$, cumpliéndose casi con igualdad en grafos Erdős–Rényi, Barabási–Albert y un subgrafo de `web-Stanford`.
- **Truncación `LM` vs. `SM`:** Conservar los $k$ mayores valores singulares (`LM`) pierde el ranking hasta $k = n$ (error $L_1 \approx 1.4\text{--}1.85$ con $k=20$). En cambio, conservar solo el menor valor singular (`SM`, $k=1$) reproduce el ranking con un error que decae empíricamente como $(1-\alpha)^{1.94}$.
- **Rendimiento y memoria:** El método de la potencia ofrece el mejor compromiso en tiempo y precisión. Con $n = 5000$ y $\alpha = 0.99$, la variante implícita reduce el uso de memoria entre **21× (ER)** y **39× (BA)** respecto a $S$ explícita. En el subgrafo real de SNAP, la potencia requiere muchas más iteraciones (61% de su cota de peor caso en $\alpha=0.999$), pero sigue siendo más rápida y precisa que `svds` (`SM`).

## Estructura del repositorio

```text
pagerank-svd/
├── pagerank_lib.py               # Librería compartida: grafos, matrices (A, S, M) y solvers
├── test.py                       # Suite de pruebas unitarias y validación matemática
├── p1_rendimiento.py             # Experimentos de rendimiento (f) y robustez/sensibilidad (g)
├── p1_diagnostico_espectral.py   # Verificación de la cota de sigma_min y alineación de u_n, v_n
├── p1_ley_sm.py                  # Ajuste empírico de la ley de decaimiento SM (k=1)
├── p2_common.py                  # Utilidades comunes: métricas de ranking (tau, precision@m, Gini)
├── espectro.py                   # Análisis del espectro sigma_i, kappa_2(M_alpha) y rango efectivo
├── error.py                      # Curvas de error vs. k y tabla de k_min (tau > 0.95)
├── nodos.py                      # Análisis por categoría de nodo (hubs, puentes, sin salida, bajo grado)
├── informe.tex                   # Informe técnico completo en LaTeX
├── requirements.txt              # Dependencias principales
└── resultados/                   # Tablas CSV y gráficas generadas por los experimentos
    └── p2/                       # Resultados de espectro, error vs. k y análisis por nodo
