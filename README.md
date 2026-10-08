# INF-8239 · Unidad 03 · Sistemas recomendadores

Autor académico: Edwin Ramón José Nolasco

Proyecto inicial de LAB08 y LAB09. El dataset de muestra prueba el código; la evidencia final usa MovieLens.

## Inicio
```bash
uv python install 3.12
uv sync
uv run pytest -q
uv run python scripts/download_data.py
uv run python scripts/audit_data.py
```

## Laboratorios
```bash
uv run python scripts/lab08_content.py
uv run python scripts/lab09_hybrid.py --factors 20 --epochs 12 --alpha 0.25
uv run python scripts/lab09_hybrid.py --factors 20 --epochs 12 --alpha 0.75
uv run python scripts/lab09_experiments.py   # 3 factores x 3 alphas x 3 semillas, Pareto y perfiles (~15 min)
uv run streamlit run app/streamlit_app.py
```

## Notebook del Ejercicio 05
`notebooks/ejercicio05.ipynb` se entrega ejecutado. Requiere los reportes de los comandos anteriores; para volver a ejecutarlo (~3 min):
```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/ejercicio05.ipynb
```

## Interpretación
No presente una recomendación como verdad. Documente datos, candidatos, puntuación, métricas, fallback y límites.

## Redacción del estudiante

Resultado principal:
La popularidad ponderada produce una lista estable y reconocible (Shawshank
Redemption, The Godfather, Fight Club), pero idéntica para todos los usuarios.
El recomendador basado en contenido personaliza por película de entrada
(p. ej., "Similares a Toy Story"), pero genera empates de similitud perfecta:
los diez similares a Toy Story tienen content_score = 1.0, por lo que su orden
es arbitrario y no refleja diferencias reales de calidad o afinidad.

Evidencia utilizada:
MovieLens Latest Small, auditado con scripts/audit_data.py: 100,836 ratings,
610 usuarios y 9,724 películas valoradas (catálogo de 9,742), con densidad de
1.70% y ratings entre 1996 y 2018 (media 3.50, desviación 1.04). Línea base:
popularidad ponderada (promedio bayesiano con la media global y cuantil
mínimo de ratings). Modelo de contenido: TF-IDF sobre géneros + similitud
coseno. Resultados en reports/popular_top10.csv, popular_mean_top10.csv,
content_recommendations.csv, content_ties.csv, cold_start_items.csv y
lab08_salida.txt.

Qué representa la similitud:
La similitud coseno mide cuánto se parecen dos películas únicamente por sus
géneros (vectores TF-IDF), no por su calidad, año, reparto ni por el
comportamiento de los usuarios. Un valor de 1.0 significa "mismo perfil de
géneros", no "igual de buena" ni "le gustará al mismo usuario".

Problema de cold start observado:
La matriz usuario-ítem es muy dispersa (densidad 1.70%). cold_start_items.csv
registra 18 películas sin ningún rating: la popularidad ponderada no puede
recomendarlas porque no hay comportamiento que aprender. El enfoque de
contenido sí puede incluirlas, ya que solo necesita sus géneros. Un usuario
nuevo, sin historial, solo recibiría la lista de popularidad como fallback.

Riesgo de sobre-especialización:
Al usar solo géneros, las recomendaciones se concentran en películas con
categorías idénticas y aparecen empates de score 1.0 (content_ties.csv
documenta 3 registros). El usuario recibe variantes de lo mismo, con baja
diversidad y poca serendipia.

Decisión o siguiente experimento:
Mantener la popularidad como fallback para usuarios nuevos y usar contenido
como complemento, no como sistema único. Siguiente experimento: añadir un
criterio de desempate (popularidad ponderada o año) y enriquecer los
atributos (tags de MovieLens, año) para reducir empates; luego comparar contra
la línea base con precision@k, cobertura y diversidad.

## Redacción LAB09

Evidencia: `reports/lab09_runs.csv` (27 corridas: factores 10/20/40 × semillas 42/7/123,
alpha 0.25/0.5/0.75 sobre el mismo modelo), `lab09_summary.csv`, `lab09_pareto.csv`,
`lab09_profiles.csv`, `lab09_profile_stability.csv`, `hybrid_metrics.json`,
`cold_start_fallback.csv` y `lab09_runs/` (salidas de los comandos del manual).
Corte temporal leave-one-out: se evalúan 587 de 610 usuarios (los 23 restantes tienen
como última valoración una película sin ratings en entrenamiento). Todas las cifras son
medias sobre tres semillas.

| Estrategia | HitRate@10 | Cobertura | Popularidad media de lo recomendado | Entrenamiento | Tamaño |
|---|---|---|---|---|---|
| Popularidad | 4.26 % | 1.2 % | 233 ratings | 0 s | 76 KB |
| Contenido (géneros) | 0.51 % | 8.3 % | 32 ratings | 0 s | 310 KB |
| Colaborativo f=20 | 3.52 % ± 0.55 | 7.0 % | 138 ratings | 49 s | 1.6 MB |
| Híbrido f=20 α=0.25 | 2.67 % ± 0.52 | 10.1 % | 96 ratings | 49 s | 1.9 MB |
| Híbrido f=20 α=0.75 | 3.52 % ± 0.55 | 7.2 % | 136 ratings | 49 s | 1.9 MB |
| Híbrido f=10 α=0.75 | 4.03 % ± 0.94 | 5.6 % | 141 ratings | 53 s | 1.1 MB |
| Híbrido f=40 α=0.5 | 4.03 % ± 0.52 | 10.5 % | 111 ratings | 68 s | 3.4 MB |

Resultado de ratings:
RMSE de la factorización: 1.032 (f=10), 1.026 (f=20) y 1.021 (f=40). Las líneas base
sin factores latentes obtienen 1.116 (media global) y 1.050 (media de cada película).
La factorización mejora solo un 2–3 % sobre la media por película y más factores
apenas reducen el error: aproxima un poco mejor los ratings, pero eso no se traduce en
mejor ranking (f=40 tiene el menor RMSE y no el mayor HitRate en todos los alpha).

Resultado de ranking:
Ninguna configuración personalizada supera a la popularidad en HitRate@10 (4.26 %).
El mejor híbrido llega a 4.03 %, dentro de una desviación entre semillas (0.5–0.9 pp,
cada acierto vale 0.17 pp con 587 usuarios). Precision@10 es HitRate/10 porque hay un
único ítem retenido por usuario. El contenido solo es casi inútil para acertar (0.51 %).
Con α=0.75 el híbrido f=20 coincide con el colaborativo: la señal de contenido apenas
reordena el Top-10.

Cobertura del catálogo:
La popularidad recomienda 1.2 % del catálogo a todos. Al bajar alpha sube la cobertura
y baja la popularidad de lo recomendado (α=0.25: 8.2–12.3 %; α=0.75: 5.6–9.7 %), a costa de
aciertos (α=0.25 pierde 0.85–1.3 pp de HitRate en los tres tamaños). Es el patrón "HitRate
sube y cobertura baja" del manual: alpha intercambia acierto por amplitud.

Comportamiento del usuario frío:
El usuario nuevo (userId 611, sin historial) recibe la lista de popularidad (Forrest
Gump, Shawshank, Pulp Fiction…), idéntica en todas las semillas y marcada como no
personalizada. El usuario de historial amplio (414, 2,697 ratings) obtiene listas sin
ítems vistos y moderadamente estables (Jaccard 0.62 entre semillas, f=20). El usuario de
historial pequeño (53, 19 ratings) recibe listas sin ítems vistos pero totalmente
inestables (Jaccard 0.00 entre semillas): con 19 ratings sus factores son casi ruido de
inicialización, y el contenido no lo rescata porque solo reordena los 100 candidatos
colaborativos (coincidencia con el Top-10 de contenido: 0 %).

Configuración seleccionada y evidencia:
alpha = 0.75 frente a 0.25: en los tres tamaños 0.75 da más aciertos (+0.85 a +1.3 pp,
mayor que la variación entre semillas) y 0.25 da más cobertura. Se selecciona el
**híbrido f=40, α=0.5**: iguala el mejor HitRate híbrido (4.03 %), duplica la cobertura
del f=10 α=0.75 (10.5 % frente a 5.6 %), recomienda ítems menos concentrados en lo
popular (111 frente a 141 ratings de media) y es el más estable entre semillas
(± 0.52 pp). Domina al f=40 α=0.75 en la tabla de Pareto. Alternativa Green AI: f=10
α=0.75 (un tercio del tamaño) si la cobertura no fuera prioritaria. Para usuarios con
menos de ~50 ratings se recomienda usar popularidad/contenido en vez de factores.

Costo computacional:
Entrenar cuesta 49–68 s y 1.1–3.4 MB; popularidad y contenido no requieren
entrenamiento. Pasar de 10 a 40 factores triplica el tamaño y suma ~15 s para bajar el
RMSE 0.011 y sin mejorar el mejor HitRate; la ganancia de f=40 es solo de cobertura.
El tiempo tiene ruido (f=10 tardó más que f=20 porque el bucle SGD en Python depende de
la carga de la máquina), por eso la frontera de Pareto con utilidad (HitRate, cobertura)
y costo (tiempo, tamaño) deja 13 de 14 configuraciones no dominadas: no hay un ganador
gratuito y la elección depende del peso que se dé a la cobertura.

Riesgo principal y mitigación:
Popularidad dominante y burbuja de filtro: el baseline más preciso es el que menos
diversidad ofrece y todos los modelos colaborativos concentran sus listas en películas
con 110–140 ratings. Otros riesgos: sesgo de selección (solo se observa lo consumido),
ausencia de demografía (no se puede evaluar equidad por grupo) y retroalimentación (lo
recomendado genera los ratings que reentrenan el modelo). Mitigación: monitorear
cobertura y popularidad media junto a HitRate, usar α≤0.5 para ampliar catálogo,
fallback explícito para historiales pequeños y declarar en la interfaz cuándo una lista
no es personalizada. Detalle en `docs/SYSTEM_CARD.md`.


## Declaración de uso de IA

### Herramientas
- **Claude** (asistente de Anthropic, modelo Claude Opus 5.5).
- Trabajo anterior al 08-10-2026 (commits `bce20e3` a `5888ff5`, LAB08 y primera versión
  de LAB09).

### Prompts relevantes (resumen)
1. Solicité un análisis general de la carpeta del proyecto y un reporte de su estado.
2. Pedí verificar si el proyecto cumplía los requisitos del manual U03.LAB09.
3. Indiqué aplicar las correcciones necesarias y validar de nuevo el resultado.
4. Rechacé la creación del módulo `hybrid.py` por no formar parte de la estructura
   del proyecto base y pedí integrar su lógica en los módulos existentes.
5. Solicité fusionar los tests del híbrido con `tests/test_matrix_factorization.py`.
6. Pedí revisar el proyecto contra el enunciado del Ejercicio 05.

### Qué generó la IA
- En `src/inf8239_u03/recommenders.py`: Top-N vectorizado, tamaño del modelo,
  `GenreSpace`, `HybridRecommender` (cuatro estrategias y fallback de usuario frío),
  `run_experiment`, `select_profiles` y `profile_recommendations`.
- En `metrics.py`: `precision_at_k` y `mean_item_popularity`.
- `scripts/lab09_hybrid.py` reescrito (`--seed`, tamaño, métricas por configuración,
  perfiles) y `scripts/lab09_experiments.py` (27 corridas, Pareto y estabilidad).
- 10 tests nuevos, `notebooks/ejercicio05.ipynb`, `docs/DATASET_CARD.md`,
  `docs/SYSTEM_CARD.md` y el borrador de la redacción de LAB09 en este README.

### Verificaciones realizadas
- `uv run pytest -q`: 17 tests pasan. `uv run ruff check .`: sin errores.
- La refactorización reproduce exactamente las métricas del script original
  (RMSE 1.0264, HitRate@10 3.75 %, cobertura 7.75 % con f=20, α=0.75, semilla 42).
- El notebook recalcula una corrida y coincide con `reports/lab09_runs.csv`
  en las 5 estrategias.
- Se ejecutaron todos los comandos de las secciones 2, 3, 5 y 9 del manual.
- Los hashes SHA-256 de la Dataset Card se calcularon sobre los CSV locales.
- Las cifras del README y de la System Card se contrastaron con los reportes; el tamaño
  del catálogo (9,701 ítems) se recalculó a partir de los datos.

### Correcciones
| Problema | Cómo se detectó | Corrección |
|---|---|---|
| `count = 0` para películas con ratings en `content_recommendations.csv` (p. ej., movieId 3754 tiene 9) | Revisión de la IA, verificada contra `ratings.csv` | El merge usa estadísticas de todas las películas (`min_count=0`) |
| `lab08_salida.txt` desactualizado y en UTF-16 | Revisión de la IA | Regenerado en UTF-8 |
| Pareto con HitRate como única utilidad: la popularidad dominaba todo pese a cubrir 1.2 % del catálogo | Revisión de resultados | Utilidad = (HitRate@10, cobertura) |
| La IA creó un módulo `hybrid.py` que no existía en el proyecto base | Rechazo del estudiante | Contenido movido a `recommenders.py`; tests fusionados |
| Un comando de la IA borró un import en los tests | `ruff` y `pytest` fallaron | Import restaurado |
| Cifras redondeadas de más en el README (+0.9 pp en lugar de +0.85 pp) y afirmaciones imprecisas en la System Card | Revisión contra los CSV | Cifras corregidas |

### Responsabilidad
La configuración seleccionada (f=40, α=0.5),
las conclusiones y la interpretación fueron revisadas y asumidas por mí. Puedo explicar
el corte temporal, la factorización por SGD, la mezcla híbrida y el criterio de Pareto.
La única referencia bibliográfica es Harper & Konstan (2015), DOI 10.1145/2827872,
incluida en el proyecto base.
