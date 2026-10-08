# System Card · Recomendador

## Usuarios y propósito
Sistema académico (INF-8239, LAB08–LAB09) que sugiere 10 películas por usuario a partir de MovieLens Latest Small. Su propósito es **comparar estrategias de recomendación y sus límites**, no tomar decisiones sobre personas ni operar en producción. Los usuarios previstos son estudiantes y docentes; la demo (`app/streamlit_app.py`) muestra solo el recomendador por contenido.

## Catálogo y candidatos
- Catálogo de LAB09: 9,701 películas con al menos un rating en entrenamiento; es el denominador de la cobertura. Las 18 películas sin ningún rating solo pueden aparecer en el recomendador de contenido de LAB08.
- Se excluyen siempre los ítems que el usuario ya valoró.
- Popularidad y contenido ordenan todo el catálogo no visto; el híbrido toma los **100 mejores candidatos colaborativos** y los reordena.

## Señales utilizadas
| Señal | Origen | Uso |
|---|---|---|
| Ratings explícitos (0.5–5) | `ratings.csv` | Factorización, popularidad y perfil de usuario |
| Orden temporal | `timestamp` | Corte leave-one-out (último rating de cada usuario a prueba) |
| Géneros | `movies.csv`, TF-IDF | Similitud de contenido y perfil ponderado por (rating − 2.5) |

No se usan tags, demografía, texto libre ni datos de navegación.

## Modelos y fallback
| Modelo | Descripción |
|---|---|
| Popularidad | Ranking por número de ratings y media (LAB09) o promedio bayesiano (LAB08). No personalizado. |
| Contenido | Coseno entre el perfil de géneros del usuario y cada película; empates resueltos por popularidad. |
| Colaborativo | Factorización matricial por SGD (sin sesgos), 10/20/40 factores, 12 épocas, lr 0.01, reg 0.05. |
| Híbrido | `α · colaborativo_normalizado + (1 − α) · contenido_normalizado` sobre 100 candidatos. |

**Configuración seleccionada:** híbrido con 40 factores y α = 0.5 (ver justificación en `README.md`, sección LAB09).

**Fallback:** un usuario sin factores latentes (nuevo o fuera del entrenamiento) recibe la lista de popularidad, que se declara **no personalizada** (`reports/cold_start_fallback.csv`). Se recomienda extender el fallback a usuarios con historial pequeño (< ~50 ratings), cuyas listas resultaron inestables.

## Métricas offline
Corte temporal leave-one-out, 587 usuarios evaluables, media de 3 semillas (42, 7, 123). Fuente: `reports/lab09_summary.csv`.

| Estrategia | RMSE | HitRate@10 | Cobertura | Tiempo | Tamaño |
|---|---|---|---|---|---|
| Media global (base) | 1.116 | — | — | 0 s | — |
| Media por película (base) | 1.050 | — | — | 0 s | — |
| Popularidad | — | 4.26 % | 1.2 % | 0 s | 76 KB |
| Contenido | — | 0.51 % | 8.3 % | 0 s | 310 KB |
| Colaborativo f=20 | 1.026 | 3.52 % | 7.0 % | 49 s | 1.6 MB |
| **Híbrido f=40 α=0.5** | 1.021 | 4.03 % | 10.5 % | 68 s | 3.4 MB |

Precision@10 = HitRate@10 / 10 porque solo hay un ítem relevante por usuario. Con 587 usuarios cada acierto vale 0.17 pp y la desviación entre semillas es 0.5–0.9 pp: diferencias menores que eso no son concluyentes.

## Cold start, cobertura y diversidad
- **Usuario nuevo:** recibe popularidad; lista estable e idéntica para todos.
- **Historial pequeño (19 ratings):** listas sin ítems vistos pero sin coincidencias entre semillas (Jaccard 0.00): la personalización no es confiable.
- **Historial amplio (2,697 ratings):** Jaccard 0.62 entre semillas (f=20).
- **Ítems fríos:** las 18 películas sin ratings no tienen factores ni popularidad; en LAB09 no aparecen en ninguna estrategia y solo el contenido de LAB08 puede sugerirlas.
- **Cobertura:** 1.2 % (popularidad) a 12.3 % (híbrido α=0.25); bajar α amplía el catálogo y reduce aciertos.

## Riesgos y monitoreo
| Riesgo | Evidencia | Mitigación / monitoreo |
|---|---|---|
| Popularidad dominante | La popularidad es la más precisa; los modelos colaborativos recomiendan películas con 110–140 ratings de media | Reportar `mean_item_popularity` y cobertura junto a HitRate |
| Burbuja de filtro | El contenido solo usa 20 géneros y produce empates masivos (362 empates para *Sabrina*) | α ≤ 0.5, más atributos (tags, año), métricas de diversidad intra-lista |
| Sesgo de selección | Solo se observan ratings de lo consumido | No interpretar ausencia de rating como rechazo; evaluar con datos de exposición si existieran |
| Ausencia de demografía | El dataset no tiene atributos de grupo | No afirmar equidad ni desempeño por grupo poblacional |
| Retroalimentación | Lo recomendado influye en los próximos ratings | Reservar exploración, reentrenar con cortes temporales y vigilar la caída de cobertura en el tiempo |
| Inestabilidad con poco historial | Jaccard 0 entre semillas | Fallback a popularidad/contenido bajo un umbral de historial |

**Uso no previsto:** decisiones comerciales, perfiles de personas reales o afirmaciones sobre gustos de poblaciones.
