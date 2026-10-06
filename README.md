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
uv run python scripts/lab09_hybrid.py --factors 20 --epochs 12 --alpha 0.75
uv run streamlit run app/streamlit_app.py
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

