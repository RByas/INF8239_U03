# Dataset Card · MovieLens

## Fuente, fecha, versión y hash
- **Fuente:** GroupLens Research, <https://files.grouplens.org/datasets/movielens/ml-latest-small.zip> (variable `MOVIELENS_URL`).
- **Versión:** MovieLens Latest Small (`ml-latest-small`), generada el 26-09-2018. Es un dataset de *desarrollo*: GroupLens puede actualizarlo, por eso se registra el hash.
- **Descarga local:** 06-10-2026 en `data/raw/ml-latest-small/` (no versionado en Git).
- **SHA-256 de los archivos extraídos:**

| Archivo | SHA-256 |
|---|---|
| `ratings.csv` | `aa289ca83157595d0df6aea1be6a4ded676ddc4385472e8313a8ed9805352646` |
| `movies.csv` | `5a5f32dd9bb3797b8e728a1b98958789d2b13f294a69fdfbc5727f8a9611aa07` |
| `tags.csv` | `92a9f8bb7916dceef6151209845788c3643f794dfa79d1feaec7121b5960399d` |
| `links.csv` | `97ad18e4e56a09363c65676b6cb3482ce3e2cea2372a24620c1599c843325f31` |

El hash del ZIP completo lo imprime `scripts/download_data.py` en cada descarga.

## Licencia y cita
Uso permitido para investigación y docencia; prohibido el uso comercial sin permiso de GroupLens; no se puede insinuar respaldo de la Universidad de Minnesota; la redistribución debe mantener la misma licencia (ver `README.txt` del dataset).

Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 19:1–19:19. DOI: 10.1145/2827872.

## Archivos y diccionario
| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| `ratings.csv` | 100,836 | `userId`, `movieId`, `rating`, `timestamp` | Valoración de 0.5 a 5.0 en pasos de 0.5; `timestamp` en segundos Unix (UTC). |
| `movies.csv` | 9,742 | `movieId`, `title`, `genres` | Título con año entre paréntesis; géneros separados por `|` (20 valores, incluye `(no genres listed)`). |
| `tags.csv` | 3,683 | `userId`, `movieId`, `tag`, `timestamp` | Etiquetas libres de usuarios. No se usan en LAB08/LAB09. |
| `links.csv` | 9,742 | `movieId`, `imdbId`, `tmdbId` | Identificadores externos. No se usan. |

## Procedimiento de descarga
```bash
uv run python scripts/download_data.py   # descarga, valida rutas del ZIP e imprime SHA-256
uv run python scripts/audit_data.py      # tamaños, rango temporal, densidad y distribución de ratings
```
`load_movielens` valida el contrato: columnas requeridas, ningún rating sin película y ratings dentro de 0.5–5.0.

## Población observada y exclusiones
- 610 usuarios seleccionados al azar por GroupLens, **todos con al menos 20 valoraciones**; periodo 29-03-1996 a 24-09-2018.
- 9,724 películas valoradas de 9,742; 18 no tienen ninguna valoración (`reports/cold_start_items.csv`).
- Densidad de la matriz usuario–ítem: 1.70 %. Media de rating 3.50, desviación 1.04.
- **Exclusiones en LAB09:** el corte temporal reserva la última valoración de cada usuario; se evalúan solo los 587 usuarios cuya película retenida aparece en entrenamiento (23 películas retenidas no tienen ningún rating previo y no tienen factores latentes).

## Calidad, sesgos y usos prohibidos
- **Sesgo de selección:** solo se observa lo que cada usuario eligió ver y valorar; la ausencia de rating no significa desinterés.
- **Sesgo de popularidad:** pocas películas concentran muchas valoraciones (cola larga), lo que favorece a los modelos que recomiendan lo popular.
- **Sin demografía:** no hay edad, género ni país; no pueden afirmarse resultados por grupo poblacional ni auditar equidad.
- **Población no representativa:** usuarios de un servicio académico estadounidense con 20 o más valoraciones; los usuarios casuales no están.
- **Géneros gruesos:** 20 categorías por película; muchas películas comparten exactamente el mismo vector, lo que produce empates de similitud.
- **Usos prohibidos:** uso comercial sin permiso, reidentificación de usuarios, presentar resultados como representativos de una población o como benchmark publicable (es un dataset de desarrollo).
