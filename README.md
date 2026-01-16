# polynomialsolverad

## Database configuration

By default the app uses a local SQLite file named `polynomialsolver.db` in the
current working directory. To point the app at a different SQLite file, set the
`POLY_DB_PATH` environment variable before running the app.

Example:

```bash
export POLY_DB_PATH=/path/to/shared/polynomialsolver.db
streamlit run app.py
```
