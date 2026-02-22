# Deploy Bet_prediction on Railway

## 1. Create a project on Railway

- Go to [railway.com/new](https://railway.com/new) and create a new project.
- Choose **Deploy from GitHub repo** and select `litaionut/bet_prediction`.

## 2. Add PostgreSQL

- In the project, click **+ New** → **Database** → **PostgreSQL**.
- Railway will create a Postgres service and expose variables like `PGHOST`, `PGUSER`, `PGPASSWORD`, `PGPORT`, `PGDATABASE`.

## 3. Configure the app service

- Open your **app service** (the one linked to GitHub).
- Go to **Variables** and add:

| Variable      | Value                    |
|---------------|--------------------------|
| `SECRET_KEY`  | A long random string     |
| `DEBUG`       | `False`                  |
| `PGHOST`      | `${{Postgres.PGHOST}}`   |
| `PGPORT`      | `${{Postgres.PGPORT}}`   |
| `PGUSER`      | `${{Postgres.PGUSER}}`   |
| `PGPASSWORD`  | `${{Postgres.PGPASSWORD}}` |
| `PGDATABASE`  | `${{Postgres.PGDATABASE}}` |

Alternatively, if Railway provides `DATABASE_URL` for the Postgres service, you can set only:

- `SECRET_KEY`
- `DEBUG=False`
- `DATABASE_URL` = `${{Postgres.DATABASE_URL}}` (or the variable name shown in the Postgres service).

## 4. Deploy

- Railway will use the **Procfile**:
  - **release**: `python manage.py migrate --noinput`
  - **web**: `gunicorn config.wsgi --log-file -`
- After deploy, open **Settings** → **Networking** → **Generate Domain** to get a public URL.

## 5. Optional: custom domain

- In **Networking**, add a custom domain and include that host in `ALLOWED_HOSTS` (or set it via an env var if you extend the app).

## 6. ML Over/Under 2.5 predictions on Railway

ML predictions are only shown if the **trained model file** exists in the app (e.g. `gemini_poisson_218.json`). These files are not in `.gitignore` by default so you can commit them.

- **To show ML on Railway:** train locally (e.g. `python manage.py build_gemini_dataset -c 218 -o gemini_dataset_218.csv` then `python manage.py train_gemini_poisson -d gemini_dataset_218.csv -o gemini_poisson_218.json`), then **add and commit** the `.json` file(s) and push. After deploy, the model will be on Railway and predictions will appear.
- **Alternative:** run the same training inside Railway after deploy (`railway ssh` then `python manage.py train_gemini_poisson ...`). The model will work until the next redeploy (container filesystem is ephemeral), so for a permanent fix, commit the model to the repo.
