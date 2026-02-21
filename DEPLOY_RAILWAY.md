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
