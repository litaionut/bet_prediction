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

## 6. Modele ML (Over/Under 2.5) pe Railway – pași clari

Aplicația salvează și citește fișierele de model (`gemini_poisson_*.json`) dintr-un folder configurat. Pe Railway poți folosi un **Volume** ca acest folder să nu se șteargă la fiecare redeploy și ca modelele să poată fi refăcute/actualizate.

---

### Pași detaliați (Volume – modele persistente)

**Ce facem:** Creăm un „disc” (Volume) atașat aplicației și îi spunem aplicației să salveze modelele acolo. La redeploy, discul rămâne, deci și modelele.

| Pas | Unde | Ce faci |
|-----|------|--------|
| **1** | Dashboard Railway → proiectul tău | Click pe **+ New** (sau **Add Service**). |
| **2** | Meniul care apare | Alege **Volume** (nu Database, nu GitHub). Se creează un serviciu nou de tip Volume. |
| **3** | Click pe **Volume-ul** creat | În dreapta vezi setările. La **Mount Path** scrie: `/data` (sau lasă ce propune Railway). Notează acest path. |
| **4** | Click pe **serviciul tău de aplicație** (cel cu deploy din GitHub, nu Volume, nu Postgres) | E cel care rulează site-ul. |
| **5** | În serviciul aplicației: tab **Settings** (sau **Variables**) | Caută secțiunea **Volumes** / **Volume Mounts**. |
| **6** | Secțiunea Volumes | Click **Add Volume** / **Mount**. Alege Volume-ul creat la pasul 2. La **Mount Path** pune același path: `/data`. Salvează. |
| **7** | Tot în serviciul aplicației: tab **Variables** | Adaugă o variabilă nouă: **Name** = `ML_MODELS_DIR`, **Value** = `/data`. (Exact path-ul de la pasul 3/6.) Salvează. |
| **8** | Redeploy | Dacă e nevoie, dă **Redeploy** la aplicație. După deploy, aplicația citește și scrie modelele în `/data`. |
| **9** | Creare/actualizare modele | Din interfața web: mergi la o ligă → **Build dataset** / **Train model**. Sau din Railway: **Settings** → **Deploy** → **Run Command** (dacă există) sau folosești CLI-ul Railway local: `railway run python manage.py train_gemini_poisson -d gemini_dataset_218.csv -o gemini_poisson_218.json`. Fișierele `.json` se salvează pe Volume și **nu se pierd** la redeploy. |

**Rezumat:** Volume = un folder persistent (`/data`). `ML_MODELS_DIR=/data` = aplicația folosește acel folder pentru modele. La antrenament, fișierele merg în `/data` și rămân după redeploy.

---

### Variantă fără Volume (modele din repo)

- Antrenezi **local** (pe PC), apoi adaugi în Git fișierele `gemini_poisson_*.json`, faci commit și push. Pe Railway aplicația le citește din cod. Dezavantaj: dacă antrenezi din nou pe Railway, noul model nu se salvează permanent decât dacă folosești Volume (pașii de mai sus).

## 7. Update game results every hour (today + yesterday)

The app can refresh scores and status from API-Football. A management command syncs **today and yesterday** by default (so yesterday's final scores are updated too). Schedule it to run every hour.

### Command

```bash
python manage.py update_today_results
```

This runs **two** API requests (one per date) and updates or creates `Game` records. Optional:

- Only today: `python manage.py update_today_results --today-only`
- Single date: `python manage.py update_today_results --date 2024-02-23`

### Run every 1 hour on Railway (Cron)

1. In your Railway project, click **+ New** → **Cron Job** (or **Add Service** and choose Cron if available).
2. Set the **schedule** to every hour, e.g. `0 * * * *` (at minute 0 of every hour).
3. Set the **command** to run in your app’s environment. If the cron runs in the same project and has access to your app service:
   - Command: `python manage.py update_today_results`
   - Ensure the cron service uses the same **Variables** as your app (e.g. `DATABASE_URL`, `API_FOOTBALL_KEY`), or link it to the app service so it inherits them.
4. Save; Railway will run the command on the schedule.

**Alternative: HTTP endpoint (no Railway Cron needed)**  
You can trigger the update from an external scheduler (e.g. [cron-job.org](https://cron-job.org)) every hour:

1. In your app **Variables**, add `CRON_SECRET` with a long random string (e.g. a UUID).
2. In the scheduler, set a request every hour to:
   `https://YOUR_APP.railway.app/football/cron/update-today-results/?secret=YOUR_CRON_SECRET`
   (or send the secret in header: `X-Cron-Secret: YOUR_CRON_SECRET`).
3. A successful run returns `{"ok": true, "message": "..."}`; otherwise 403 (wrong secret) or 500 (API error).
