# Create Django superuser on Railway

`railway run` runs the command **on your PC** with Railway env vars. The DB host `postgres.railway.internal` only works **inside Railway**, so you must run createsuperuser **inside** the app (via SSH).

## 1. Set variables in Railway (one time)

In **Railway Dashboard** → your **app service** → **Variables**, add:

| Variable | Value |
|----------|--------|
| `DJANGO_SUPERUSER_USERNAME` | e.g. `admin` |
| `DJANGO_SUPERUSER_EMAIL` | e.g. `admin@example.com` |
| `DJANGO_SUPERUSER_PASSWORD` | your secure password (min 8 chars) |

Save. Wait for redeploy if needed.

## 2. Link project (if not already linked)

```powershell
cd C:\Users\ili\GitHub\Bet_prediction
railway link
```

Choose **project** → **environment** → **app service** (not Postgres).

## 3. Run migrations and createsuperuser inside Railway (SSH)

Connect to the app container:

```powershell
railway ssh
```

In the **remote** shell, run migrations first (creates `auth_user` and other tables), then createsuperuser:

```bash
python manage.py migrate --noinput
python manage.py createsuperuser --noinput
```

You should see: `Superuser created successfully.` Then type `exit` to leave the SSH session.

If you see `relation "auth_user" does not exist`, it means migrations were not applied—run `migrate` first.

## 4. (Optional) Remove the 3 variables from Railway after creating the user

Log in at: `https://<your-app>.railway.app/admin/`

---

**Why not `railway run`?**  
`railway run python manage.py ...` runs **locally** with Railway env vars. The database host `postgres.railway.internal` is only reachable from inside Railway’s network, so the connection fails from your machine. Running the command inside `railway ssh` fixes this.
