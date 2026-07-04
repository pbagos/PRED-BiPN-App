# Deployment Guide

How to deploy PRED-BiPN to a server for online access.

## Option 1: Render.com (free tier available)

1. Push your code to a GitHub repository
2. Go to https://render.com, then New, then Web Service
3. Connect your GitHub repo
4. Configure:
   - Environment: Python 3
   - Build Command: `pip install -r backend/requirements.txt`
   - Start Command: `cd backend && gunicorn -w 1 -b 0.0.0.0:$PORT app:app`
5. Click Create Web Service

The app will be live at `https://your-app-name.onrender.com`

## Option 2: Heroku

```bash
heroku create your-app-name
git push heroku main
```

The app will be live at `https://your-app-name.herokuapp.com`

## Option 3: Any Linux VPS (AWS, DigitalOcean, Linode, etc.)

```bash
git clone <your-repo> PRED-BiPN-App
cd PRED-BiPN-App
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

Set up Nginx as a reverse proxy for production use.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5050` | Port the server listens on |
| `FLASK_ENV` | `production` | Set to `development` to enable debug mode |
| `RENDER` | _(unset)_ | Set to any value when running on Render (adjusts tmp folder path) |

## Files Used by Deployment Platforms

- `Procfile`: start command for Heroku/Render
- `runtime.txt`: specifies Python 3.11.0
- `backend/requirements.txt`: all dependencies including `gunicorn`

## Testing Locally Before Deploying

```bash
cd PRED-BiPN-App
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend
python app.py
```

Open http://localhost:5050

## Notes

- The `/download` endpoint serves result files from the `tmp` folder
- On Render, `tmp` is set to `/tmp/uploads` (writable ephemeral storage)
- Result files are not persisted between server restarts on ephemeral platforms
