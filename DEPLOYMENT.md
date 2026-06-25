# Deployment

This app is prepared for a paid always-on Docker web service. Render is the
recommended simple path for this project because it supports Docker web
services and WebSockets, which Streamlit uses.

## Recommended: Render paid web service

1. Push this repository to GitHub.
2. In Render, create a new Blueprint from the repository.
3. Confirm the service uses `render.yaml`.
4. Keep the instance type on a paid plan such as `starter`.
5. Deploy.

The app health check is:

```text
/_stcore/health
```

The Docker image uses `requirements-deploy.txt`, which pins the runtime
packages used by the saved model artifacts. Keep `requirements.txt` for local
development if you want broader version ranges.

The container starts Streamlit with:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Local Docker test

Build the image:

```bash
docker build -t medical-appointment-analytics .
```

Run it locally:

```bash
docker run --rm -p 8501:8501 -e PORT=8501 medical-appointment-analytics
```

Then open:

```text
http://localhost:8501
```

## Notes

Free Render web services spin down after idle time. A paid instance avoids that
free-tier sleep behavior. The `.dockerignore` file excludes notebooks, virtual
environments, and unused training artifacts so deploys are smaller.
