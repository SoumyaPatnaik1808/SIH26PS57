# AetherSound AI Dashboard

## Run locally

Start the FastAPI service from the repository's `backend/` directory:

```bash
python -m uvicorn main:app --reload
```

Then start the dashboard from this directory:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The dashboard sends image uploads to `http://localhost:8000/api/v1/detect` by default.

To use another backend URL, set `NEXT_PUBLIC_API_URL` before starting Next.js, for example:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

The image is sent as multipart field `file`. Survey values (`altitude`, `towfish_lat`, and `towfish_lon`) are sent as query parameters, matching the FastAPI endpoint signature.
