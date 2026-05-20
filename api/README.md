# ChestXray FastAPI Backend

RESTful API service for chest X-ray multi-label classification.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your DATABASE_URL, JWT_SECRET_KEY, MODEL_PATH

# Run (development)
uvicorn src.main:app --reload --port 8000

# Run (production)
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Docker

```bash
docker build -t chestxray-api .
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/chestxray \
  -e JWT_SECRET_KEY=your-secret \
  -e MODEL_PATH=models/model.onnx \
  -v ./models:/app/models \
  chestxray-api
```

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/auth/register` | No | Register new user |
| `POST` | `/auth/login` | No | Login, get JWT token |
| `GET`  | `/auth/me` | Yes | Current user info |
| `POST` | `/api/predict` | Opt | Upload image, get classification |
| `GET`  | `/api/patients` | No | List patients (paginated) |
| `GET`  | `/api/patients/{id}` | No | Get patient by ID |
| `GET`  | `/api/patients/{id}/images` | No | Get patient's images |
| `GET`  | `/api/predictions/{id}` | No | Get prediction details |
| `GET`  | `/api/model/info` | No | Model version & metrics |
| `GET`  | `/health` | No | Health check |

## Configuration

| Env Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |
| `JWT_SECRET_KEY` | `change-me-...` | Secret for JWT signing |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Token expiry |
| `MODEL_PATH` | `models/model.onnx` | Path to ONNX/TorchScript model |
| `MODEL_DEVICE` | `cpu` | `cpu` or `cuda` |
| `RATE_LIMIT` | `100/minute` | Per-IP rate limit |
| `CONFIDENCE_THRESHOLD` | `0.5` | Min confidence to include in results |

## Testing

```bash
pytest tests/ -v
```

## OpenAPI Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
