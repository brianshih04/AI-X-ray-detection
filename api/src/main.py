"""FastAPI application factory."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.middleware.rate_limit import limiter, rate_limit_exceeded_handler
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown events."""
    # Startup: load ML model
    logger.info("Starting %s v%s ...", settings.APP_NAME, settings.APP_VERSION)
    from src.services.inference import inference_service
    try:
        inference_service.load_model()
        if inference_service.is_loaded:
            logger.info("ML model loaded successfully (type=%s)", inference_service.model_type)
        else:
            logger.warning("ML model not loaded — running in stub mode")
    except Exception as e:
        logger.error("Failed to load ML model: %s — running in stub mode", e)

    yield

    # Shutdown
    logger.info("Shutting down %s", settings.APP_NAME)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Chest X-ray Multi-label Classification API",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.DEBUG else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

    # Register routers
    from src.api.auth import router as auth_router
    from src.api.core import router as core_router

    app.include_router(auth_router)
    app.include_router(core_router)

    # Health check
    @app.get("/health", tags=["Health"])
    async def health_check():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()
