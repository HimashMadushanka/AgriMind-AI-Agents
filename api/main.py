from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="AI Paddy Farming Intelligence API", version="0.1.0", description="Prototype; not a substitute for agricultural officers.")
app.include_router(router, prefix="/api/v1")
