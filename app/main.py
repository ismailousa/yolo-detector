from fastapi import FastAPI
from app.api import yolo
from app.schemas.constants import HOST, PORT

app = FastAPI(title="Yolo detector API", openapi_url="/openapi.json")


@app.get("/", status_code=200)
async def root() -> dict:
    """
    Root GET
    """
    return {"Yolo detector": "running"}


app.include_router(yolo.router, prefix="/yolo", tags=["yolo detector"])

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="debug")
