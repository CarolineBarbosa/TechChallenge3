from fastapi import FastAPI
import controller

app = FastAPI()

app.include_router(controller.router)