from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import skill_routes, course_routes  

app = FastAPI()

# ✅ Allow frontend access (adjust origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(skill_routes.router)
app.include_router(course_routes.router)  # if you have curriculum grouping

@app.get("/")
def root():
    return {"message": "Welcome to the Smart Career Recommender API"}
