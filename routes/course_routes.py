from fastapi import APIRouter, Form
from services.course_matcher import load_and_match_courses

router = APIRouter()

@router.post("/analyze_curriculum")
async def analyze_curriculum(job_title: str = Form(...)):
    try:
        print(f"Received job title: {job_title}")
        # Process the job title and match with courses
        matches = load_and_match_courses(job_title.strip().lower())
        return {"job_title": job_title, "relevant_courses": matches}
    except Exception as e:
        return {"error": str(e)}
