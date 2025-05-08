from pydantic import BaseModel
from typing import List

class SkillRequest(BaseModel):
    job_title: str
    cv_skills: List[str]
