
from pydantic import BaseModel

class CourseRequest(BaseModel):
    job_title: str
