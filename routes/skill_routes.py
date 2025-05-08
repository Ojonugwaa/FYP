


# import logging
# from fastapi import APIRouter, HTTPException, UploadFile, Form
# from fastapi.responses import JSONResponse
# from services.missing_skill_finder import identify_missing_skills
# from services.course_recommender import recommend_courses
# from services.cv_text_extractor import extract_text_from_cv
# from services.skill_predictor import extract_job_title_and_skills
# import re
# import pandas as pd

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# router = APIRouter()

# # Load job dataset
# try:
#     df = pd.read_csv('data/IT jobs for training.csv')
#     logger.info("Job dataset loaded successfully.")
# except Exception as e:
#     logger.error(f"Error loading job dataset: {str(e)}")
#     raise HTTPException(status_code=500, detail="Error loading job dataset")

# # Helper function for text preprocessing
# def clean_cv_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"[^a-z0-9\s]", "", text)
#     text = ' '.join(text.split())
#     return text

# @router.post("/analyze_skills")
# async def analyze_skills(
#     job_title: str = Form(""),
#     cv_skills: str = Form(""),
#     cv_file: UploadFile = None
# ):
#     logger.info("Received a skill analysis request.")

#     extracted_job_title = job_title.strip().lower()
#     extracted_skills = [skill.strip().lower() for skill in cv_skills.split(",") if skill.strip()] if cv_skills else []

#     if cv_file:
#         try:
#             file_content = await cv_file.read()
#             cv_text = extract_text_from_cv(file_content, cv_file.filename)
#             cv_text_clean = clean_cv_text(cv_text)
#             extracted_job_title_from_cv, predicted_skills = extract_job_title_and_skills(cv_text_clean)
#             logger.info(f"Extracted from CV - Job Title: {extracted_job_title_from_cv}, Skills: {predicted_skills}")

#             if extracted_job_title_from_cv:
#                 extracted_job_title = extracted_job_title_from_cv.lower()
#             if predicted_skills:
#                 extracted_skills = predicted_skills
#         except Exception as e:
#             logger.error(f"Error processing uploaded CV: {str(e)}")
#             raise HTTPException(status_code=500, detail=f"Error processing uploaded CV: {str(e)}")

#     if not extracted_job_title:
#         logger.error("Job title is missing even after processing.")
#         raise HTTPException(status_code=400, detail="Job title could not be determined from CV or form.")

#     # Find matching job title in dataset
#     matching_row = df[df['job_title'].str.lower() == extracted_job_title]

#     if matching_row.empty:
#         logger.warning(f"Job title '{extracted_job_title}' not found in dataset.")
#         raise HTTPException(status_code=404, detail=f"Job title '{extracted_job_title}' not found in dataset.")

#     job_skills = matching_row.iloc[0]['skills']
#     missing_skills = identify_missing_skills(extracted_skills, job_skills)
#     course_recommendations = recommend_courses(missing_skills)

#     logger.info(f"Analysis Completed: Missing skills: {missing_skills}")

#     return JSONResponse({
#         "job_title": extracted_job_title,
#         "job_skills": job_skills,
#         "cv_skills": extracted_skills,
#         "missing_skills": missing_skills,
#         "course_recommendations": course_recommendations
#     })


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("skill_routes:router", host="127.0.0.1", port=8000, reload=True)


import logging
from fastapi import APIRouter, HTTPException, UploadFile, Form
from fastapi.responses import JSONResponse
from services.missing_skill_finder import identify_missing_skills
from services.course_recommender import recommend_courses
from services.cv_text_extractor import extract_text_from_cv
from services.skill_predictor import extract_job_title_and_skills
import re
import pandas as pd
from fuzzywuzzy import process

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Load job dataset
try:
    df = pd.read_csv('data/IT jobs for training.csv')
    logger.info("Job dataset loaded successfully.")
except Exception as e:
    logger.error(f"Error loading job dataset: {str(e)}")
    raise HTTPException(status_code=500, detail="Error loading job dataset")

# Helper function for text preprocessing
def clean_cv_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s,]", "", text)
    text = ' '.join(text.split())
    return text

@router.post("/analyze_skills")
async def analyze_skills(
    job_title: str = Form(""),
    cv_skills: str = Form(""),
    cv_file: UploadFile = None
):
    logger.info("Received a skill analysis request.")

    extracted_job_title = job_title.strip().lower()
    extracted_skills = [skill.strip().lower() for skill in cv_skills.split(",") if skill.strip()] if cv_skills else []

    if cv_file:
        try:
            file_content = await cv_file.read()
            cv_text = extract_text_from_cv(file_content, cv_file.filename)
            cv_text_clean = clean_cv_text(cv_text)
            logger.info(f"Extracted CV text: {cv_text_clean[:500]}...")
            extracted_job_title_from_cv, predicted_skills = extract_job_title_and_skills(cv_text_clean)
            logger.info(f"Extracted from CV - Job Title: {extracted_job_title_from_cv}, Skills: {predicted_skills}")

            if extracted_job_title_from_cv and extracted_job_title_from_cv != "Not detected":
                extracted_job_title = extracted_job_title_from_cv.lower()
            if predicted_skills:
                extracted_skills = predicted_skills
        except Exception as e:
            logger.error(f"Error processing uploaded CV: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing uploaded CV: {str(e)}")

    logger.info(f"Final extracted job title: {extracted_job_title}")
    if not extracted_job_title or extracted_job_title == "not detected":
        # Try fuzzy matching
        job_titles = df['job_title'].str.lower().tolist()
        if extracted_job_title and extracted_job_title != "not detected":
            best_match, score = process.extractOne(extracted_job_title, job_titles)
            if score > 80:
                logger.info(f"Fuzzy matched '{extracted_job_title}' to '{best_match}'")
                extracted_job_title = best_match
            else:
                logger.error("Job title is missing or not detected.")
                raise HTTPException(status_code=400, detail="Job title could not be determined from CV or form.")
        else:
            logger.error("Job title is missing or not detected.")
            raise HTTPException(status_code=400, detail="Job title could not be determined from CV or form.")

    matching_row = df[df['job_title'].str.lower() == extracted_job_title]
    if matching_row.empty:
        logger.warning(f"Job title '{extracted_job_title}' not found in dataset.")
        raise HTTPException(status_code=404, detail=f"Job title '{extracted_job_title}' not found in dataset.")

    job_skills = matching_row.iloc[0]['skills']
    missing_skills = identify_missing_skills(extracted_skills, job_skills)
    course_recommendations = recommend_courses(missing_skills)

    logger.info(f"Analysis Completed: Missing skills: {missing_skills}")

    return JSONResponse({
        "job_title": extracted_job_title,
        "job_skills": job_skills,
        "cv_skills": extracted_skills,
        "missing_skills": missing_skills,
        "course_recommendations": course_recommendations
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("skill_routes:router", host="127.0.0.1", port=8000, reload=True)