import pandas as pd

import openpyxl
#from utils.docx_parser import extract_courses

# Paths to the course data (using Excel instead of DOCX files)
COURSE_FILES = [
    "data/Computer_Science_Course.xlsx",  # Path to BSc Computer Science Excel file
    "data/mis_course.xlsx"        # Path to BSc Management Information Systems Excel file
]

# Job keywords for matching (keep this as it is)
JOB_KEYWORDS = {
    "data scientist": ["machine learning", "statistics", "python", "data mining"],
    "software engineer": ["algorithms", "programming", "software", "data structures", "system analysis","internet programming", "compiler"], 
    "cybersecurity analyst": ["security", "encryption", "firewall", "cyber"],
    "ai engineer": ["deep learning", "neural networks", "machine learning", "ai"],
    "network engineer": ["network", "protocol", "routing", "switching"],
    "ui/ux designer": ["human-computer interaction", "application package"],
    "database administrator": ["database", "protocol", "routing", "switching"],
    "product manager": ["project management", "design thinking", "entrepreneurship"]

}





def extract_courses(doc_path):
    # Open the Excel file
    workbook = openpyxl.load_workbook(doc_path)
    sheet = workbook.active  # Assuming courses are in the active sheet

    courses = []

    # Loop through the rows in the sheet
    for row in sheet.iter_rows(min_row=2, values_only=True):  # Skip the header row
        # Assuming the format is: [Course Code, Course Name, Description]
        course_code, course_name, description = row

        if course_code and course_name:
            courses.append({
                "title": course_name,  # Course name as title
                "description": description if description else "No description available"
            })

    print(f"Extracted courses: {courses}")  # Debugging line
    return courses


def load_and_match_courses(job_title):
    print(f"Job Title in load_and_match_courses: {job_title}")
    all_courses = []
    for path in COURSE_FILES:
        all_courses.extend(extract_courses(path))
    
    print(f"All Courses: {all_courses}")

    relevant = []
    job_keywords = JOB_KEYWORDS.get(job_title.lower(), [])
    print(f"Job Title: {job_title}, Keywords: {job_keywords}")
    
      
    for course in all_courses:
        text = f"{course['title']} {course['description']}".lower()
        print(f"Checking course: {course['title']}")  # Debugging line
        if any(keyword in text for keyword in job_keywords):
            relevant.append(course)

    print(f"Relevant Courses: {relevant}")  # Debugging line
    return relevant

