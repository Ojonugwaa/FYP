
# from docx import Document
# import re

# def extract_courses(doc_path):
#     document = Document(doc_path)
#     courses = []
#     current_course = None

#     for para in document.paragraphs:
#         line = para.text.strip()
#         if not line:
#             continue

#         # Match course code (e.g., CSC 301, MKT 201)
#         if re.match(r"^[A-Z]{3}\s?\d{3}", line):
#             if current_course:
#                 courses.append(current_course)
#             current_course = {"title": line, "description": ""}
#         elif current_course:
#             current_course["description"] += " " + line

#     if current_course:
#         courses.append(current_course)

#     return courses
