 Bridging the Skill Gap Between University and Industry

This project uses deep learning to help students align their university-acquired skills with industry-required skills by analyzing CVs, job descriptions, and university curricula.

Project Overview

The goal of this system is to provide students with personalized feedback on what industry-relevant skills they possess and what skills they still need to acquire—based on their CV and a target job description. The system also recommends resources (courses, tutorials) and highlights relevant university courses.



 Core Features

1. Skill Extraction Model
- BERT-based NER model trained on annotated CV and job description data.
- Extracts job titles and technical skills from uploaded CVs and job descriptions.
- Identifies missing skills based on user input and job requirements.

 2.  Course Recommendation System
- Recommends relevant Coursera and Udemy courses (from static datasets).
- Matches missing skills with appropriate learning resources.
- Works without relying on live API access.

 3.  Curriculum Grouping System
- Analyzes BSc Computer Science and Management Information Systems curricula (Covenant University).
- Groups university courses by relevant job titles.
- Students can search for a job and see related university courses.

 4. Job-Skill Relevance Mapping
- Matches extracted skills to university-taught topics.
- Highlights gaps between education and job market needs.

5.  Upcoming Features
- Trending Skill Detection: Mine job postings for emerging skill trends.
- Personalized Learning Paths: Recommend dynamic learning plans based on current and desired skills.



Technologies Used

- Backend: FastAPI, Pydantic
- NLP: BERT-based NER (Hugging Face Transformers)

- Data Processing: Pandas, Excel
- IDE: Visual Studio Code

 


 How It Helps

- Helps students prepare for the job market before graduation.
- Empowers universities to see which parts of their curriculum map to in-demand jobs.
- Promotes self-directed learning based on real industry expectations.



Future Work

- Add dashboards for job trends and curriculum coverage.
- Integrate LinkedIn job data for richer skill tracking.
- Enable real-time course updates via APIs once stable network access is available.


 License

This project is under the MIT License. Feel free to use, modify, and share!


Want to Contribute?

Open an issue or submit a pull request. Contributions are welcome!




