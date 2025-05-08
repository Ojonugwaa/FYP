import pandas as pd

def identify_missing_skills(cv_skills, job_skills):
    cv_skills_set = set(cv_skills)  # Set of user skills
    job_skills_set = set(job_skills.split(', '))  # Set of job-required skills

    # If there's only one skill in cv_skills, it should still be treated as a set
    if len(cv_skills) == 1 and not cv_skills[0]:
        cv_skills_set = set()  # Handle empty case

    missing_skills = list(job_skills_set - cv_skills_set)
    return missing_skills


# test
if __name__ == "__main__":
    print("🔍 Test Missing Skill Finder from Job Title")

    # Load dataset
    df = pd.read_csv("data/IT jobs for training.csv")

    # Ask for job title
    job_title_input = input("Enter the job title: ").strip().lower()
    matched = df[df['job_title'].str.lower() == job_title_input]

    if matched.empty:
        print("Job title not found in dataset.")
    else:
        job_skills = matched.iloc[0]['skills']
       
        # Ask for user skills
        cv_skills_input = input("Enter your current skills (comma-separated): ")
        cv_skills = [skill.strip() for skill in cv_skills_input.strip().split(',')]

        missing = identify_missing_skills(cv_skills, job_skills)

        print("\nMissing Skills:")
        if missing:
            for skill in missing:
                print(f" - {skill}")
        else:
            print("You have all the required skills!")
