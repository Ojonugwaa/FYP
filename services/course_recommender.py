import pandas as pd

# Load once at module level
coursera_df = pd.read_excel("data/coursera_course_dataset.xlsx")
udemy_df = pd.read_excel("data/udemy_courses.xlsx")


def fetch_offline_courses(df, skill, platform, limit=5):
    """Filter courses containing the skill keyword"""
    filtered = df[df['Title'].str.contains(skill, case=False, na=False)]
    results = filtered.head(limit)
    if platform == "coursera":
        return results[['Title', 'course_url']].to_dict(orient='records')
    elif platform == "udemy":
        return results[['Title', 'course_url']].to_dict(orient='records')
    return []

def recommend_courses(missing_skills):
    recommendations = {}
    for skill in missing_skills:
        recommendations[skill] = {
            "coursera": fetch_offline_courses(coursera_df, skill, "coursera", limit=5),
            "udemy": fetch_offline_courses(udemy_df, skill, "udemy", limit=5)
        }
    return recommendations


# ✅ TEST: Only run this if executing the script directly
if __name__ == "__main__":
    sample_skills = ["Python", "Machine Learning", "Data Analysis"]
    results = recommend_courses(sample_skills)
    from pprint import pprint
    pprint(results)
