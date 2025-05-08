
import pandas as pd
def annotate_skills(text: str, skills: str):
    """
    Annotate a job description with skill entities.

    Args:
        text (str): The job description.
        skills (str): A comma-separated list of skills.

    Returns:
        dict: A dictionary with the original text and a list of entity spans.
    """
    entities = []

    if pd.isna(text) or pd.isna(skills):
        return {'text': text, 'entities': []}

    skill_list = [s.strip() for s in skills.split(',') if s.strip()]
    text_lower = text.lower()

    for skill in skill_list:
        skill_lower = skill.lower()
        start = text_lower.find(skill_lower)

        # Find all instances, not just the first
        while start != -1:
            end = start + len(skill)
            entities.append({'start': start, 'end': end, 'label': 'SKILL'})
            start = text_lower.find(skill_lower, end)  # Look for next match

    return {'text': text, 'entities': entities}
