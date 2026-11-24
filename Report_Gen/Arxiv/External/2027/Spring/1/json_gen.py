import json
import os
import shutil


def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)


data_to_save = \
    {
        # -----------------------------------------------------------------------------------------------------------------------
        "Version":
            """1""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2027""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """e, media & entertainment, each with sign-up steps, a runnable Python example, and sample output.""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """
            262 free APIs across news, social media, weather & environment, finance, census, housing, political science, law, health, agriculture & food, education, jobs & labor, science & space, AI & machine learning, engineering & energy, transportation & travel, sports & gaming, and culture, media & entertainment, each with sign-up steps, a runnable Python example, and sample output.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            262 free APIs across news, social media, weather & environment, finance, census, housing, political science, law, health, agriculture & food, education, jobs & labor, science & space, AI & machine learning, engineering & energy, transportation & travel, sports & gaming, and culture, media & entertainment, each with sign-up steps, a runnable Python example, and sample output.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            262 free APIs across news, social media, weather & environment, finance, census, housing, political science, law, health, agriculture & food, education, jobs & labor, science & space, AI & machine learning, engineering & energy, transportation & travel, sports & gaming, and culture, media & entertainment, each with sign-up steps, a runnable Python example, and sample output.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            262 free APIs across news, social media, weather & environment, finance, census, housing, political science, law, health, agriculture & food, education, jobs & labor, science & space, AI & machine learning, engineering & energy, transportation & travel, sports & gaming, and culture, media & entertainment, each with sign-up steps, a runnable Python example, and sample output.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            262 free APIs across news, social media, weather & environment, finance, census, housing, political science, law, health, agriculture & food, education, jobs & labor, science & space, AI & machine learning, engineering & energy, transportation & travel, sports & gaming, and culture, media & entertainment, each with sign-up steps, a runnable Python example, and sample output.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            2
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
            262 free APIs across news, social media, weather & environment, finance, census, housing, political science, law, health, agriculture & food, education, jobs & labor, science & space, AI & machine learning, engineering & energy, transportation & travel, sports & gaming, and culture, media & entertainment, each with sign-up steps, a runnable Python example, and sample output.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            262 free APIs across news, social media, weather & environment, finance, census, housing, political science, law, health, agriculture & food, education, jobs & labor, science & space, AI & machine learning, engineering & energy, transportation & travel, sports & gaming, and culture, media & entertainment, each with sign-up steps, a runnable Python example, and sample output.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Additional Resources":
            """
            262 free APIs across news, social media, weather & environment, finance, census, housing, political science, law, health, agriculture & food, education, jobs & labor, science & space, AI & machine learning, engineering & energy, transportation & travel, sports & gaming, and culture, media & entertainment, each with sign-up steps, a runnable Python example, and sample output.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Amir Jafari",
        "Proposed by email": "ajafari@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gwu.edu",
        "collaborator": "George Washington Univeristy",
        "funding_opportunity": "",
        "github_repo": "",
        # -----------------------------------------------------------------------------------------------------------------------
    }


os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}External{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}External{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy(__file__, output_file_path)
print(f"Data saved to {output_file_path}")
