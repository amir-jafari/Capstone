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
            """Collaborative editing: Multiple users can add""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """
            • Collaborative editing: Multiple users can add or revise content and track revision histories.
            • Easy linking: Pages connect through internal links and simple formatting syntax rather than requiring complex code.
            • Access controls: Some wikis are open to the public, while internal or enterprise wikis restrict editing to team members or organizations.

            Usage Examples & Related Projects

            • Public reference: Wikipedia is the most famous example, alongside sister projects like Wiktionary, Wikibooks, and Wikivoyage.
            • Community & fan hubs: Dedicated wikis document video games, pop culture, or niche hobbies (often hosted on platforms like Fandom or MediaWiki).
            • Workplace documentation: Teams use internal wikis to store onboarding guides, meeting notes, and company policies.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            • Collaborative editing: Multiple users can add or revise content and track revision histories.
            • Easy linking: Pages connect through internal links and simple formatting syntax rather than requiring complex code.
            • Access controls: Some wikis are open to the public, while internal or enterprise wikis restrict editing to team members or organizations.

            Usage Examples & Related Projects

            • Public reference: Wikipedia is the most famous example, alongside sister projects like Wiktionary, Wikibooks, and Wikivoyage.
            • Community & fan hubs: Dedicated wikis document video games, pop culture, or niche hobbies (often hosted on platforms like Fandom or MediaWiki).
            • Workplace documentation: Teams use internal wikis to store onboarding guides, meeting notes, and company policies.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            • Collaborative editing: Multiple users can add or revise content and track revision histories.
            • Easy linking: Pages connect through internal links and simple formatting syntax rather than requiring complex code.
            • Access controls: Some wikis are open to the public, while internal or enterprise wikis restrict editing to team members or organizations.

            Usage Examples & Related Projects

            • Public reference: Wikipedia is the most famous example, alongside sister projects like Wiktionary, Wikibooks, and Wikivoyage.
            • Community & fan hubs: Dedicated wikis document video games, pop culture, or niche hobbies (often hosted on platforms like Fandom or MediaWiki).
            • Workplace documentation: Teams use internal wikis to store onboarding guides, meeting notes, and company policies.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            • Collaborative editing: Multiple users can add or revise content and track revision histories.
            • Easy linking: Pages connect through internal links and simple formatting syntax rather than requiring complex code.
            • Access controls: Some wikis are open to the public, while internal or enterprise wikis restrict editing to team members or organizations.

            Usage Examples & Related Projects

            • Public reference: Wikipedia is the most famous example, alongside sister projects like Wiktionary, Wikibooks, and Wikivoyage.
            • Community & fan hubs: Dedicated wikis document video games, pop culture, or niche hobbies (often hosted on platforms like Fandom or MediaWiki).
            • Workplace documentation: Teams use internal wikis to store onboarding guides, meeting notes, and company policies.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            • Collaborative editing: Multiple users can add or revise content and track revision histories.
            • Easy linking: Pages connect through internal links and simple formatting syntax rather than requiring complex code.
            • Access controls: Some wikis are open to the public, while internal or enterprise wikis restrict editing to team members or organizations.

            Usage Examples & Related Projects

            • Public reference: Wikipedia is the most famous example, alongside sister projects like Wiktionary, Wikibooks, and Wikivoyage.
            • Community & fan hubs: Dedicated wikis document video games, pop culture, or niche hobbies (often hosted on platforms like Fandom or MediaWiki).
            • Workplace documentation: Teams use internal wikis to store onboarding guides, meeting notes, and company policies.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            2
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
            • Collaborative editing: Multiple users can add or revise content and track revision histories.
            • Easy linking: Pages connect through internal links and simple formatting syntax rather than requiring complex code.
            • Access controls: Some wikis are open to the public, while internal or enterprise wikis restrict editing to team members or organizations.

            Usage Examples & Related Projects

            • Public reference: Wikipedia is the most famous example, alongside sister projects like Wiktionary, Wikibooks, and Wikivoyage.
            • Community & fan hubs: Dedicated wikis document video games, pop culture, or niche hobbies (often hosted on platforms like Fandom or MediaWiki).
            • Workplace documentation: Teams use internal wikis to store onboarding guides, meeting notes, and company policies.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            • Collaborative editing: Multiple users can add or revise content and track revision histories.
            • Easy linking: Pages connect through internal links and simple formatting syntax rather than requiring complex code.
            • Access controls: Some wikis are open to the public, while internal or enterprise wikis restrict editing to team members or organizations.

            Usage Examples & Related Projects

            • Public reference: Wikipedia is the most famous example, alongside sister projects like Wiktionary, Wikibooks, and Wikivoyage.
            • Community & fan hubs: Dedicated wikis document video games, pop culture, or niche hobbies (often hosted on platforms like Fandom or MediaWiki).
            • Workplace documentation: Teams use internal wikis to store onboarding guides, meeting notes, and company policies.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Additional Resources":
            """
            • Collaborative editing: Multiple users can add or revise content and track revision histories.
            • Easy linking: Pages connect through internal links and simple formatting syntax rather than requiring complex code.
            • Access controls: Some wikis are open to the public, while internal or enterprise wikis restrict editing to team members or organizations.

            Usage Examples & Related Projects

            • Public reference: Wikipedia is the most famous example, alongside sister projects like Wiktionary, Wikibooks, and Wikivoyage.
            • Community & fan hubs: Dedicated wikis document video games, pop culture, or niche hobbies (often hosted on platforms like Fandom or MediaWiki).
            • Workplace documentation: Teams use internal wikis to store onboarding guides, meeting notes, and company policies.
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
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy(__file__, output_file_path)
print(f"Data saved to {output_file_path}")
