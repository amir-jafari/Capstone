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
            """3""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2026""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """modrl""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """
            Establish version == 1.0.0 of a modular Bandit, Classical and Deep RL Python Library with notation, examples and applications.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            - Bandits: Synthetic NumPy probability distribution or datasets contingent on application.
            - Classical and Deep RL: Integration with Gymnasium or Farama Foundation Environments (env.reset(), env.step(), env.close())
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Most RL Python Libraries are not modular or flexible. These libraries run in standard .train() and .predict()
            functionality which restricts researchers and practicioners.

            This project aims to develop a library that enables the user to have more of a modular functioanlity when it comes to 
            training Bandit and RL frameworks.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            In short, the capstone project will consists of three phases:
                  1. Package functionality and examples development (weeks 1-9)
                  2. Sphinx front-end documentation (weeks 10-11)
                  3. Research paper preparation (weeks 12-14)
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            Week 1:     `modrl/modrl/bandits/classical` (classical bandit algorithms: eg, ucb, ts)
            Week 2:     `modrl/modrl/evaluation` (classical bandit performance evaluation)
            Week 3:     `modrl/examples/bandits/classical` (collab notebook examples of classical bandit algorithms)
            Week 4:     `modrl/modrl/agents/classical` (classical RL: mc (on-off first visit), td (sarsa, q, doubleq))
            Week 5:     `modrl/modrl/evaluation` (classical RL performance evaluation)
            Week 6:     `modrl/examples/agents/classical` (collab notebook examples of classical RL algorithms)
            Week 7:     `modrl/modrl/agents/deep` (deep RL: dqn, vpg, ppo)
            Week 8:     `modrl/modrl/evaluation` (deep RL performance evaluation)
            Week 9:     (presentation)`modrl/examples/agents/deep` (collab notebook examples of deep RL algorithms)
            Week 10:    `sphinx` front-end docs: structure, notation, args, and returns
            Week 11:    `sphinx` front-end docs: style, collab examples, installation, and cheatsheets
            Week 12:    (package-live and research paper) research paper structure
            Week 13:    (research paper) draft research paper revision
            Week 14:    (research paper) final research paper revision

            Notes: since algorithm code will be provided for weeks 1, 4, and 7; students may create additional algorithms or 
            applications, if needed.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            One or maximum two students that have previously taken `DATS 6450: Reinforcement Learning` and attained a grade of A.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
            - Modular RL functionality.
            - Seamless gymnasium Integration.
            - Educational Alignment.
            - Designed for Research and Experimentation.
            - Open Source and Community-Driven.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Collaboration and time-scheduling conflicts. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Additional Resources":
            """
            Code and meeting time with Tyler Wallett will be provided.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Tyler Wallett",
        "Proposed by email": "twallett@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gwu.edu",
        "collaborator": "None",
        "funding_opportunity": "I wish :(",
        "github_repo": "https://github.com/twallett/modrl",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy(__file__, output_file_path)
print(f"Data saved to {output_file_path}")
print("\n" + "=" * 80)
print("DATASET DOWNLOAD INSTRUCTIONS")
print("=" * 80)
print("\n1. Fakeddit (PRIMARY - Largest multimodal dataset):")
print("   Kaggle: https://www.kaggle.com/datasets/mdepak/fakeddit")
print("   Download: kaggle datasets download -d mdepak/fakeddit")
print("\n2. MEME Dataset:")
print("   GitHub: https://github.com/TIBHannover/MM-Claims")
print("   Or Hugging Face: datasets.load_dataset('limjiayi/hateful_memes_expanded')")
print("\n3. FakeNewsNet:")
print("   GitHub: https://github.com/KaiDMML/FakeNewsNet")
print("   Download: git clone https://github.com/KaiDMML/FakeNewsNet.git")
print("\n4. LIAR Dataset (text baseline):")
print("   Direct: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
print("   Or: datasets.load_dataset('liar')")
print("\n5. MultiOFF:")
print("   GitHub: https://github.com/bharathichezhiyan/Multimodal-Offensive-Dataset")
print("\n" + "=" * 80)
print("All datasets are publicly available with no approval required!")
print("=" * 80)