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
            """5""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2025""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Fall""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Health-Aware School Meal Recommendations with Contextual Bandits""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The goal of this project is to develop a free and open source analysis and recommendation tool that can be used 
            by non-technical school nutritionists, cafeteria staff, and researchers to optimize school meal offerings for 
            both student preference and healthiness. The tool will leverage Contextual Multi-Armed Bandit (CMAB) algorithms 
            to recommend meals that balance popularity and nutrition. Our project is affiliated with Fairfax County Public 
            Schools (FCPS), which provides the historical meal sales data used in this project. Built by data scientists, the 
            tool is designed to be used by non-technical stakeholders, empowering them to make data-driven decisions to 
            improve student nutrition while maintaining participation.

            Develop or refine a methodological approach using CMAB to recommend meals based on contextual features such as 
            school, time of day, and day of the week, while incorporating a healthiness weighting into the reward function.  

            Integrate this CMAB-based recommendation system into an open source library for school nutrition 
            research, so that future stakeholders can use your methodology to make healthier, data-informed decisions. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            FCPS Sales Data: Box Folder Link
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            School nutrition is a critical factor in student health, academic performance, and long-term well-being. However, 
            cafeteria participation is often influenced by student preferences for popular but less healthy food options, 
            making it challenging for nutritionists to balance meal appeal with nutritional goals. Data-driven approaches 
            can help address this challenge, but many school staff and researchers lack the technical expertise to analyze 
            meal sales and optimize offerings at scale. 

            Students can apply their Data Science and Reinforcement Learning skills to develop a methodology using 
            Contextual Multi-Armed Bandits (CMAB) that recommends meals based on both popularity and healthiness, 
            tailored to each school and time of day. By integrating this methodology into an open source tool, non-technical 
            users will be empowered to make evidence-based decisions that improve student nutrition while maintaining 
            participation rates. In doing so, students contribute to healthier school environments and provide a scalable 
            framework for future research in data-driven school nutrition planning.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            [Understanding the Reinforcement Learning (RL) Framework] 
            Students will learn the RL framework, focusing on Contextual Multi-Armed Bandits (CMAB), including:

            - Understanding CMAB assumptions and limitations: stationarity, independence, exploration-exploitation trade-off.
            - State space design: selecting relevant contextual features (e.g., school, time_of_day, day_of_week).
            - Action space design: defining meal options as arms in the bandit framework (e.g. item_id).
            - Reward design: formulating reward signals based on sales data and health penalty factor (reward = total_meals_served + λ * healthiness_score_of_item).
            - Algorithm selection: evaluating CMAB algorithms (e.g. LinUCB).

            [`utils/env.py`]
            Students will implement the CMAB environment to simulate meal recommendation scenarios using the FCPS dataset:

            - load_data() -> func: Load preprocessed FCPS sales CSV.
            - get_states() -> func: Return matrix (m × n) where m = time steps, n = contextual features.
            - get_actions() -> func: Return matrix (m × p) where m = time steps, p = meal options.
            - get_health_scores() -> func: Vector (p × 1) with healthiness score for each meal option.

            [`model.py` & `main.py`]
            Students will implement the LinUCB algorithm and related methods for training and inference:

            - LinUCB() & __init__() -> class: Initialize CMAB model.
            - self.train() -> method/func: Train the model on observed rewards.
            - self.action() -> method/func: Select a valid meal given the current context, considering only meals available at that time step (mask).
            - self.calculate_reward() (sometimes called bandit()) -> method/func: Compute reward for the chosen action.
            - self.update() -> method/func: Update model parameters based on observed reward.
            - self.reset() -> method/func: Reset the model to initial state.
            - self.save() -> method/func: Save model parameters.
            - self.recommend() -> method/func: Provide meal recommendations based on learned policy.

            - `main.py` - train and roughly evaluate the CMAB model using the custom environment and FCPS dataset.

            [`utils/metrics.py` & `utils/plot.py`]
            Students will learn how to measure performance and visualize results:

            - calulate_regret() -> func: Compute regret for evaluation.
            - calculate_cumulative_reward() -> func: Compute cumulative reward over time.

            - plot_top_meals() -> func: Visualize top-performing meals.
            - plot_recommendations() -> func: Plot recommendations over time or by context.

            [`benchmark.py`]
            Students will learn how to run systematic experiments and analyze results:

            - Run multiple experiments with different λ values.
            - Hyperparameter tuning.
            - bench_results_to_csv() -> func: Save benchmarking results to CSV for analysis.

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            [Understanding the Reinforcement Learning (RL) Framework] 1 week
            [`utils/env.py`] 2 weeks
            [`main.py` & `model.py`] 5 weeks
            [`utils/metrics.py` & `utils/plot.py`] 2 weeks (start writting research paper here)
            [`benchmark.py`] 2 weeks
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            Given the scope and complexity of the project, it is recommended to have 2-3 students working collaboratively.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
            This project will contribute to the broader research and educational community by providing an open-source 
            tool for optimizing school meal recommendations that balance student preferences and healthiness. The tool 
            will be designed for non-technical users, including school nutritionists, cafeteria staff, and educational 
            researchers. Research findings, including insights on student meal choices and health-driven reward weighting, 
            can be published in academic journals as well as presented in educational conferences, helping to inform 
            policy and improve school nutrition practices. The methodology and tool will be made openly available for 
            future researchers and school districts to utilize, enabling data-driven decisions to support healthier school 
            environments.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            - Implementing Contextual Multi-Armed Bandits (CMAB) can be complex for students.  
            - Designing the state and action spaces correctly may be challenging.  
            - Shaping the reward function to balance popularity and healthiness requires careful consideration.  
            - Handling unavailable meal options at each time step requires proper action masking.  
            - Debugging interactions between the environment and the bandit model can be difficult.  
            - Accurately computing cumulative rewards and regret is essential and may be error-prone.  
            - Data preprocessing and encoding categorical features from the FCPS dataset may present challenges.
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Tyler Wallett",
        "Proposed by email": "twallett@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gwu.edu",
        "collaborator": "Fairfax County Public Schools (FCPS)",
        "funding_opportunity": "Open Source Community Project",
        "github_repo": "None",
        # -----------------------------------------------------------------------------------------------------------------------
    }

os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")