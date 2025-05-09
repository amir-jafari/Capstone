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
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Time Series and Tabular Data Modeling: Classical Approaches vs. Transformer-Based Models""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The goal of this project is to explore and compare different approaches to modeling time series and tabular data. 
            The project will cover:

            1. Time Series Modeling: Investigate classical models such as AR (AutoRegressive), ARMA (AutoRegressive Moving Average),
               SARIMA (Seasonal AutoRegressive Integrated Moving Average), and the Box-Jenkins methodology. The aim is to apply 
               these models to multiple datasets to understand their strengths and limitations.

            2. Transformer-Based Models: Explore transformer-based models and their variations, such as TabNet and PyTorch transformers, 
               and how they can be adapted for time series forecasting.

            3. Comparison with Classical Techniques: Apply classical regression techniques on tabular datasets and compare their 
               performance with transformer-based models and neural network-based approaches.

            Students will work on finding and utilizing appropriate datasets, implement these models, and perform comparative analysis 
            to determine the effectiveness of each approach.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            Students will need to identify and utilize multiple datasets, which may include:

            1. Time Series Data: Financial data, energy consumption data, weather data, etc., which are fully time series datasets.

            2. Tabular Data: Datasets such as UCI Machine Learning Repository datasets, Kaggle datasets, or other publicly available 
               sources that are suited for regression analysis.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            This project addresses the need to understand how different modeling approaches perform on time series and tabular data. 
            By comparing classical methods with modern transformer-based techniques, students will gain insights into:

            1. The effectiveness of classical time series models on different types of data.
            2. How transformer models can be adapted for time series forecasting.
            3. The comparative performance of classical regression techniques versus modern deep learning approaches in tabular 
               data modeling.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            The project will be divided into several key phases:

            1. Dataset Selection: Identify and obtain datasets suitable for time series and tabular data analysis.

            2. Model Implementation:
                - Implement classical time series models (AR, ARMA, SARIMA, Box-Jenkins) on the time series datasets.
                - Apply transformer-based models such as TabNet and PyTorch transformers to the same datasets.
                - Perform regression analysis on tabular datasets using classical techniques and compare them with transformer 
                  and neural network-based models.

            3. Model Comparison:
                - Analyze the performance of each model using appropriate metrics (e.g., RMSE, MAE for time series; R², MSE for regression).
                - Compare and contrast the outcomes to draw conclusions about the strengths and limitations of each approach.

            4. Evaluation and Reporting:
                - Document the findings and present a comprehensive comparison between classical and modern approaches.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            This is a rough timeline for the project:

            - (2 Weeks) Dataset Selection and Familiarization
            - (4 Weeks) Time Series Modeling with Classical Techniques
            - (4 Weeks) Implementation of Transformer-Based Models
            - (4 Weeks) Regression Analysis on Tabular Data
            - (3 Weeks) Model Comparison and Evaluation
            - (2 Weeks) Final Reporting and Presentation
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            Given the scope of work and the need to explore multiple datasets and modeling techniques, this project is suitable 
            for a team of 1-2 students.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Potential challenges include:

            1. Data Preprocessing: Ensuring datasets are correctly preprocessed for both time series and tabular data analysis.
            2. Model Complexity: Understanding the complexities of transformer models and how to effectively adapt them for time 
               series data.
            3. Comparative Analysis: Developing a robust framework for comparing classical models with transformer-based approaches.
            4. Computational Resources: Managing the computational demands of training large transformer models.
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Dr. Amir Jafari",
        "Proposed by email": "ajafari@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gmail.com",
        "github_repo": "https://github.com/amir-jafari/Capstone",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy('json_gen.py', output_file_path)
print(f"Data saved to {output_file_path}")
