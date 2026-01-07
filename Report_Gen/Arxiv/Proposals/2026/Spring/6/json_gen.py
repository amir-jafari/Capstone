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
            """6""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2026""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Google Trends and Inflation (Time Series Project)""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """
            The team is seeking support to evaluate the use of Google Trends data in estimating inflation and other
            economic indicators. For this project, the students will review existing literature from the IMF
            and other organizations that have used Google Trends data to assess economic activity, and they will
            conduct their own analysis. The datasets will be the Consumer Price Index from countries with high 
            inflation and Google Trends data. After reviewing the literature, we expect the students to develop 
            a model to determine how CPI changes over time based on Google Trends data for specific terms.

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            Google data is proprietary, and students will need to be hired as non-fee consultants at the World Bank
            to access the data.       """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            Week 1:    
            Week 2:    
            Week 3:    
            Week 4:     
            Week 5:    
            Week 6:     
            Week 7:     
            Week 8:     
            Week 9:     
            Week 10:    
            Week 11:    
            Week 12:    
            Week 13:    
            Week 14:    

            TOTAL: 14 weeks (one semester)

            KEY MILESTONES:
            - Week 2:  
            - Week 4:  
            - Week 6:  
            - Week 8:  
            - Week 10: 
            - Week 12: 
            - Week 14: 
            
            DELIVERABLES BY WEEK 14:

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            RECOMMENDED: 1 students

            ROLE DISTRIBUTION FOR 1 STUDENTS:

            Student 1: 
            - Responsibilities:
             
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
          
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            TECHNICAL CHALLENGES AND SOLUTIONS:

        
            RISK MITIGATION TIMELINE:
            - Weeks 1-2: 
            - Weeks 3-4: 
            - Weeks 5-6: 
            - Weeks 7-8: 
            - Weeks 9-10: 
            - Weeks 11-12: 
            - Weeks 13-14: 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Additional Resources":
            """
                     """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "World Bank ",
        "Proposed by email": "",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gwu.edu",
        "collaborator": "",
        "funding_opportunity": "",
        "github_repo": "https://github.com/amir-jafari",
        # -----------------------------------------------------------------------------------------------------------------------
    }

os.makedirs(
    os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + os.sep + f'Arxiv{os.sep}Proposals{os.sep}{data_to_save["Year"]}{os.sep}{data_to_save["Semester"]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy(__file__, output_file_path)
print(f"Data saved to {output_file_path}")
