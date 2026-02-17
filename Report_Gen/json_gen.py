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
            """7""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2026""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Enhancing a multi-agent platform to support health professions understanding of professional roles (Building on the existing BRIDGE Project)""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """
            Our team is seeking to take the next step in an already successful approach Of developing and implementing a custom multi agent generative AI platform for students in the health professions to help them understand roles and responsibilities. The system as it currently operates has six separate AI agents that a student can select from, each one of those agents is trained to behave and represents a different healthcare profession such as medical doctor, nurse, public health professional, healthcare administrator, social worker, physical therapist. We would like to enhance the realism of agent responses in this next iteration of the project. Interested students should have familiarity and a desire to create more human like and authentic AI agent interactions with humans that are training for professional roles. Trying to achieve here is a simulacrum of expertise. To create models of ideal collaborators in each one of these health professions. Agents that know when to share generously. To respond with just a couple sentences. Know when to refer to one of the other colleague agents so that students can get the best response from the right expert agent. 

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            Training and RAG data are either open source and freely available on the web or are proprietary GW resources that can be utilized for this project . Data is proprietary to GW and its partners and is FERPA protected, and students will need to agree to parameters of the GW IRB (Dr. Wiss) to work with data from student interactions with the platform.       """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """ Review open source code developed for the initial pilot and gain familiarity with the current system.  From there, plan for the addition of 3-4 additional agents, and develop a plan for new agent features such as: referring to a colleague agent when appropriate, appropriate response length, triggering another agent to respond/weigh-in where appropriate, expanding the knowledge base for each agent via RAG. From there we will engage in a process together where we will create new rules and behaviors for each one of the agents that mirrors expert mentorship and conversational responses for each of the generative AI agents. Also plan to increase the fidelity and helpfulness of their responses by growing the resource libraries for each AI agent using RAG.


            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            Week 1:    
            Week 2:    Weeks one and two - ideation, planning, and high level requirements gathering
            Week 3:    
            Week 4:    
            Week 5:    
            Week 6:    
            Week 7:    Weeks 3-7 - Prototype development and iteration round one
            Week 8:    Week 8 - ideation round 2 and develop final development plan 
            Week 9:    Weeks 9 through 12 - Taking the prototype
            Week 10:    
            Week 11:    
            Week 12:    Pilot test with 200 students from different health professions program 
            Week 13:    Weeks 13-14 Project wrap up and continuity planning
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
            RECOMMENDED: 2 students

            ROLE DISTRIBUTION FOR 1 STUDENTS:

            Student 1:
            - Responsibilities: Lead coder

 Student 2:
            - Responsibilities: Lead systems archtect

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
         Outside of this project students will have the opportunity to participate in research activities related to the pilot.  Please know that our last two pilots in this area have led to several publications and interested students working on the project will have the opportunity to also collaborate on this research separate from the requirements for this capstone course.
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
        "Proposed by": "Dr. Andrew Wiss - GWSPH assistant Dean for academic innovation and professorial lecture in the Department of Health Policy and management ",
        "Proposed by email": "awiss@gwu.edu",
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