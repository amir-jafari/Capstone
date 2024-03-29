import streamlit as st
import pandas as pd
import numpy as np
# Initialize session state for storing proposals if it doesn't exist
if 'proposals' not in st.session_state:
    st.session_state['proposals'] = []
if 'approved' not in st.session_state:
    st.session_state['approved'] = []
if 'rejected' not in st.session_state:
    st.session_state['rejected'] = []
# Initialize session state for uploaded document if it doesn't exist
if 'uploaded_word_doc' not in st.session_state:
    st.session_state['uploaded_word_doc'] = None
if 'uploaded_word_doc_name' not in st.session_state:
    st.session_state['uploaded_word_doc_name'] = ""
if 'completion' not in st.session_state:
    st.session_state['completion'] = []
if 'approved_completion' not in st.session_state:
    st.session_state['approved_completion'] = []
if 'edit_completion' not in st.session_state:
    st.session_state['edit_completion'] = []
if 'to_edit_proposal' not in st.session_state:
    st.session_state['to_edit_proposal'] = []
# If the 'editing_index' is not in session_state, add it.
if 'editing_index' not in st.session_state:
    st.session_state.editing_index = None
if 'show_edit_form' not in st.session_state:
    st.session_state.show_edit_form = None




def submit_proposal(proposal_data):
    # Here you would implement saving the proposal data to a database or another storage
    # For simplicity, we're appending it to the session state
    st.session_state['proposals'].append(proposal_data)
    
    st.success("Proposal submitted successfully!")


def proposal_request_form():
    with st.form("proposal_form"):
        st.subheader("Proposal Request Form")
        left_col, right_col = st.columns(2)
        
        with left_col:
            name = st.text_input("Name")
            project_name = st.text_input("Project Name")
            mentor = st.text_input("Mentor for the project")
            github_link = st.text_input("Github Link")
            objective = st.text_area("Objective")
            rationale = st.text_area("Rationale")
            timeline = st.text_area("Timeline")
            contributors = st.text_input("Contributors")

        with right_col:
            semester = st.selectbox("Semester", options=["Spring", "Summer", "Fall"])
            expected_students = st.number_input("Expected number of students", min_value=1, value=1)
            mentor_email = st.text_input("Mentor email")
            dataset = st.text_area("Dataset")
            approach = st.text_area("Approach")
            possible_issues = st.text_area("Possible Issues")
            year = st.selectbox("Year", options=["2021", "2022", "2023", "2024"])

        submitted = st.form_submit_button("Submit")
        
        if submitted:
            proposal_data = {
                "name": name,
                "project_name": project_name,
                "mentor": mentor,
                "github_link": github_link,
                "objective": objective,
                "rationale": rationale,
                "timeline": timeline,
                "contributors": contributors,
                "semester": semester,
                "expected_students": expected_students,
                "mentor_email": mentor_email,
                "dataset": dataset,
                "approach": approach,
                "possible_issues": possible_issues,
                "year": year,
            }
            # proposal_data = pd.DataFrame.from_dict(proposal_data, orient = "index")
            submit_proposal(proposal_data)
# Function to display pending proposals
# def pending_approval_page():
#     st.subheader("Pending Approval")
#     df = pd.DataFrame(st.session_state.proposals)
#     st.write(df)

def submit_proposal(proposal_data):
    st.session_state['proposals'].append(proposal_data)
    st.success("Proposal submitted successfully!")

def approve_proposal(index):
    proposal = st.session_state['proposals'].pop(index)
    st.session_state['approved'].append(proposal)
    st.rerun()

def reject_proposal(index):
    proposal = st.session_state['proposals'].pop(index)
    st.session_state['rejected'].append(proposal)
    st.rerun()

def edit_proposal(index):
    proposal = st.session_state['proposals'].pop(index)
    st.session_state['to_edit_proposal'].append(proposal)
    st.rerun()
def show_to_edit_proposals():
    df_edit = pd.DataFrame(st.session_state.to_edit_proposal)
    st.write(df_edit)

    for index, row in df_edit.iterrows():
        # The unique key for each button is created by appending the index to a base string
        if st.button(f"Edit {row['name']}", key=f"button_{index}"):
            # Save the index of the proposal being edited
            st.session_state['editing_index'] = index
            # Use Streamlit's session state to display the form
            st.session_state['show_edit_form'] = True
            # Break the loop to prevent more than one form from showing
            break
    # Check if we should display the editing form
    if st.session_state.get('show_edit_form', False):
        # Obtain the index of the proposal being edited
        index = st.session_state['editing_index']
        row = df_edit.loc[index]

        with st.form(key='edit_proposal_form'):
                st.subheader("Edit Proposal Request Form")
                left_col, right_col = st.columns(2)
                
                with left_col:
                    name = st.text_input("Name",value=df_edit.loc[index,"name"])
                    project_name = st.text_input("Project Name",value=df_edit.loc[index,"project_name"])
                    mentor = st.text_input("Mentor for the project",value=df_edit.loc[index,"mentor"])
                    github_link = st.text_input("Github Link",value=df_edit.loc[index,"github_link"])
                    objective = st.text_area("Objective",value=df_edit.loc[index,"objective"])
                    rationale = st.text_area("Rationale",value=df_edit.loc[index,"rationale"])
                    timeline = st.text_area("Timeline",value=df_edit.loc[index,"timeline"])
                    contributors = st.text_input("Contributors",value=df_edit.loc[index,"contributors"])

                with right_col:
                    semester = st.selectbox("Semester", options=["Spring", "Summer", "Fall"])
                    expected_students = st.number_input("Expected number of students",value=df_edit.loc[index,"expected_students"])
                    mentor_email = st.text_input("Mentor email",value=df_edit.loc[index,"mentor_email"])
                    dataset = st.text_area("Dataset",value=df_edit.loc[index,"dataset"])
                    approach = st.text_area("Approach",value=df_edit.loc[index,"approach"])
                    possible_issues = st.text_area("Possible Issues",value=df_edit.loc[index,"possible_issues"])
                    year = st.selectbox("Year", options=["2021", "2022", "2023", "2024"])

                submitted = st.form_submit_button("Submit")
                if submitted:
                            proposal_data_edit = {
                                "name": name,
                                "project_name": project_name,
                                "mentor": mentor,
                                "github_link": github_link,
                                "objective": objective,
                                "rationale": rationale,
                                "timeline": timeline,
                                "contributors": contributors,
                                "semester": semester,
                                "expected_students": expected_students,
                                "mentor_email": mentor_email,
                                "dataset": dataset,
                                "approach": approach,
                                "possible_issues": possible_issues,
                                "year": year,
                            }
                            # Update the appropriate proposal in the session state
                            st.session_state.to_edit_proposal[index] = proposal_data_edit
                            # Reset flags to hide the form
                            st.session_state['show_edit_form'] = False
                            st.session_state['editing_index'] = None

                            # Optionally, you can move the updated proposal back to the 'proposals' list
                            updated_proposal = st.session_state.to_edit_proposal.pop(index)
                            st.session_state.proposals.append(updated_proposal)

                            # Rerun the app to refresh the state and UI
                            st.rerun() 

def format_proposal_as_markdown(proposal):
    markdown_template = f"""
# Capstone Proposal
## {proposal["project_name"]}
### Proposed by: {proposal["name"]}
#### Mentor Email: {proposal["mentor_email"]}
#### Advisor: {proposal["mentor"]}
#### George Washington University  
#### Data Science Program

## 1 Objective:
{proposal["objective"]}

## 2 Dataset:
{proposal["dataset"]}

## 3 Rationale:
{proposal["rationale"]}

## 4 Approach:
{proposal["approach"]}

## 5 Timeline:
{proposal["timeline"]}

## 6 Expected Number of Students:
{proposal["expected_students"]}

## 7 Possible Issues:
{proposal["possible_issues"]}

## Contact
- Author: {proposal["name"]}
- Email: [{proposal["mentor_email"]}](mailto:{proposal["mentor_email"]})
- GitHub: [{proposal["github_link"]}]
"""
    return markdown_template

            
            



def pending_approval_page():

    if st.session_state['proposals']:
        for index, proposal in enumerate(st.session_state['proposals']):
            proposal_markdown = format_proposal_as_markdown(proposal)
            st.markdown(proposal_markdown, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Yes", key=f"approve_{index}"):
                    approve_proposal(index)
            with col2:
                if st.button("No", key=f"reject_{index}"):
                    reject_proposal(index)
            with col3:
                if st.button("Edit", key=f"edit_{index}"):
                    edit_proposal(index)
    else:
        st.write("No pending proposals")

def show_approved():
    df_approved = pd.DataFrame(st.session_state.approved)
    st.write(df_approved)
def show_rejected():
    df_rejected = pd.DataFrame(st.session_state.rejected)
    st.write(df_rejected)

# Function to save the uploaded Word document to the session state
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        # Read the file data into a bytes object
        bytes_data = uploaded_file.read()
        # Save the bytes data in the session state
        st.session_state['uploaded_word_doc'] = bytes_data
        st.session_state['uploaded_word_doc_name'] = uploaded_file.name
        st.success(f"Uploaded {uploaded_file.name} successfully.")

# Function to handle the project completion form
def completion_form():
    st.subheader("Project Completion Form")
    with st.form(key='completion_form'):
        # You can add other input fields as necessary
        project_title = st.text_input("Paper Title")
        video_link = st.text_input("Video Link")
        github_repo = st.text_input("GitHub Repository")
        project_website = st.text_input("Project Website Link if available")

        # Word document upload field
        uploaded_file = st.file_uploader("Upload your project document", type=['docx'])

        # Form submission button
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            # Save the uploaded Word document when the form is submitted
            save_uploaded_file(uploaded_file)
            completion = {
                "project title" : project_title,
                "Video Link" : video_link,
                "github repo" : github_repo,
                "project website": project_website,
                "Project Document" : uploaded_file.name
            }
            submit_completion(completion)


            # You can add logic to save other form fields to the session state or a database

def submit_completion(completion):
    st.session_state['completion'].append(completion)
    st.success("Project completion form submitted successfully!")

def approve_completion(index):
    completion = st.session_state['completion'].pop(index)
    st.session_state['approved_completion'].append(completion)
    st.rerun()
def edit_completion(index):
    completion = st.session_state['completion'].pop(index)
    st.session_state['edit_completion'].append(completion)
    st.rerun()
def show_approved_completion():
    df_approved = pd.DataFrame(st.session_state.approved_completion)
    st.write(df_approved)
def show_to_edit_completion():
    df_edit = pd.DataFrame(st.session_state.edit_completion)
    st.write(df_edit)



def pending_completion():
    if st.session_state['completion']:
        for index, completion in enumerate(st.session_state['completion']):
            st.write(completion)
            col1, col2 = st.columns()
            with col1:
                if st.button("Yes", key=f"approve_{index}"):
                    approve_completion(index)
            with col2:
                if st.button("Edit", key=f"reject_{index}"):
                    edit_completion(index)
    else:
        st.write("No pending project completion forms")


def main():
    st.image('gw-data-science-header.jpg', use_column_width=True)
    st.title("Data Science Capstone Website")
    page = st.selectbox("Navigate to:", ["Proposal Request", "Pending Approval","Edit Proposals", "Rejected", "Approved Projects", "Project Completion Form","Project completion aprroval", "Pending Completion", "Completed Projects"], index=0)

    if page == "Proposal Request":
        proposal_request_form()

    elif page == "Pending Approval":
        st.subheader("Pending Approval")
        pending_approval_page()

    elif page == "Edit Proposals":
        st.subheader("Edit Proposals")
        show_to_edit_proposals()
   
    elif page == "Rejected":
        st.subheader("Rejected")
        show_rejected()
  
    elif page == "Approved Projects":
        st.subheader("Approved Projects")
    
        show_approved()
    elif page == "Project Completion Form":

        completion_form()
        # Display info about the uploaded file (if any)
        if st.session_state['uploaded_word_doc'] is not None:
            st.write(f"Uploaded Word document: {st.session_state['uploaded_word_doc_name']}")
    elif page == "Project completion aprroval":
        st.subheader("Project completion aprroval")
        pending_completion()

    elif page == "Pending Completion":
        st.subheader("Pending Completion")
        show_to_edit_completion()
  
    
    elif page == "Completed Projects":
        st.subheader("Completed Projects")
        show_approved_completion()


if __name__ == "__main__":
    main()
