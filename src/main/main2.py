#%%
# -*- coding: utf-8 -*-
"""
Author: Hema Chandra Puchakayala
Date: 2025-09-24
Version: 1.0
"""


import os
import sys
project_root = "/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12"
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# sys.path.insert(0, '/Users/hema/Desktop/GWU/Aug_2025/Capstone/fall-2025-group12')
from src.component.env_double_check import * 
from src.component.model import *
import numpy as np
import matplotlib.pyplot as plt

#%%

# all hyperparemeters in caps here and pass them to your environment class

'''
alpha: Dirichlet concentration for topic distributions and drift rate in preference updates. Higher = more uniform; lower = more peaked/specific.
a, b: Beta distribution parameters for initial user/document preference distributions. (The choice of a and b controls the skewness of the generated values.)
beta1, beta2: Coefficients controlling how document 'quality' and 'popularity' contribute to the user's click probability. (Higher beta1/beta2 increase the influence of quality/popularity in click scoring.)

'''
NUM_USERS = 1
NUM_DOCUMENTS = 2 # more than one document to choose from
SEED = 42
CATEGORIES = ['SPORTS','POLITICS']
# CATEGORIES = ['#MeToo','1976 U.S. bicentennial']#,'2000 Presidential Election','2004 Presidential Election','2008 Congressional Elections','2008 Presidential Election','2010 Congressional Elections','2012 Congressional Elections','2012 Presidential Election','2014 Congressional Elections','2016 Presidential Election','2018 Congressional elections','2020 Congressional elections','2020 Presidential Election','2024 Congressional elections','2024 presidential election','401(k)s','403(b)','529 plans','45911','Abortion','Academics','Accounting','Acid Reflux','ACT','Activism','Acupuncture','Acute Lymphocytic Leukemia','ADHD','Addiction','Adoption','Advanced Placement','adventure travel','Advertising','Affirmative Action','Affordable Care Act','aging','AIDS/ HIV','Airline Credit Cards','Airlines','Alaska Vacations','Alcohol','All-Inclusive Vacations','Allergic Rhinitis (hay Fever)','allergies','altcoin','Alternative Fuels','Alternative Medicine','Alternative Minimum Tax','Alzheimers Disease','Ambien','American Health Care Act','Anaphylactic Shock','Animal Cruelty','Animals','annuities','Anthrax','Anthropology','anti-inflammatory','Antibiotics','Antihistamines','Antioxidants','Anxiety','Aortic Aneurysm','Appearance','Arab region universities','Architecture','Arizona Vacations','Arrhythmia','Art','Arthritis','Artificial Intelligence','Arts And Sciences Graduate Programs','Aspirin','Assassinations','assisted living','asthma','Autism','Autoimmune Diseases','autonomous vehicles','Baby Boomers','Back Problems','Back To School','Bacteria','Bad Credit Credit Cards','Bailout','Balance Transfer Credit Cards','Banking','Bankruptcy','Baseball','Basketball','battery stocks','Beach Vacations','The Beatles','Bees','Behavior','Benefits','Benghazi','Berlin Wall','Biologic Drugs','Biology','biotechnology','Bipolar Disorder','Bird Flu','Birds','Birth','Birth Control','Birth Rate','Bitcoin','Black Friday','BlackBerry','blockchain','Blogs','Blood Disorders','Blood Sugar','Body Fat','Body Image','Bonds','Bone Joint Health','Bones','Books','Boston Marathon Bombing','BPA','Brain Cancer','brain health','Brain Tumor','Breast Cancer','Brexit','Broad Prize For Urban Education','Brokers','Bronchitis','Bronchodilators','Budget Travel','Bullying','Business','Business Credit Cards','Business Growth','Business School','business travel','C Sections','Cabinet Members','Caffeine','Calcium','California Vacations','Cameras','Campaign Finance','Campaigns','Camping Vacations','Campus Health','campus life','Cancer','Cancer Screening And Treatment','Candidates','capital gains','capital gains taxes','Capitalism','Car Manufacturers','Cardiology','Cardiomyopathy','Careers','Caregiving','Caribbean Vacations','Cars','Cash Back Credit Cards','CBD','Celebrities','Celiac Disease','Cervical Cancer','Charter Schools','Cheating','chemical weapons','Chemistry','Chemotherapy','Children','childrens health','Cholesterol','Christianity','Christmas','Chronic Lymphocytic Leukemia','Circulatory Conditions','Circumcision','Cities','Civil Rights','Civil War','Clinical Trials','Cloning','Clothing','cloud stocks','Coal','Cocaine','Cold War','College Admissions','College Applications','College Athletics','College Endowments','College Majors','College Search','Colleges','Colon Cancer','Colon Health','Colonoscopy','Colorado Vacations','Commodities','Common Cold','Common Core','Communism','Community','Community Colleges','Community Service','Computers','Concussions','Congenital Heart Disease','Congress','Conjunctivitis','Conservatives','Constipation','Constitution','Consultants','Consumers','Continuing Education','Convention2016','Cooking','COPD (chronic obstructive pulmonary disease)','Copyright','Coronary Artery Disease','Coronavirus','Company Culture','Corporate Earnings','Corporate Taxes','Corruption','Cosmetic Procedures','Cosmetics','Counseling','Couples Vacations','Courts','Cover Letters','CPAC','Credit','Credit Cards','Credit Monitoring','Credit Reports','Credit Scores','Credit Unions','Crime','Criminal Justice Programs','Criminal Law','Crohns Disease','Cruises','Cryptocurrency','CT scans','Culinary School','currency','Customer Service','Cybersecurity','Cycling','DACA','Death','Death Penalty','Debates','Debit Cards','Debt','decentralized finance','Defense Spending','Deficit And National Debt','Deforestation','Dementia','Democracy','Democratic National Convention','Democratic Party','Demographics','Dental Health','dentists','Depression','Dermatology','Detainees','Developing Countries','diabetes','Diabetes Type 1','Diabetes Type 2','diet and nutrition','Diets','Digestive Disorders','Digestive Health','Digital Piracy','Dinosaurs','Diplomacy','Disability','Disarmament','Discrimination','Diseases','Disney Vacations','Diversity','dividends','Divorce','DIY','Dizziness','DNA tests','Doctors','Dogs','Domestic Abuse','Dow Jones Industrial Average','Down Syndrome','DREAM Act','Driving','Drones','Dropout Rates','Drought','Drug Abuse','Drugs','e-commerce','E. coli','Ear Nose And Throat Conditions','Early Childhood Education','Earmarks','Earth Day','Earthquakes','Eating Disorders','Ebola','Economic Austerity','Economic Depression','Economic Growth','Economic Stimulus','Economics','Economy','Eczema','Education','Education Graduate School','Education Policy','Education Reform','Elections','Electoral College','electric vehicles','Electricity','Elementary School','Email','Emergency Planning','Emerging Markets','Emmys','Employment','Employment Law','Endangered Species','Endocrinology','Endometrial Cancer','Energy','Energy Efficiency','Energy Policy And Climate Change','Engineering','Engineering Graduate School','Enlarged Heart','Entertainment','Entrepreneurship','Environment','Epilepsy','Erectile Dysfunction','ESG investing','Espionage','Estate Planning','Estate Taxes','Estrogen','Ethanol','Ethereum','Ethics','Euro','Europe Vacations','Evangelicals','Every Student Succeeds Act','Evolution','Exchange Traded Funds','Executive Orders','Executive Pay','Executives','exercise and fitness','Existing Home Sales','Exports','Extinction','Extraterrestrials','Eye Health','Eye Problems','FAFSA','Fair Credit Credit Cards','Fall Vacations','Family','Family Health','Family Law','Family Vacations','Farm Bill','Farming','Fashion','Fatigue','Federal Budget','Federal Taxes','Female Voters','Feminism','Fever','Fibroids','Fibromyalgia','Financial Advisors','Financial Aid','Financial Goals','Financial Literacy','Financial Regulation','fintech','First Aid','First Ladies','FISA','Fiscal Cliff','Fiscal Policy','Fish','Flexible Spending Accounts','Floods','Florida Vacations','Food Allergies','Food And Drink','Food Safety','Food Stamps','Food Vacations','Foot Problems','Football','For Profit Colleges','Foreclosures','Foreign Aid','Foreign Investment','Foreign Policy','Fracking','Franchising','Fraternities And Sororities','Fraud','Freebies','Freedom of Information Act','Freedom Of Speech','Fuel Efficiency','Fuel Taxes','Fundraising','Funds','Funerals','futures','Gallbladder Problems','Gambling','Games','Gardasil','Gardening','Gas Credit Cards','Gas Prices','Gastroenterology','GDP','Gender','Gender Bias','gene editing','Generation X','Generic Drugs','Genetic Information Nondiscrimination Act','Genetics','Genocide','Genome','Gentrification','Geography','Geology','Georgia Vacations','GERD','Geriatrics','Gestational Diabetes','Ask Us Anything','Parental Advice','The Getting In interview','Gifts','Global Economy','Global Universities','Global Universities','Global Warming','Globalization','Gluten','GMAT','Gold','Golf','Gout','Government','Government Contracts','Government Intervention','Government Shutdown','Governors','GPS','Graduate Schools','Graduation Rates','Graves Disease','GRE','Great Depression','Greenhouse Gases','Gun Control And Gun Rights','H1N1','Habits','Hair Loss','Hanukkah','Happiness','Hate Crimes','Hawaii Vacations','Hazing','Headaches','health','Health Care','Health Care Reform','health equity','Health Insurance','Hearing Problems','Heart Attacks','heart disease','Heart Failure','Heart Health','Heart Surgery','Heart Transplants','heart-healthy diet','Heating','Heatstroke','Hedge Funds','Hemochromatosis','Hepatitis','Heroin','Hidden Fees','High Blood Pressure','high blood pressure diet','High Credit Limits Cards','High School','high-fiber diet','high-protein diet','Hinduism','Hip Replacement','Hiring','Hispanic Voters','Hispanics','Historically Black Colleges And Universities','History','Hockey','Holiday Shopping','Holidays','Holocaust','Home Improvements','Home Insurance','Home Prices','Home Refinancing','Home Renovations','Home Repair','Homeland (series)','Homelessness','Hormones','Horses','Hospitals','Hotel Credit Cards','Hotels and Resorts','Housing','Housing Market','HRT','Human Body','Human Papillomavirus','Human Rights','Hunting','Hurricane Harvey','Hurricane Irma','Hurricane Katrina','Hurricane Maria','Hurricane Sandy','Hurricanes','Hydrogen','Hysterectomy','Identity Theft','Illinois Vacations','Immigration','Immigration Reform','impact investing','Impeachment','Imports','Inauguration','Income','Income Investing','Income Tax','Incontinence','independent living','Index Funds','Indigestion','Inequality','Infant Mortality','Infants','Infections','infectious diseases','Infertility','Inflation','Influenza','Infrastructure','Injuries','Innovation','Insects','Insomnia','Insurance','Intellectual Property','Interest Rates','Interior Design','International Baccalaureate','international relations','International Students','International Trade','International Treaties','Internet','Internships','Interrogations','Interviewing','Investing','Investing for Retirement','Investing Insights','invisible braces','Iowa Caucus','iPhone','iPods','IPOs','Iraq War (1991)','Iraq War (2003 2011)','IRAs','Irritable Bowel Syndrome','Islam','January 6th','Jewelry','Applying','Jobs Reports','Journalism','Judaism','Juneteenth','junk bonds','K 12 Education','Keystone XL Pipeline','Kidnapping','Kidney Problems','Kidney Transplants','Knee Replacement','Korean War','Kurds','Kyoto Protocol','Labor','Lake Vacations','Landlording','landscaping','Languages','Laptops','Law','Law Practice','Law School','Lawsuits','Lead Poisoning','Leadership','Legislation','Leukemia','Lgbt Rights','Liberals','Libor','Libraries','Life Insurance','Litigation & Appeals','Liver Problems','Loans','Lobbying','long-term care','Longevity','Lottery','Louisiana Vacations','Low Interest Credit Cards','low-calorie diet','LSAT','Lung Cancer','Lung Disease','Lung Transplants','Lupus','Luxury','Lyme Disease','Macular Degeneration','Magazines','Malaria','Mammogram','Management','Mantle Cell Lymphoma','Manufacturing','March Madness','Marijuana','Marketing','Marriage','Massage','Math','MBAs','MCAT','Measles','Media','Medicaid','Medical Marijuana','Medical Prevention','Medical Quality','Medical Records','Medical School','Medical Screening','Medical Technology','Medical Travel','Medicare','Medicare Supplement','Medicine','Meditation','Meetings','melanoma','Memorial Day','Memory','memory care','mens health','Meningitis','Menopause','Mental Health','Mergers','Metabolic Syndrome','Metabolism','Metastatic Prostate Cancer','metaverse','Methodology','Michigan Vacations','Mid-Atlantic Vacations','Middle School','Mideast Peace','Midwest Vacations','Military','Military Bases','Military Courts','Military Credit Cards','Military Strategy','Milk','Millennials','Minimum Wage','Mining','Minority Students','Missiles','Mobility','Monetary Policy','Money','Money Market Funds','monkeypox','Montana Vacations','MOOCs','Mormonism','Mortgages','Mountain Vacations','Movies','Moving','MRSA','Multiple Sclerosis','Mumps','Murder','Muscle Problems','Music','mutual funds','NAFTA','Nasdaq','Nation','National Assessment Of Educational Progress','National Parks','National Security','National Security Terrorism And The Military','Native Americans','Natural Disasters','Natural Gas','Nephrology','Net Neutrality','Networking','Neuroimaging','Neurology','Nevada Vacations','New Deal','New England Vacations','New Hampshire Primaries','New Home Sales','New Jersey Vacations','New Year','New York Vacations','Newspapers','NFTs','No Annual Fee Credit Cards','No Balance Transfer Fees Credit Cards','No Child Left Behind','No Foreign Transaction Fee Credit Cards','Nobel Prize','Nonprofits','Norovirus','North Carolina Vacations','NSAIDs','Nuclear Power','Nuclear Weapons','Nurses','Nursing Homes','Nursing Programs','Obama Administration','Obama Transition','Obesity','Obstetrics And Gynecology','Occupational Health','OCD','Oceans','Offbeat','Ohio Vacations','oil','The Olympics','Oncology','Online Education','Online MBA','Online Savings Accounts','Ophthalmology','Opioids','options','Oral Cancer','Organic Food','Osteoarthritis','Osteopathy','Osteoporosis','Otolaryngology','Outdoor Living','Outsourcing','Ovarian Cancer','Over The Counter Drugs','Pacemakers','Pacific Northwest Vacations','Pain Management','Pancreatic Cancer','Pandemic','Pap Smear','parent loans','Parenting','Paris Terror Attacks','Parkinsons Disease','Passports','Patents','Patient Advice','Patient Safety','Patients','Paying For College','Paying For Community College','Paying For Graduate School','pediatric cardiology','Peer Pressure','Pell Grants','Pending Home Sales','Pennsylvania Vacations','Pensions','performing arts','personal budgets','Personal Finance','Personal Injury Law','Pets','Pharmacies','Philanthropy','Photo Galleries','Physical Therapy','Physics','Pirates','Plagiarism','Plants','Plastic','Plastic Surgery','Pneumonia','Podcasts','Police','Polio','Politics','Polls','Pollution','Polygamy','Population','Populism','Portfolio Management','Poverty','POWs','Prediabetes','Pregnancy','Preschool','Prescription Drugs','President','Presidential Pardon','Presidential Speech Transcripts','Prices','Primaries','Prison Sentences','Prisons','Privacy','Private Schools','Product Safety','Productivity','Products','Profits','Property Taxes','Proposition 8','Prostate','Prostate Cancer','Prosthetics','Prostitution','Protein','Psoriasis','Psoriatic Arthritis','Psychiatry','Psychology','PTSD','Public Health','Public Schools','Pulmonary Fibrosis','Pulmonology','Quality Of Life','Quantitative Easing','U.S. News Quizzes','Race','Racism','Radiation','Radiation Oncology','Radio','Rankings','Real Estate','Rebates','Recalls','Recession','Recipes','Recruiters','Recycling','Refugees','REITs','Relationships','Religion','Renewable Energy','Renting','Republican National Convention','Republican Party','Research','Respiratory Problems','Restaurants','Restless Leg Syndrome','Resumes','Retailers','Retirement','Revenue','Revolutionary War','Rewards Credit Cards','Rheumatoid Arthritis','Rheumatology','Rhodes Scholars','Ritalin','Road Trips','Robots','Running Mates','Russia investigation','Safety','Salaries and Benefits','Sales','Sales Tax','Salmonella','Sarbanes Oxley Act','SAT','Satellites','Saving For College','Savings','Schizophrenia','Scholarships','School Shootings','School Vouchers','Science','Scoliosis','Second Amendment','Second Careers','Secured Credit Cards','Seizures','semiconductors','Senior Citizens','Senior Health','Sensory Problems','sequestration','Sex','Sex Education','Sexism','Sexual Abuse','sexual assault','sexual health','Sexual Misconduct','Sexually Transmitted Diseases','Shingles','Shopping','short squeeze','Sickle Cell Disease','Sign Up Bonus Credit Cards','Sinusitis','Skiing','Skin Cancer','Skin Conditions','Slavery','sleep','Sleep Apnea','Sleep Disorders','Small Business','Smart Cities','Smartphones','Smoking And Tobacco','Soccer','Social Anxiety Disorder','Social Media','Social Networking','Social Security','Social Security Numbers','Socialism','socially responsible investing','Sociology','Sodium','Software','Solar Energy','Solo Vacations','South Carolina Vacations','Southeast Vacations','Southwest Vacations','Space','Space Station','SPDRs','Special Forces','Speech Problems','Speeches','Sports','Sports Medicine','spring','Spring Vacations','stablecoin','Stafford Loans','Standardized Tests','Starter Cards For Building Credit','Startups','State Budgets','State Law','State Of The Union','Statins','Steel','STEM','Stem Cells','STEM education','STEM jobs','Steroids','Stock Market','Stock Market News','Stress','Stroke','Student Credit Cards','student debt','Student Engagement','Student Loans','Students','Study Abroad','Subprime Mortgages','Sugar','Suicide','Summer','Summer Vacations','super PACs','Superdelegates','Supplements','Surgeon General','Surgery','Surveys','Sustainability','SUVs','Tanning','Target-Date Funds','Tariffs','Tax Code','Tax Cuts','Tax Deductions','Tax Exemptions','Tax Returns','Taxes','Tea Party','Teachers','Technology','Teen Pregnancy','Teens','Telecommuting','telehealth','Telephones','Television','Temporary Employment','Tennessee Vacations','Tennis','Terrorism','Testicular Cancer','Testing','Testosterone','Texas Vacations','Textbooks','Thanksgiving','Theme Park Vacations','Therapy','Thyroid','Tipping','Title IX','TOEFL','Torture','Tours','Toys','Trade','Traffic','Traffic Fatalities','Trafficking','Training','trains','Transfer Students','Transgender People','Transplants','Transportation','Travel','Travel Credit Cards','Travel Gear','Travel Insurance','Travel Rewards','Travel Tips','Tropical Vacations','Troubled Assets Relief Program','Trump transition','Tuberculosis','Tuition','U.S. Attorneys','U.S. intelligence','U.S. West Vacations','Ulcer','Ulcerative Colitis','Unemployment','Unions','United Kingdom Vacations','Unsecured Credit Cards','Urban Planning','US Vacations','Utah Vacations','Vacation Ideas','Vacation Rentals','Vacations','Vaccines','Vaping','vegetarian-friendly diet','Venture Capitalism','Veterans','Viagra','Vice President','Video Games','Vietnam War','Violence','Virginia Vacations','Vitamin D','Vitamins','Volcanoes','Volunteering','Voters','Wages','Wall Street','War Crimes','War In Afghanistan (2001 2014)','Washington DC Vacations','Washington State Vacations','Water','Water Park Vacations','Water Safety','Waterboarding','Watergate','Wealth','Weapons','Weather','Websites','Weekend Getaways','Weight Loss','weight loss drugs','West Nile Virus','Whales','White House','Wi Fi','Wildfires','Wills','wind power','Winter Vacations','Wiretapping','Wisconsin Vacations','Womens Colleges','womens health','Womens Rights','Women’s History','Work Life Balance','Working Women','World','World Cup','World News','World War I','World War II','Yoga','Young Professionals','Young Voters','Zika']

ALPHA = 0.1  
A = 2         
B = 5         
BETA1 = 0.3  
BETA2 = 0.3   
GAMMA1 = 0.3
GAMMA2 = 0.3

# environment hyperparameters should be passed here 
env = DriftingEnvironment(NUM_USERS, NUM_DOCUMENTS,ALPHA,A,B,BETA1,BETA2,GAMMA1,GAMMA2, CATEGORIES, SEED)


# model hyperparameters should be passed here

NUM_STATES = len(CATEGORIES) 
NUM_ACTIONS = NUM_DOCUMENTS
EPSILON = 0.05
ALPHA = 0.01
GAMMA = 0.01


# model = QLearning()

model = TDQ(NUM_STATES, NUM_ACTIONS, EPSILON, ALPHA, GAMMA, SEED)


# latent_preference_list = []
# random_model_rewards = []
# q_learning_model_rewards = []

#%%

EPISODES = 100
NUM_ROUNDS = 365


all_random_rewards = np.zeros((EPISODES, NUM_ROUNDS))
all_tdq_rewards = np.zeros((EPISODES, NUM_ROUNDS))

all_latent_prefs_actual =[]
all_latent_prefs_random = []  
all_latent_prefs_tdq = []

all_sigma_random = []  
all_phi_random = []

all_sigma_tdq = []
all_phi_tdq = []

tdq_cum_rew_list = []
random_cum_rew_list = []

for episode in range(EPISODES):
    print(f"\n--- Episode {episode + 1} ---")

    users, documents = env.reset() 
    # cumulative_reward_random = 0
    random_cum_rew = 0
    latent_preference_list_random = []
    sigma_list_random = []
    phi_list_random = []

    for step in range(NUM_ROUNDS):
        step_reward = 0
        for user_index, user in enumerate(users):
            action = np.random.choice(NUM_DOCUMENTS)
            selected_document = documents[action]
            reward, updated_user = env.step(user, selected_document)
            random_cum_rew += reward
            users[user_index] = updated_user
            # step_reward += reward
            latent_preference_list_random.append(updated_user.theta.copy())
            sigma_list_random.append(updated_user.sigma)
            phi_list_random.append(updated_user.phi)
            if episode == 0:
                all_latent_prefs_actual.append(updated_user.theta.copy())
    random_cum_rew_list.append(random_cum_rew)
        # step_reward = np.sum(step_reward)  
        # cumulative_reward_random += step_reward
        # all_random_rewards[episode, step] = cumulative_reward_random

    
    all_latent_prefs_random.append(np.array(latent_preference_list_random))
    all_sigma_random.append(np.array(sigma_list_random))
    all_phi_random.append(np.array(phi_list_random))

    # TDQ model run
    users, documents = env.reset()
    tdq_cum_rew = 0
    # cumulative_reward_tdq = 0
    latent_preference_list_tdq = []
    sigma_list_tdq = []
    phi_list_tdq = []

    for step in range(NUM_ROUNDS):
        step_reward = 0.0  
        for user_index, user in enumerate(users):

            click_probs = np.array([env.response_model.score(user, doc) for doc in documents])
            state_key = tuple(np.round(click_probs, 4)) 

            action = model.esoft(state_key, len(documents))
            selected_document = documents[action]

            reward, updated_user = env.step(user, selected_document)
            tdq_cum_rew += reward
            users[user_index] = updated_user

            scalar_reward = reward if np.isscalar(reward) else np.sum(reward)

            latent_preference_list_tdq.append(updated_user.theta.copy())
            sigma_list_tdq.append(updated_user.sigma)
            phi_list_tdq.append(updated_user.phi)

            click_probs_next = np.array([env.response_model.score(updated_user, doc) for doc in documents])
            next_state_key = tuple(np.round(click_probs_next, 4))
            model.update(action, state_key, next_state_key, scalar_reward)
    
    tdq_cum_rew_list.append(tdq_cum_rew)
all_latent_prefs_tdq.append(np.array(latent_preference_list_tdq))
all_sigma_tdq.append(np.array(sigma_list_tdq))
all_phi_tdq.append(np.array(phi_list_tdq))

avg_random_rewards = np.mean(all_random_rewards, axis=0)
avg_tdq_rewards = np.mean(all_tdq_rewards, axis=0)


avg_latent_prefs_random = np.mean(np.array(all_latent_prefs_random), axis=0)
avg_latent_prefs_tdq = np.mean(np.array(all_latent_prefs_tdq), axis=0)
avg_sigma_random = np.mean(np.array(all_sigma_random), axis=0)
avg_phi_random = np.mean(np.array(all_phi_random), axis=0)
avg_sigma_tdq = np.mean(np.array(all_sigma_tdq), axis=0)
avg_phi_tdq = np.mean(np.array(all_phi_tdq), axis=0)

#%%

plt.figure(figsize=(10, 6))
plt.plot(np.array(random_cum_rew_list), label='Random Model')
plt.plot(np.array(tdq_cum_rew_list), label='TDQ Model')
plt.xlabel('Time (Rounds)')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Over Time')
plt.legend()
plt.grid(True)
plt.show()

#%%

actual_latent_preferences = np.array(all_latent_prefs_actual) 
print(actual_latent_preferences.sum(axis=1)) 
plt.figure(figsize=(10, 6))
for category_index in range(actual_latent_preferences.shape[1]):
    plt.plot(actual_latent_preferences[:, category_index], label=f'Category {category_index}')

plt.xlabel('Time (Rounds)')
plt.ylabel('Latent Preference Score')
plt.title('User Latent Preferences Drift Over Time')
plt.legend()
plt.show()


#%%

num_categories = avg_latent_prefs_tdq.shape[1]

for category_index in range(num_categories):
    plt.plot(avg_latent_prefs_random[:, category_index], linestyle='dashed', label=f'Random Cat {category_index}')
    plt.plot(avg_latent_prefs_tdq[:, category_index], label=f'TDQ Cat {category_index}')

plt.xlabel('Time (Rounds)')
plt.ylabel('Average Latent Preference Score')
plt.title('Average Latent Preferences Over Episodes: Random vs TDQ')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()

# #%%
# plt.figure(figsize=(10, 6))
# plt.plot(avg_random_rewards, label='Random Model')
# plt.plot(avg_tdq_rewards, label='TDQ Model')
# plt.xlabel('Time (Rounds)')
# plt.ylabel('Average Cumulative Reward')
# plt.title('Average Cumulative Reward Over Time')
# plt.legend()
# plt.grid(True)
# plt.show()


plt.figure(figsize=(12, 6))

# %%
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
plt.figure(figsize=(10, 5))
plt.bar(range(len(documents[0].x)), height=documents[0].x, label='Document 0')
add_labels(CATEGORIES, documents[0].x)
plt.xlabel('Categories')
plt.ylabel('Topic Distribution')
plt.title('Document Topic Distributions')
plt.xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
# %%
def add_labels(x, y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
plt.figure(figsize=(10, 5))
plt.bar(range(len(documents[1].x)), height=documents[1].x, label='Document 1')
add_labels(CATEGORIES, documents[1].x)
plt.xlabel('Categories')
plt.ylabel('Topic Distribution')
plt.title('Document Topic Distributions')
plt.xticks(range(len(CATEGORIES)), CATEGORIES, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
# %%
num_categories = avg_latent_prefs_tdq.shape[1]

for category_index in range(num_categories):
    plt.plot(avg_latent_prefs_tdq[:, category_index], label=f'TDQ Cat {category_index}')

plt.plot(avg_sigma_tdq, linestyle='dashed', label=f'Sigma')
plt.plot(avg_phi_tdq, linestyle='dotted', label=f'Phi')

plt.xlabel('Time (Rounds)')
plt.ylabel('Average Latent Preference, Sigma, Phi Scores')
plt.title('Average Latent Preferences, Sigma, Phi Over Episodes: TDQ Model')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.show()

# %%

cumulative_random_rewards = np.cumsum(random_cum_rew_list)
cumulative_tdq_rewards = np.cumsum(tdq_cum_rew_list)

plt.figure(figsize=(10, 6))
plt.plot(cumulative_random_rewards, label='Random Model Cumulative Reward')
plt.plot(cumulative_tdq_rewards, label='TDQ Model Cumulative Reward')
plt.xlabel('Time (Rounds)')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Over Time')
plt.legend()
plt.grid(True)
plt.show()
# %%
