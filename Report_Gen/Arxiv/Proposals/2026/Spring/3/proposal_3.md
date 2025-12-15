
# Capstone Proposal
## modrl
### Proposed by: Tyler Wallett
#### Email: twallett@gwu.edu
#### Advisor: Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  

            Establish version == 1.0.0 of a modular Bandit, Classical and Deep RL Python Library with notation, examples and applications.
            

![Figure 1: Example figure](2026_Spring_3.png)
*Figure 1: Caption*

## 2 Dataset:  

            - Bandits: Synthetic NumPy probability distribution or datasets contingent on application.
            - Classical and Deep RL: Integration with Gymnasium or Farama Foundation Environments (env.reset(), env.step(), env.close())
            

## 3 Rationale:  

            Most RL Python Libraries are not modular or flexible. These libraries run in standard .train() and .predict()
            functionality which restricts researchers and practicioners.

            This project aims to develop a library that enables the user to have more of a modular functioanlity when it comes to 
            training Bandit and RL frameworks.
            

## 4 Approach:  

            In short, the capstone project will consists of three phases:
                  1. Package functionality and examples development (weeks 1-9)
                  2. Sphinx front-end documentation (weeks 10-11)
                  3. Research paper preparation (weeks 12-14)
            

## 5 Timeline:  

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
            


## 6 Expected Number Students:  

            One or maximum two students that have previously taken `DATS 6450: Reinforcement Learning` and attained a grade of A.
            

## 7 Possible Issues:  

            Collaboration and time-scheduling conflicts. 
            


## Contact
- Author: Amir Jafari
- Email: [ajafari@gwu.edu](mailto:ajafari@gwu.edu)
- GitHub: [https://github.com/twallett/modrl](https://github.com/https://github.com/twallett/modrl)
