
# Capstone Proposal
## Reinforcement Learning for Pseudo-Labeling
### Proposed by: Tyler Wallett
#### Email: twallett@gwu.edu
#### Advisor: Amir Jafari
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  

            The goal of this project is to develop a reinforcement learning (RL) framework that can automatically 
            assign pseudo-labels to unlabeled data, improving the performance of semi-supervised learning models. 
            The tool will be designed for data scientists and researchers to experiment with different RL strategies 
            to generate high-quality labels while minimizing labeling errors. 

            Develop or refine a methodological approach using RL to select unlabeled examples for labeling, balancing 
            exploration (diverse samples) and exploitation (high-confidence samples).   

            Apply the framework to publicly available datasets (e.g., CIFAR-10, MNIST) to 
            evaluate how pseudo-labeling improves model performance under limited labeled data.  

            Integrate the pseudo-labeling methodology into an open-source library to allow other researchers to 
            leverage RL-based semi-supervised labeling for their own datasets.
            

![Figure 1: Example figure](2025_Fall_6.png)
*Figure 1: Caption*

## 2 Dataset:  

            Publicly available classification datasets, such as: MNIST or CIFAR-10.
            

## 3 Rationale:  

            Semi-supervised learning is widely used when labeled data is limited, but manually labeling large datasets 
            is time-consuming and costly. Pseudo-labeling allows models to leverage unlabeled data by assigning estimated 
            labels, but existing heuristics may propagate errors and reduce model performance. 

            By applying reinforcement learning to pseudo-labeling, students can develop an adaptive strategy that selects 
            the most informative unlabeled examples and balances risk and reward in label assignment. This approach 
            reduces labeling errors, improves model accuracy, and provides a scalable solution for leveraging unlabeled 
            datasets in machine learning research.
            

## 4 Approach:  

            [Understanding the Reinforcement Learning (RL) Framework] 
            Students will learn how to formulate pseudo-labeling as an RL problem, including:

            - Understanding Markov Decision Process (MDP) assumptions and limitations: markov property, stationarity, limitations.
            - Understanding sequential decision making: actions affect future states and rewards.
            - Understanding terminal states and episodic tasks: pseudo-labeling epidode ends when all unlabeled data is processed.
            - State space design: selecting features of unlabeled examples, model predictions, and uncertainty measures 
            (e.g. [feature1, ..., featuren, softmax_predictions, cross_entropy]).
            - Action space design: assign a pseudo-label or skip labeling an example (e.g. {0, 1, ..., 9} for MNIST).
            - Reward design: reward high-confidence correct pseudo-labels and penalize incorrect assignments.
            - Algorithm selection: evaluate classical RL methods such as Q-learning, and deep RL methods such as Policy Gradient approaches.

            [`model.py` & `train.py`]
            Students will learn how to adapt existing RL repositories for pseudo-labeling:

            - Utilize existing RL repos (e.g. `twallett` GitHub Repo: `rl-lecture-code`) with existing RL models (e.g. Policy Gradient, PPO Clip).
            - Train RL agent on existing OpenAI Gymnasium environments (e.g. `CartPole-v1`) just to become familiar with code structure.

            [`utils/env.py` & `test.py`]
            Students will learn how to build a custom OpenAI Gymnasium environment for pseudo-labeling:

            - PseudoLabelEnv() & __init__() -> class: Initialize Custom OpenAI Gymnasium environment.
            OpenAI Gymnasium Core Methods:
            - self.reset() -> method/func: Reset environment to initial state.
            - self.step(action) -> method/func: Take action (assign pseudo-label or skip) and return next_state, reward, done, info.
            - (OPTIONAL) self.render() -> method/func: visualization of environment state.
            - self.close() -> method/func: Clean up resources.

            Additional Custom Pseudo-Labeling Methods:
                Basic MDP Custom Methods:
                - self.load_data() -> method/func: Load labeled and unlabeled datasets.
                - self.split_data() -> method/func: Split data into training and test sets.
                - self.get_state() -> method/func: Return feature representations for RL state.
                - self.calculate_reward() -> method/func: Compute reward based on correctness of pseudo-label and downstream model performance.

                Downstream Model Custom Methods:
                - DownstreamModel() & __init__() -> class: Initialize Downstream DL Model (MLP or CNN).
                - self.train_downstream_model() -> method/func: Train model on labeled + pseudo-labeled data.

                Evaluation Custom Methods:
                - evaluate_pseudo_labels() -> method/func: Compare pseudo-labels with ground truth for evaluation.
                - get_cum_rew() -> method/func: Compute cumulative reward over an episode.
                - get_classification_metric_gain() -> method/func: Measure improvement in downstream model using pseudo-labels.

            - `test.py` similar to `train.py` but for evaluating trained RL agent on unseen data.

            [`benchmark.py`]
            Students will learn how to systematically evaluate pseudo-labeling approaches:

            - Run experiments with different RL algorithms and hyperparameters.
            - Run experiments with different datasets.
            - Record pseudo-labeling performance and downstream model classification metrics.
            - Export results for analysis.
            

## 5 Timeline:  

            [Understanding RL Framework] 1 week
            [`model.py` & `train.py`] 2 weeks
            [`utils/env.py` & `test.py`] 7 weeks
            [`benchmark.py`] 2 weeks (start writting research paper here)
            


## 6 Expected Number Students:  

            Given the scope and complexity of the project, it is recommended to have 2-3 students working collaboratively.
            

## 7 Possible Issues:  

            - Reinforcement Learning concepts can be difficult for students to grasp, especially MDP assumptions.  
            - Designing state, action, and reward spaces for pseudo-labeling may be non-trivial.  
            - Training RL agents can be computationally expensive and time-consuming.  
            - Risk of overfitting to small datasets or poor generalization to new unlabeled data.  
            - Debugging custom environments and reward functions can be challenging.  
            - Evaluation of pseudo-label quality requires careful metric design.  
            


## Contact
- Author: Amir Jafari
- Email: [ajafari@gwu.edu](mailto:ajafari@gwu.edu)
- GitHub: [None](https://github.com/None)
