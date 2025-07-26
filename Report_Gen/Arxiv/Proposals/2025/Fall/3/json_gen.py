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
            """2025""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Fall""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Predictive Modeling and Explainable AI for Autism Spectrum Disorder Diagnosis: Advanced Machine Learning Analysis of SFARI-Supported Datasets""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            This project aims to develop and validate advanced machine learning models for autism spectrum disorder (ASD) 
            diagnosis prediction using comprehensive datasets from SFARI-supported research in collaboration with the Department of 
            Psychological & Brain Sciences. The research focuses on creating novel interpretable AI algorithms that:

            1. **Predict autism diagnosis accurately** using structured behavioral data, demographic information, and unstructured 
            free-text responses from social inference tasks
            2. **Identify autism subclusters** within diagnosed populations using unsupervised learning techniques to discover phenotypic 
            subtypes and similar behavioral patterns
            3. **Extract insights from unstructured text data** using advanced natural language processing techniques on free-form responses 
            about social impressions and person descriptions
            4. **Develop explainable AI frameworks** that identify the most reliable predictors for both diagnosis classification and cluster 
            membership determination
            5. **Create generalizable models** that can transfer knowledge from experimental datasets to new populations and different 
            assessment measures
            6. **Implement feature importance analysis** to understand what behavioral, cognitive, and linguistic characteristics drive model predictions
            7. **Build cross-domain validation systems** that test model performance on out-of-sample data from different experimental paradigms

            The research contributes novel interpretable machine learning algorithms specifically designed for autism research and provides
             practical diagnostic tools that can enhance early identification and personalized intervention approaches. This work targets
              publication in top-tier venues such as Nature Machine Intelligence, JAMA Psychiatry, Molecular Autism, Autism Research, or
               specialized AI in healthcare journals, and aligns with SFARI's mission to advance autism research through data analysis.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            The project leverages comprehensive, real-world autism research datasets through established collaboration
             with the Department of Psychological & Brain Sciences and SFARI-supported resources:

            **Primary Experimental Datasets (Available through Collaboration):**

            1. **Social Inference Learning Task Dataset**:
               - Structured behavioral data from autism and neurotypical participants
               - Learning performance metrics across multiple experimental blocks
               - Demographic information including age, gender, cognitive abilities, and diagnostic status
               - Item preference ratings and similarity judgments for social stimuli
               - Computational modeling parameters from existing published research (Rosenblau et al., 2020, Biological Psychiatry: CCNI)

            2. **Free-Text Response Corpus** (Novel NLP Opportunity):
               - Unstructured written responses describing learned person impressions
               - Categorized content analysis including: social traits, personality descriptions, behavioral observations, trait adjectives usage
               - Multi-participant responses providing rich linguistic patterns associated with autism vs. neurotypical social cognition
               - Cross-validation dataset with different experimental setup (2 vs. 3 learning runs, different stimuli, varied response formats)

            3. **Neurocognitive Assessment Battery**:
               - Standardized cognitive assessments and behavioral questionnaires
               - Social cognition measures and autism diagnostic assessments
               - Individual difference measures in social inference abilities
               - Multi-modal data integration opportunities

            **SFARI-Supported Public Datasets for Model Validation:**

            4. **SPARK Dataset** (Simons Foundation Powering Autism Research):
               - 50,000+ participants with autism and family members
               - Comprehensive phenotypic data, medical histories, and behavioral assessments
               - Standardized diagnostic instruments and outcome measures
               - Available through SFARI Base with controlled access

            5. **Simons Simplex Collection (SSC)**:
               - Detailed phenotypic characterization of autism families
               - Cognitive assessments, behavioral measures, and developmental histories
               - High-quality diagnostic validation data for model benchmarking

            6. **Simons Searchlight Database**:
               - Genetic and phenotypic data from individuals with autism-associated gene variants
               - Longitudinal developmental data and outcome measures
               - Genotype-phenotype correlation opportunities

            **Benchmark Datasets from Literature:**

            7. **Published Autism ML Datasets**:
               - ASD screening datasets with established accuracy benchmarks (98% children, 81% adults performance)
               - Multi-site validation datasets for cross-population generalization testing
               - Established feature sets and evaluation metrics for performance comparison

            **Data Integration and Processing Framework:**
            - Subject-level data splits ensuring no individual appears in both training and test sets
            - Multi-modal data fusion techniques combining structured and unstructured data
            - Cross-dataset validation protocols for generalizability assessment
            - Privacy-preserving analysis methods compliant with HIPAA and IRB requirements
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Autism spectrum disorder affects 1 in 36 children, yet current diagnostic approaches rely heavily on 
            subjective clinical assessments that can delay diagnosis by years. Recent advances in machine learning have
             shown promising results for autism prediction, with some studies achieving 98% accuracy in children,
              but significant research gaps remain:

            **Research Gap 1: Limited Integration of Structured and Unstructured Data**
            Existing autism ML research primarily uses structured behavioral data or questionnaires, rarely integrating 
            rich unstructured text data that captures natural language patterns associated with autism. The free-text
             responses from social inference tasks represent a novel data source that could reveal linguistic and cognitive
              patterns invisible to traditional structured assessments.

            **Research Gap 2: Insufficient Explainable AI in Autism Research**
            Current machine learning approaches in autism research often function as "black boxes," providing predictions 
            without explaining which features drive classifications. This limits clinical utility and scientific
             understanding of autism heterogeneity. There's a critical need for interpretable models that can identify
              the most reliable diagnostic predictors.

            **Research Gap 3: Lack of Autism Subtype Discovery Using Advanced Clustering**
            While autism is recognized as highly heterogeneous, most research treats it as a single diagnostic category. 
            Advanced unsupervised learning techniques could identify meaningful subtypes within autism populations,
             potentially leading to more personalized interventions and better understanding of autism phenotypes.

            **Research Gap 4: Limited Cross-Domain Generalization Testing**
            Existing studies rarely test whether models trained on one experimental paradigm generalize to different 
            assessment approaches or populations. This limits real-world clinical applicability and scientific validity.

            **Research Gap 5: Underutilization of SFARI-Supported Large Datasets**
            Despite substantial investments in large-scale autism datasets (SPARK, SSC, Simons Searchlight), many datasets
             remain underutilized for advanced machine learning applications. This project directly addresses SFARI's call
              for increased utilization of existing data resources.

            This research addresses these critical gaps by developing novel interpretable machine learning algorithms 
            specifically designed for autism research, contributing to both computational psychiatry and clinical 
            autism research. The work has strong potential for high-impact publication and directly aligns with SFARI's mission:

            **Target Publication Venues:**
            - **Nature Machine Intelligence** (Impact Factor: 25.9) - Premier venue for AI applications in healthcare
            - **JAMA Psychiatry** (Impact Factor: 19.3) - Top psychiatric research journal
            - **Molecular Autism** (Impact Factor: 6.3) - Leading autism research venue
            - **Autism Research** (Impact Factor: 4.9) - Specialized autism research journal
            - **Nature Digital Medicine** (Impact Factor: 12.9) - Digital health and AI applications
            - **Journal of Medical Internet Research** (Impact Factor: 7.4) - Health informatics and digital health

            **SFARI Mission Alignment:**
            This project directly supports SFARI's goal to "advance the basic science of autism and related 
            neurodevelopmental disorders" through innovative data analysis of existing resources, maximizing the value of 
            previous research investments while generating new scientific knowledge.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            **Research Methodology**: Experimental machine learning approach combining supervised learning for diagnosis 
            prediction, unsupervised learning for subtype discovery, natural language processing for text analysis, and 
            explainable AI techniques for model interpretation

            **Phase 1: Data Integration and Preprocessing (Weeks 1-2)**
            - Comprehensive data integration of structured behavioral data, demographic information, and free-text responses
            - Advanced text preprocessing including tokenization, lemmatization, and feature extraction from social inference descriptions
            - Data quality assessment and missing value imputation using advanced techniques (MICE, KNN imputation)
            - Establishment of subject-level train-validation-test splits ensuring no data leakage between individuals
            - SFARI dataset access setup and initial exploratory data analysis

            **Phase 2: Advanced Predictive Modeling Development (Weeks 3-8)**
            - **Autism Diagnosis Prediction Models** (Weeks 3-4):
              * Implementation of ensemble methods (Random Forest, XGBoost, LightGBM) for structured data classification
              * Deep learning approaches using neural networks with attention mechanisms for complex pattern recognition
              * Multi-modal fusion techniques combining structured behavioral data with text features
              * Advanced feature engineering including interaction terms and domain-specific feature construction

            - **Natural Language Processing for Free-Text Analysis** (Weeks 5-6):
              * BERT and RoBERTa-based text classification for autism vs. neurotypical language patterns
              * Topic modeling (LDA, BERTopic) to discover themes in person descriptions across groups
              * Sentiment analysis and linguistic feature extraction (syntactic complexity, semantic coherence)
              * Custom embeddings training on autism-specific text data for domain adaptation

            - **Unsupervised Learning for Autism Subtype Discovery** (Weeks 7-8):
              * Advanced clustering algorithms (hierarchical clustering, DBSCAN, Gaussian mixture models) for autism phenotype discovery
              * Dimensionality reduction techniques (PCA, UMAP, t-SNE) for high-dimensional behavioral data visualization
              * Cluster validation using internal metrics (silhouette score, calinski-harabasz index) and external validation
              * Subtype characterization using interpretable machine learning techniques

            **Phase 3: Explainable AI and Interpretability Framework (Weeks 9-11)**
            - Implementation of model-agnostic interpretability techniques (SHAP, LIME, permutation importance)
            - Development of custom feature importance analysis for autism-specific behavioral patterns
            - Creation of interactive visualization dashboards for model explanation and clinical decision support
            - Cross-validation of feature importance across different models and datasets
            - Clinical relevance assessment of identified predictive features

            **Phase 4: Cross-Domain Validation and Generalization Testing (Weeks 12-14)**
            - Validation on SFARI-supported datasets (SPARK, SSC, Simons Searchlight) for model generalizability
            - Cross-dataset performance evaluation and domain adaptation techniques
            - Out-of-sample prediction testing on held-out experimental data with different paradigms
            - Statistical analysis of model robustness and generalization capabilities
            - Comparison with existing autism prediction benchmarks and published results

            **Phase 5: Research Publication and Clinical Translation (Weeks 15-16)**
            - Preparation of high-impact research manuscript for Nature Machine Intelligence or JAMA Psychiatry
            - Development of clinical decision support prototype and user interface mockups
            - Creation of open-source toolkit for autism researchers to apply developed methods
            - Preparation of SFARI grant application based on preliminary results for continued research
            - Documentation of best practices for autism ML research and interpretable AI in clinical settings
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            **Week 1**: Data integration, preprocessing, and comprehensive exploratory analysis of collaborative dataset
            **Week 2**: SFARI dataset access setup, text preprocessing, and establishment of evaluation frameworks
            **Weeks 3-4**: Supervised learning model development for autism diagnosis prediction using ensemble and deep learning approaches
            **Weeks 5-6**: Natural language processing implementation for free-text analysis and linguistic pattern discovery
            **Weeks 7-8**: Unsupervised learning algorithms for autism subtype discovery and phenotype clustering
            **Week 9**: Explainable AI framework implementation with SHAP, LIME, and custom interpretability techniques
            **Week 10**: Interactive visualization development and clinical decision support interface design
            **Week 11**: Feature importance validation and clinical relevance assessment of identified predictors
            **Week 12**: Cross-domain validation on SFARI datasets (SPARK, SSC, Simons Searchlight)
            **Week 13**: Out-of-sample generalization testing and statistical robustness analysis
            **Week 14**: Performance benchmarking against published results and comprehensive evaluation
            **Week 15**: High-impact journal manuscript preparation and submission to target venue
            **Week 16**: Clinical translation framework development, open-source toolkit creation, and future grant preparation
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            This project is designed for **1 student** with strong interdisciplinary background in:
            - **Machine Learning and Deep Learning** (required): Experience with ensemble methods, neural networks, 
            natural language processing, and explainable AI techniques
            - **Statistical Analysis and Data Science** (required): Advanced statistical methods, cross-validation, model evaluation, 
            and hypothesis testing
            - **Natural Language Processing** (preferred): Text preprocessing, transformer models (BERT, RoBERTa), and linguistic
             feature extraction
            - **Clinical Research Methods** (preferred): Understanding of psychological assessment, autism research, and clinical validation approaches
            - **Programming and Software Development** (required): Python, R, machine learning libraries (scikit-learn, PyTorch, transformers), and data visualization

            The interdisciplinary scope combining machine learning, clinical psychology, and autism research ensures the
             single student can make substantial individual contributions suitable for first-author publication in a high-impact venue.
              The collaboration with Department of Psychological & Brain Sciences provides domain expertise while maintaining clear individual 
              research contributions for the student.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
            **Novel Theoretical Contributions:**
            1. **Multi-Modal Autism Prediction Framework**: New machine learning architecture that integrates structured 
            behavioral data with unstructured text responses for enhanced diagnostic accuracy
            2. **Autism Phenotype Discovery Algorithms**: Novel unsupervised learning approaches specifically designed for 
            identifying autism subtypes using behavioral and linguistic patterns
            3. **Interpretable AI for Clinical Autism Research**: Advanced explainable AI techniques adapted for autism 
            research that provide clinically meaningful feature importance analysis
            4. **Cross-Domain Generalization Methodology**: Frameworks for testing autism ML model performance across different 
            experimental paradigms and populations

            **Methodological Contributions:**
            1. **Autism-Specific NLP Pipeline**: Comprehensive natural language processing framework for analyzing social 
            inference text responses in autism research
            2. **Clinical Interpretability Framework**: Systematic approach for translating machine learning insights into 
            clinically actionable information for autism diagnosis and intervention
            3. **Multi-Dataset Validation Protocol**: Standardized methodology for evaluating autism prediction models across 
            diverse SFARI-supported datasets
            4. **Privacy-Preserving Autism Research Methods**: Techniques for conducting advanced ML analysis while maintaining
             participant privacy and data security

            **Practical Contributions:**
            1. **Open-Source Autism ML Toolkit**: Complete software package for autism researchers to apply developed methods to new datasets
            2. **Clinical Decision Support Prototype**: Interactive system for clinicians to understand autism risk factors and diagnostic predictions
            3. **SFARI Data Analysis Framework**: Reusable pipeline for advanced analysis of SFARI-supported datasets
            4. **Autism Subtype Characterization Tool**: System for identifying and characterizing autism phenotypes in research and clinical settings

            **Publication Strategy and Target Venues:**

            **Primary Targets:**
            - **Nature Machine Intelligence** (Impact Factor: 25.9): Premier venue for AI applications in healthcare and biomedical research
            - **JAMA Psychiatry** (Impact Factor: 19.3): Top-tier psychiatric research journal with high clinical impact
            - **Molecular Autism** (Impact Factor: 6.3): Leading specialized venue for autism research with strong computational focus

            **Secondary Targets:**
            - **Nature Digital Medicine** (Impact Factor: 12.9): Digital health applications and AI in medicine
            - **Autism Research** (Impact Factor: 4.9): Specialized autism research venue with established audience
            - **Journal of Medical Internet Research** (Impact Factor: 7.4): Health informatics and digital health research

            **Conference Venues:**
            - **International Society for Autism Research (INSAR) Annual Meeting**: Premier autism research conference
            - **NeurIPS Machine Learning for Health (ML4H) Workshop**: AI applications in healthcare
            - **AAAI Conference on Artificial Intelligence**: AI methodology and applications
            - **AMIA Annual Symposium**: Biomedical informatics and health AI

            **SFARI-Specific Outcomes:**
            - **SFARI Webinar Presentation**: Dissemination of results to autism research community
            - **SFARI Annual Report Contribution**: Documentation of successful data utilization
            - **Follow-up Grant Applications**: Foundation for larger-scale SFARI funding requests

            **Expected Research Impact:**
            This research addresses fundamental challenges in autism diagnosis and research through innovative application 
            of explainable AI techniques. The integration of structured and unstructured data sources provides new insights 
            into autism heterogeneity and diagnostic markers. The work is positioned for high citation impact due to:
            - Direct clinical relevance for early autism identification
            - Novel methodological contributions to interpretable machine learning in healthcare
            - Comprehensive utilization of valuable SFARI-supported datasets
            - Open-source tool development for broader research community adoption

            **Broader Impact:**
            - **Clinical Practice**: Improved diagnostic tools for earlier autism identification
            - **Research Community**: Advanced methods for autism heterogeneity research
            - **Public Health**: Population-level screening and risk assessment capabilities
            - **Technology Transfer**: Foundation for commercial diagnostic tool development
            - **Policy Influence**: Evidence-based insights for autism research funding and clinical guidelines
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            **Technical Challenges and Mitigation Strategies:**

            **Challenge 1: Data Quality and Heterogeneity Across Sources**
            - Risk: Different data collection protocols and assessment instruments across collaborative and SFARI datasets may introduce bias
            - Mitigation: Implement robust data harmonization techniques, use domain adaptation methods, conduct sensitivity
             analyses across different data sources, and collaborate closely with domain experts for validation

            **Challenge 2: Model Interpretability vs. Performance Trade-offs**
            - Risk: Highly interpretable models may sacrifice predictive accuracy, while complex models may lack clinical interpretability
            - Mitigation: Develop ensemble approaches combining interpretable and complex models, implement post-hoc explanation techniques,
             validate interpretations with clinical experts, and create multiple model variants optimized for different use cases

            **Challenge 3: Generalization Across Autism Heterogeneity**
            - Risk: Models may not generalize well across different autism subtypes, age groups, or cognitive ability levels
            - Mitigation: Implement stratified validation approaches, develop subtype-specific models, use robust cross-validation
             techniques, and validate across diverse populations within SFARI datasets

            **Challenge 4: Natural Language Processing Complexity**
            - Risk: Free-text responses may be highly variable, contain minimal information, or require extensive preprocessing
            - Mitigation: Implement multiple NLP approaches (rule-based, statistical, neural), develop custom text quality metrics,
             use data augmentation techniques, and create backup strategies using structured data only

            **Challenge 5: SFARI Dataset Access and Integration Complexity**
            - Risk: Accessing and integrating multiple SFARI datasets may face administrative, technical, or timeline challenges
            - Mitigation: Establish early contact with SFARI Base administrators, prepare comprehensive data access requests, 
            develop modular analysis pipelines that can work with subsets of data, and maintain backup analysis plans

            **Challenge 6: Statistical Power and Sample Size Limitations**
            - Risk: Some analyses, particularly subtype discovery, may require larger sample sizes than available
            - Mitigation: Implement power analysis frameworks, use appropriate statistical techniques for smaller samples,
             conduct simulation studies, and focus on effect sizes rather than statistical significance alone

            **Ethical and Privacy Considerations:**

            **Challenge 7: Participant Privacy and Data Security**
            - Risk: Working with sensitive autism research data requires strict privacy protection and secure handling
            - Mitigation: Implement data de-identification protocols, use secure computing environments, follow IRB guidelines
             strictly, obtain necessary approvals, and develop privacy-preserving analysis methods

            **Challenge 8: Clinical Translation and Validation**
            - Risk: ML models may not translate effectively to real-world clinical settings or may create biased diagnostic tools
            - Mitigation: Collaborate with clinicians throughout development, conduct bias audits across demographic groups, 
            implement fairness-aware ML techniques, and design human-in-the-loop validation protocols

            **Project Management and Collaboration Risks:**

            **Risk: Interdisciplinary Coordination Complexity**
            - Mitigation: Establish clear communication protocols with Psychology & Brain Sciences collaborators, schedule
             regular meetings, define clear roles and responsibilities, and maintain detailed project documentation

            **Risk: Technical Implementation Timeline**
            - Mitigation: Prioritize core contributions, implement agile development practices, prepare backup analysis
             approaches, and maintain regular advisor consultations

            **Risk: Data Access and Processing Delays**
            - Mitigation: Start data access procedures early, develop parallel analysis tracks, implement automated
             preprocessing pipelines, and prepare contingency plans for data availability issues

            **Publication Strategy Risks:**

            **Risk: Competitive Research Environment**
            - Mitigation: Focus on unique methodological contributions and novel data integration approaches, target multiple 
            publication venues, prepare workshop papers for early visibility, and maintain active engagement with autism research community

            **Risk: Journal Review Process Complexity**
            - Mitigation: Ensure rigorous methodological approaches, collaborate with domain experts for clinical validation, 
            prepare comprehensive supplementary materials, and be prepared for iterative peer review process

            **SFARI Grant Application Considerations:**

            **Risk: Alignment with SFARI Priorities and Review Criteria**
            - Mitigation: Carefully study SFARI mission and funding priorities, engage with SFARI scientific team during 
            development, prepare preliminary results that demonstrate feasibility, and ensure clear articulation of autism research relevance
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Dr. Amir Jafari and Dr. Gabriela Rosenblau in collaboration with Department of Psychological & Brain Sciences",
        "Proposed by email": "ajafari@gwu.edu, grosenblau@email.gwu.edu>",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gwu.edu",
        "collaborator": "Department of Psychological & Brain Sciences",
        "funding_opportunity": "SFARI 2025 Data Analysis Request for Applications ($300,000 over 2 years)",
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