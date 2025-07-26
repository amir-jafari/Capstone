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
            """1""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2025""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Fall""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Intelligent Document Processing Workflow Optimization: AI-Enhanced Automation using n8n and Advanced NLP Techniques""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            This project aims to develop and validate an intelligent document processing workflow optimization system that leverages 
            n8n automation platform enhanced with state-of-the-art Natural Language Processing (NLP) techniques. The research focuses 
            on creating novel algorithms that:

            1. **Automatically classify and route documents** using advanced transformer-based models (BERT, RoBERTa) optimized for workflow efficiency
            2. **Extract and structure information** from unstructured documents using named entity recognition (NER) and relation extraction techniques
            3. **Optimize processing workflows dynamically** based on document content, complexity, and organizational processing patterns
            4. **Predict processing bottlenecks and delays** using sequence-to-sequence models and historical workflow data
            5. **Generate intelligent workflow recommendations** that adapt to document types, user patterns, and organizational requirements
            6. **Implement real-time quality assessment** for processed documents using automated validation and error detection

            The research contributes novel NLP algorithms for document workflow optimization and provides a practical automation framework that significantly improves organizational document processing efficiency. This work targets publication in top-tier NLP and information systems venues such as EMNLP, NAACL, Information Systems, or ACM TOIS.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            The project leverages multiple comprehensive document processing datasets with established benchmarks:

            **Primary NLP Datasets with Clear Benchmarks:**
            1. **RVL-CDIP Dataset** (https://www.cs.cmu.edu/~aharley/rvl-cdip/):
               - 400,000 grayscale document images across 16 categories
               - Standard benchmark for document classification with established baselines (85.8% accuracy)
               - Includes invoices, letters, forms, emails, handwritten documents, advertisements, etc.

            2. **IIT-CDIP Test Collection** (https://ir.nist.gov/cdip/):
               - 6+ million document images from tobacco industry litigation
               - Complex document structures for information extraction benchmarking
               - Real-world document processing scenarios with ground truth annotations

            3. **FUNSD Dataset** (https://guillaumejaume.github.io/FUNSD/):
               - Form Understanding in Noisy Scanned Documents
               - 199 real, fully annotated forms for key-value extraction
               - Established benchmark for structured information extraction (F1-score: 79.27%)

            **Workflow and Process Optimization Datasets:**
            4. **Business Process Intelligence Datasets** (4TU.ResearchData):
               - Real-world business process logs with timestamps and document processing activities
               - Multiple organizational contexts for workflow pattern analysis
               - Performance metrics and bottleneck identification benchmarks

            5. **Document Processing Workflow Templates** (https://n8n.io/workflows/):
               - 200+ document automation workflow templates from n8n community
               - Various document processing patterns: OCR, classification, extraction, routing
               - Real-world automation scenarios and best practices

            **Specialized NLP Benchmarks:**
            6. **CoNLL-2003 NER Dataset** (https://www.clips.uantwerpen.be/conll2003/ner/):
               - Named Entity Recognition benchmark for information extraction
               - Established performance metrics (F1-score: 92.4% with BERT)
               - Critical for document content understanding and structuring

            7. **SQuAD 2.0 Dataset** (https://rajpurkar.github.io/SQuAD-explorer/):
               - Reading comprehension for question-answering on documents
               - Advanced document understanding benchmarks
               - Useful for intelligent information retrieval from processed documents

            **Synthetic and Augmented Datasets:**
            8. **Generated Document Workflows**: Custom synthetic workflows based on organizational patterns
            9. **Multilingual Document Processing**: Extended datasets for international document processing scenarios
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Organizations spend billions of hours annually on manual document processing, with studies showing that knowledge workers 
            spend 20-40% of their time searching for and processing documents. Current document processing workflows lack intelligent 
            optimization capabilities, leading to several critical research gaps:

            **Research Gap 1: Limited Intelligent Document Classification and Routing**
            Existing document processing systems use rule-based classification that cannot adapt to new document types or organizational
             changes. There's insufficient research on dynamic, context-aware document routing optimization.

            **Research Gap 2: Lack of Predictive Workflow Optimization**
            Current systems are reactive rather than predictive. No existing framework can predict processing bottlenecks based on 
            document characteristics and automatically optimize workflow allocation.

            **Research Gap 3: Insufficient Integration of Advanced NLP with Workflow Automation**
            Limited research exists on integrating state-of-the-art transformer models with practical workflow automation platforms for
             real-world document processing optimization.

            **Research Gap 4: Absence of Context-Aware Quality Assessment**
            Current document processing lacks intelligent quality control that adapts to document types, organizational standards, 
            and processing contexts.

            **Research Gap 5: Limited Multi-Document Workflow Optimization**
            Existing systems process documents individually without considering batch optimization, document relationships, or workflow 
            interdependencies.

            This research addresses these gaps by developing novel NLP-enhanced algorithms specifically designed for document workflow 
            optimization, contributing to both natural language processing and intelligent systems research. The work has strong potential
             for high-impact publication in top-tier venues:

            **Target Publication Venues:**
            - **EMNLP** (Empirical Methods in Natural Language Processing) - Top NLP conference
            - **NAACL** (North American Chapter of ACL) - Premier NLP venue
            - **Information Systems Journal** (Impact Factor: 6.2) - Leading IS journal
            - **ACM Transactions on Information Systems** (Impact Factor: 5.6)
            - **IEEE Intelligent Systems** (Impact Factor: 5.1)
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            **Research Methodology**: Experimental computer science approach combining advanced NLP model development, workflow optimization algorithms, and comprehensive empirical validation using established benchmarks

            **Phase 1: Data Preparation and Baseline Establishment (Weeks 1-2)**
            - Comprehensive preprocessing of RVL-CDIP and FUNSD datasets with workflow pattern extraction
            - Analysis of n8n document processing templates and organizational workflow patterns
            - Establishment of baseline performance metrics using existing document classification and extraction methods
            - Development of document complexity scoring and workflow mapping frameworks

            **Phase 2: Advanced NLP Model Development (Weeks 3-8)**
            - **Intelligent Document Classification System** (Weeks 3-4):
              * Fine-tuning BERT/RoBERTa models on RVL-CDIP dataset for enhanced document categorization
              * Development of hierarchical classification for complex document types
              * Integration of document content analysis with workflow routing optimization
              * Multi-label classification for documents requiring multiple processing paths

            - **Advanced Information Extraction Framework** (Weeks 5-6):
              * Named Entity Recognition (NER) system development using CoNLL-2003 benchmarks
              * Key-value pair extraction using FUNSD dataset with transformer architectures
              * Relation extraction algorithms for understanding document structure and dependencies
              * Layout-aware information extraction using visual and textual features

            - **Predictive Workflow Optimization Engine** (Weeks 7-8):
              * LSTM/Transformer-based models for predicting processing time and resource requirements
              * Bottleneck prediction using historical workflow data and document characteristics
              * Dynamic workflow allocation algorithms using reinforcement learning
              * Multi-objective optimization balancing accuracy, speed, and resource utilization

            **Phase 3: n8n Platform Integration and System Development (Weeks 9-11)**
            - Integration of developed NLP models into n8n workflow automation platform
            - Development of real-time document processing pipeline with intelligent routing
            - Creation of adaptive workflow management system with performance monitoring
            - Implementation of quality assessment and error detection mechanisms
            - Design of user-friendly dashboard for workflow monitoring and optimization insights

            **Phase 4: Comprehensive Evaluation and Benchmarking (Weeks 12-14)**
            - Performance evaluation on established datasets (RVL-CDIP, FUNSD, CoNLL-2003)
            - Comparison with existing document processing systems and baseline approaches
            - Workflow efficiency analysis using business process intelligence metrics
            - Statistical significance testing and ablation studies to validate component contributions
            - Real-world simulation using synthetic organizational document processing scenarios

            **Phase 5: Publication Preparation and System Documentation (Weeks 15-16)**
            - Preparation of research manuscript for submission to target venue (EMNLP, NAACL, or Information Systems)
            - Development of comprehensive open-source framework for community adoption
            - Creation of detailed technical documentation and deployment guides
            - Preparation of demonstration system with interactive document processing scenarios
            - Development of benchmark suite for future document workflow optimization research
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            **Week 1**: Dataset acquisition, preprocessing, and comprehensive literature review of document processing and NLP workflow optimization
            **Week 2**: Baseline establishment, workflow pattern analysis, and performance metric framework development
            **Weeks 3-4**: Advanced document classification system development using transformer models (BERT/RoBERTa fine-tuning)
            **Weeks 5-6**: Information extraction framework implementation with NER and relation extraction capabilities
            **Weeks 7-8**: Predictive workflow optimization engine development using sequence models and reinforcement learning
            **Week 9**: n8n platform integration, API development, and real-time processing pipeline implementation
            **Week 10**: Adaptive workflow management system development with intelligent routing and monitoring
            **Week 11**: Quality assessment system implementation and user interface development
            **Week 12**: Comprehensive evaluation on benchmark datasets (RVL-CDIP, FUNSD, CoNLL-2003)
            **Week 13**: Performance comparison with existing systems, statistical analysis, and ablation studies
            **Week 14**: Real-world simulation testing and workflow efficiency validation
            **Week 15**: Research manuscript preparation and submission to target publication venue
            **Week 16**: Open-source framework documentation, demonstration system finalization, and presentation preparation
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            This project is designed for **1 student** with strong technical background in:
            - **Natural Language Processing and Machine Learning** (required): Experience with transformer models (BERT, RoBERTa), deep learning frameworks (PyTorch/TensorFlow), and NLP preprocessing techniques
            - **Software Development and API Integration** (required): Python programming, RESTful API development, and workflow automation platform integration
            - **Data Analysis and Statistical Methods** (required): Performance evaluation, statistical significance testing, and benchmark analysis
            - **Document Processing Knowledge** (preferred): Understanding of OCR, document classification, and information extraction techniques

            The focused scope on document processing workflow optimization ensures the single student can make substantial individual contributions suitable for first-author publication in a high-impact NLP or information systems venue. The project combines established NLP techniques with novel workflow optimization approaches, providing clear individual research contributions.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
            **Novel Theoretical Contributions:**
            1. **Context-Aware Document Classification Algorithms**: Novel transformer-based approaches that consider both document 
            content and workflow context for optimal processing routing
            2. **Predictive Workflow Optimization Framework**: New algorithms for predicting document processing bottlenecks and 
            dynamically optimizing resource allocation
            3. **Multi-Modal Document Understanding**: Integration of visual and textual features for enhanced document analysis and 
            workflow decision-making
            4. **Adaptive Quality Assessment System**: Intelligent quality control algorithms that adapt to document types and 
            organizational processing standards

            **Methodological Contributions:**
            1. **Comprehensive Document Processing Benchmark Suite**: Standardized evaluation metrics and datasets for document 
            workflow optimization research
            2. **Real-Time Workflow Optimization Methodology**: Framework for continuous workflow improvement based on processing 
            patterns and performance feedback
            3. **Cross-Domain Document Processing Evaluation**: Systematic approach for evaluating document processing systems across 
            different organizational contexts

            **Practical Contributions:**
            1. **Open-Source Intelligent Document Processing Platform**: Complete n8n-based system for organizational document workflow optimization
            2. **NLP Integration Toolkit**: APIs and frameworks for integrating advanced NLP models with workflow automation platforms
            3. **Production-Ready Deployment Framework**: Scalable system architecture for real-world organizational implementation
            4. **Interactive Workflow Monitoring Dashboard**: User-friendly interface for tracking and optimizing document processing performance

            **Publication Strategy and Target Venues:**

            **Primary Targets:**
            - **EMNLP (Empirical Methods in Natural Language Processing)**: Premier venue for NLP applications and system development
            - **NAACL (North American Chapter of ACL)**: Top-tier conference for language processing research with practical applications

            **Secondary Targets:**
            - **Information Systems Journal** (Impact Factor: 6.2): Leading venue for information systems and workflow optimization research
            - **ACM Transactions on Information Systems** (Impact Factor: 5.6): Top journal for information retrieval and processing systems
            - **IEEE Intelligent Systems** (Impact Factor: 5.1): Excellent venue for AI applications in organizational systems

            **Workshop and Conference Venues:**
            - **Workshop on Document Intelligence at AAAI**: Specialized venue for document processing research
            - **ACM Symposium on Document Engineering**: Focused conference for document processing and workflow systems
            - **ICDAR (International Conference on Document Analysis and Recognition)**: Premier venue for document processing research

            **Expected Research Impact:**
            This research addresses fundamental challenges in organizational document processing with immediate practical applications. The integration of advanced NLP techniques with workflow automation provides both theoretical contributions to natural language processing and practical solutions for organizational efficiency. The work is positioned for high citation impact due to the universal need for document processing optimization across industries and the novel application of transformer models to workflow automation systems.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            **Technical Challenges and Mitigation Strategies:**

            **Challenge 1: Document Dataset Quality and Diversity**
            - Risk: Training datasets may not represent all organizational document types and processing scenarios
            - Mitigation: Combine multiple benchmark datasets (RVL-CDIP, FUNSD, IIT-CDIP), augment with synthetic data generation, 
            and develop domain adaptation techniques for new document types

            **Challenge 2: Model Performance and Computational Efficiency**
            - Risk: Transformer models may be computationally expensive for real-time document processing workflows
            - Mitigation: Implement model optimization techniques (quantization, pruning, distillation), develop efficient architectures,
             and design scalable processing pipelines with GPU acceleration

            **Challenge 3: Integration Complexity with Existing Systems**
            - Risk: Challenges integrating advanced NLP models with diverse organizational document management systems
            - Mitigation: Develop modular architecture with standard APIs, extensive compatibility testing, and flexible deployment 
            options (cloud, on-premise, hybrid)

            **Challenge 4: Evaluation and Validation Complexity**
            - Risk: Difficulty establishing comprehensive evaluation metrics that capture both NLP performance and workflow optimization effectiveness
            - Mitigation: Use established NLP benchmarks, develop custom workflow efficiency metrics, conduct ablation studies, and 
            perform statistical significance testing

            **Challenge 5: Generalization Across Document Types and Organizations**
            - Risk: Models may not generalize well to new document types or organizational contexts
            - Mitigation: Implement transfer learning approaches, develop domain adaptation techniques, and create extensive evaluation
             across diverse document categories and processing scenarios

            **Challenge 6: Real-Time Processing Requirements**
            - Risk: System may not meet real-time processing demands for high-volume document workflows
            - Mitigation: Optimize processing pipelines, implement efficient batch processing, develop caching mechanisms, and 
            design scalable distributed architecture

            **Project Management and Timeline Risks:**

            **Risk: Scope Expansion and Feature Creep**
            - Mitigation: Maintain clear weekly milestones, regular advisor consultations, prioritize core contributions, and prepare incremental publication strategy

            **Risk: Dataset Access and Preprocessing Complexity**
            - Mitigation: Early dataset acquisition, parallel preprocessing workflows, backup dataset options, and automated preprocessing pipelines

            **Risk: Model Training and Optimization Time**
            - Mitigation: Utilize pre-trained models, efficient fine-tuning strategies, cloud computing resources, and parallel experimentation approaches

            **Publication Strategy Risks:**

            **Risk: Competitive Research Environment**
            - Mitigation: Focus on novel workflow optimization aspects, target multiple publication venues, prepare workshop papers for
             early visibility, and maintain active research community engagement

            **Risk: Review Process Timeline**
            - Mitigation: Target conferences with faster review cycles, prepare backup venue options, develop strong experimental 
            validation, and ensure reproducible research practices
            """,

        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Dr. Amir Jafari",
        "Proposed by email": "ajafari@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gwu.edu",
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