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
            """4""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2026""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Deep Learning for Video Understanding: Action Recognition and Temporal Feature Extraction""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """
            The goal of this project is to develop a comprehensive video understanding system that can classify 
            human actions, extract temporal features, and perform exploratory data analysis on video datasets. 
            Students will explore state-of-the-art deep learning architectures for video processing and build 
            end-to-end pipelines from raw video to actionable insights.

            Key Objectives:
            1. Build a robust video preprocessing pipeline that can:
               - Extract frames from videos at various frame rates (1 fps, 5 fps, 30 fps)
               - Perform video quality assessment (resolution, bitrate, compression artifacts)
               - Extract optical flow and motion features
               - Handle various video formats (MP4, AVI, MKV, WebM)
               - Generate video thumbnails and key frames

            2. Develop comprehensive video EDA (Exploratory Data Analysis) toolkit:
               - Statistical analysis: video length distribution, frame count, fps analysis
               - Visual analysis: frame diversity, motion intensity, scene complexity
               - Temporal analysis: shot detection, scene transitions, temporal patterns
               - Content analysis: object presence, dominant colors, brightness levels
               - Annotation analysis: class distribution, temporal annotations

            3. Implement and compare multiple video classification architectures:
               - 2D CNNs with temporal aggregation (ResNet-50 + LSTM)
               - 3D CNNs for spatiotemporal learning (C3D, I3D, R(2+1)D)
               - Two-stream networks (spatial + temporal streams)
               - Vision transformers for video (ViViT, TimeSformer, Video Swin Transformer)
               - Efficient models for deployment (MobileNetV3 + GRU, X3D)

            4. Extract and visualize temporal features:
               - Frame-level embeddings from pre-trained models
               - Temporal attention weights showing important frames
               - Action localization in untrimmed videos
               - Video summarization using key frame extraction

            5. Build practical applications:
               - Real-time action recognition from webcam
               - Video search and retrieval system
               - Automated video highlights generation
               - Anomaly detection in surveillance videos

            6. Create interactive visualization dashboard:
               - Video playback with frame-by-frame analysis
               - Feature embedding visualization (t-SNE, UMAP)
               - Confusion matrices and per-class metrics
               - Temporal heatmaps showing model attention

            7. Deploy as accessible tool:
               - Web application for video upload and analysis
               - REST API for batch video processing
               - Command-line tool for researchers
               - Documentation and tutorials
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            All datasets are publicly available with no access restrictions:

            PRIMARY DATASETS (Action Recognition):

            1. UCF101 (RECOMMENDED FOR BEGINNERS):
               - URL: https://www.crcv.ucf.edu/data/UCF101.php
               - Alternative: https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition
               - Size: 13,320 videos, ~6.5 GB
               - Classes: 101 human action categories
               - Content: YouTube videos of actions (sports, instruments, human-object interactions)
               - Duration: 2-16 seconds per video
               - Resolution: 320×240 pixels
               - Splits: 3 official train/test splits provided
               - Format: AVI files
               - Download: Direct download or Kaggle
               - Time: 15-20 minutes
               - Paper: https://arxiv.org/abs/1212.0402

            2. HMDB51 (Human Motion Database):
               - URL: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
               - Size: 7,000 videos, ~2 GB
               - Classes: 51 action categories
               - Content: Movies and YouTube clips
               - Duration: Variable length
               - Splits: 3 official train/test splits
               - Format: AVI files
               - Download: Direct download (registration required but instant)
               - Time: 5-10 minutes
               - Paper: https://ieeexplore.ieee.org/document/6126543

            3. Kinetics-400 (Large-Scale):
               - URL: https://github.com/cvdfoundation/kinetics-dataset
               - Alternative: https://www.kaggle.com/datasets/shivamb/kinetics-400-mini (preprocessed mini)
               - Size: ~400,000 videos, ~450 GB (full) or ~5 GB (mini)
               - Classes: 400 human action classes
               - Content: YouTube clips
               - Duration: ~10 seconds per clip
               - Resolution: Variable (typically 360p or higher)
               - Format: MP4 files
               - Download: Use provided downloader script or Kaggle preprocessed version
               - Time: Several hours for full dataset (recommend using preprocessed subset)
               - Paper: https://arxiv.org/abs/1705.06950
               - NOTE: Use Kinetics-400-Mini (5000 videos, 5 GB) for faster prototyping

            4. Something-Something V2 (Fine-Grained Actions):
               - URL: https://developer.qualcomm.com/software/ai-datasets/something-something
               - Size: 220,847 videos, ~20 GB
               - Classes: 174 action classes (e.g., "putting X into Y", "taking X from Y")
               - Content: Crowdsourced videos of humans interacting with objects
               - Duration: 2-6 seconds
               - Format: WebM files
               - Download: Free registration required (instant approval)
               - Time: 30-45 minutes
               - Paper: https://arxiv.org/abs/1706.04261

            5. Moments in Time (Scene Understanding):
               - URL: http://moments.csail.mit.edu/
               - Size: 1 million videos, ~100 GB (full) or ~5 GB (mini subset)
               - Classes: 339 different actions and activities
               - Content: 3-second clips from various sources
               - Duration: 3 seconds per video
               - Format: MP4 files
               - Download: Direct download or via provided scripts
               - Time: 10-15 minutes for mini, several hours for full
               - Paper: https://arxiv.org/abs/1801.03150


            SPECIALIZED DATASETS:

            6. ActivityNet (Untrimmed Videos - Advanced):
               - URL: http://activity-net.org/download.html
               - Size: 20,000 untrimmed videos, ~500 GB
               - Classes: 200 activity classes
               - Content: Long YouTube videos with temporal annotations
               - Duration: 1-10 minutes per video
               - Format: MP4 files
               - Download: YouTube downloader provided
               - Time: Several hours
               - Use Case: Temporal action localization
               - Paper: https://arxiv.org/abs/1705.00754

            7. UCF Crime Dataset (Anomaly Detection):
               - URL: https://www.crcv.ucf.edu/projects/real-world/
               - Size: 1,900 videos, ~30 GB
               - Classes: 13 anomaly types (robbery, assault, etc.)
               - Content: Surveillance videos
               - Duration: Variable length (untrimmed)
               - Format: MP4 files
               - Download: Direct download
               - Time: 45-60 minutes
               - Use Case: Video anomaly detection

            8. MSR-VTT (Video Captioning):
               - URL: https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/
               - Alternative: https://www.kaggle.com/datasets/vishnutheepb/msr-vtt
               - Size: 10,000 videos, ~40 GB
               - Content: Video clips with text descriptions
               - Duration: 10-30 seconds
               - Format: MP4 + JSON annotations
               - Download: Kaggle or official site
               - Time: 60-90 minutes
               - Use Case: Video-text multimodal learning


            PRE-EXTRACTED FEATURES (For Faster Prototyping):

            9. UCF101 Features:
               - I3D Features: https://github.com/hassony2/kinetics_i3d_pytorch
               - R(2+1)D Features: Available on Kaggle
               - Size: ~500 MB (much smaller than raw videos)
               - Format: NumPy arrays (.npy files)
               - Use Case: Skip video processing, directly train classifiers


            DATASET SELECTION GUIDE:

            Beginner (Start Here):
            - UCF101 (6.5 GB, 13K videos) - Best for learning
            - HMDB51 (2 GB, 7K videos) - Good for quick experiments

            Intermediate:
            - Kinetics-400-Mini (5 GB, 5K videos) - Subset for prototyping
            - Something-Something V2 (20 GB, 220K videos) - Fine-grained actions

            Advanced:
            - Kinetics-400 Full (450 GB) - Large-scale training
            - ActivityNet (500 GB) - Temporal localization


            RECOMMENDED STARTING POINT:
            Use UCF101 for initial development (small, well-documented, fast to download), then scale up 
            to Kinetics-400 or Something-Something V2 for final experiments and paper results.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Video understanding is a fundamental challenge in computer vision with applications spanning 
            surveillance, autonomous vehicles, healthcare, sports analytics, entertainment, and human-computer 
            interaction. Unlike images, videos contain rich temporal information that requires specialized 
            architectures to capture motion patterns and temporal dependencies.

            WHY THIS PROJECT IS TIMELY AND HIGHLY PUBLISHABLE:

            1. TRANSFORMERS FOR VIDEO ARE TRENDING:
               - Vision transformers (ViT) recently dominated image classification
               - Video transformers (ViViT, TimeSformer, Video Swin) are now state-of-the-art
               - Opportunity to compare against 3D CNNs and show efficiency gains
               - Self-attention reveals interpretable temporal patterns

            2. EFFICIENT VIDEO MODELS ARE CRITICAL:
               - Real-time video processing requires lightweight architectures
               - Mobile deployment is increasingly important (edge computing)
               - Model compression and knowledge distillation for video is under-explored
               - X3D, MobileViT-V2 show promise but need more evaluation

            3. TEMPORAL FEATURE EXTRACTION IS VALUABLE:
               - Pre-trained video models (like CLIP for images) are emerging
               - Transfer learning for video is less mature than for images
               - Feature visualization helps understand what models learn
               - Applications in video retrieval, summarization, captioning

            4. MULTIMODAL VIDEO UNDERSTANDING:
               - Combining visual, audio, and text (video captions) is hot
               - Few works systematically study multimodal fusion for video
               - CLIP4Clip, Frozen, etc. show potential but need more research

            5. PRACTICAL APPLICATIONS:
               - Sports analytics: automated highlight detection, player tracking
               - Healthcare: fall detection, patient monitoring, surgical video analysis
               - Security: anomaly detection, crowd analysis, violence detection
               - Education: automated lecture summarization, student engagement detection
               - Entertainment: video recommendation, content moderation, trend detection

            6. INTERPRETABILITY IMPERATIVE:
               - Black-box video models are hard to trust for critical applications
               - Attention visualization shows which frames matter
               - Saliency maps highlight important spatial regions over time
               - Temporal class activation maps (TCAM) for action localization

            7. PUBLICATION VENUES (STRONG ACCEPTANCE FOR QUALITY WORK):

               Top Computer Vision Conferences:
               - CVPR (Computer Vision and Pattern Recognition) - flagship CV conference
               - ICCV (International Conference on Computer Vision)
               - ECCV (European Conference on Computer Vision)
               - BMVC (British Machine Vision Conference)

               Machine Learning Conferences:
               - NeurIPS (Neural Information Processing Systems) - video learning track
               - ICML (International Conference on Machine Learning)
               - ICLR (International Conference on Learning Representations)

               Multimedia & Video:
               - ACM Multimedia (MM) - dedicated video understanding track
               - ICME (International Conference on Multimedia and Expo)
               - MMM (MultiMedia Modeling)

               AI Conferences:
               - AAAI (Association for the Advancement of AI)
               - IJCAI (International Joint Conference on AI)

               Specialized Workshops:
               - CVPR Workshop on Large Scale Video Understanding
               - ICCV Workshop on Video Analysis and Understanding
               - NeurIPS Workshop on Self-Supervised Learning

               Journals:
               - IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
               - International Journal of Computer Vision (IJCV)
               - IEEE Transactions on Multimedia (TMM)
               - Computer Vision and Image Understanding (CVIU)


            NOVELTY AND CONTRIBUTION OPPORTUNITIES:

            Students can contribute by:
            - Comprehensive benchmark comparing 2D CNNs, 3D CNNs, and transformers on same datasets
            - Efficient video transformers with reduced computational cost
            - Novel temporal attention mechanisms (hierarchical, sparse, adaptive)
            - Transfer learning study: ImageNet → UCF101 → Kinetics → target domain
            - Few-shot video classification with meta-learning
            - Self-supervised pre-training for video (contrastive learning, masked autoencoders)
            - Temporal action detection in untrimmed videos
            - Video quality assessment using deep learning
            - Explainable video classification with attention visualization
            - Real-time video understanding on edge devices (Raspberry Pi, mobile)


            RESEARCH QUESTIONS TO EXPLORE:

            1. How do different temporal modeling approaches compare? (LSTM vs. 3D Conv vs. Transformer)
            2. What is the optimal frame sampling rate for different action types?
            3. Do pre-trained image models transfer well to video tasks?
            4. How much does optical flow improve action recognition?
            5. Can we detect actions with minimal frames (1-5 frames)?
            6. What temporal patterns do models learn? (visualized via attention)
            7. How robust are video models to video compression and quality degradation?
            8. Can we achieve real-time performance on consumer hardware?


            BROADER IMPACT:

            - Accessibility: Tools for automatically captioning videos for hearing-impaired users
            - Safety: Automated detection of dangerous situations in surveillance
            - Healthcare: Non-intrusive monitoring of elderly patients
            - Education: Analyzing student engagement in online learning
            - Sports: Democratizing advanced analytics for amateur athletes
            - Environment: Wildlife monitoring and behavior analysis
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            (Due to character limits, this will be provided in the complete Python file)
            See complete implementation details in the full proposal document.

            PHASE 1: VIDEO DATA PIPELINE & EDA (Weeks 1-2)
            PHASE 2: BASELINE MODELS - 2D CNN + TEMPORAL AGGREGATION (Weeks 3-4)
            PHASE 3: 3D CONVOLUTIONAL NETWORKS (Weeks 5-6)
            PHASE 4: VIDEO TRANSFORMERS (Weeks 7-8)
            PHASE 5: TEMPORAL FEATURE EXTRACTION & VISUALIZATION (Weeks 9-10)
            PHASE 6: DEPLOYMENT & APPLICATIONS (Weeks 11-12)
            PHASE 7: PAPER WRITING & WEB APPLICATION (Weeks 13-14)
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
            Week 1:     Dataset Download & Video EDA
            Week 2:     Advanced Feature Extraction (Frames, Optical Flow, Motion Analysis)
            Week 3:     2D CNN + LSTM Baseline
            Week 4:     Alternative Temporal Aggregation (AvgPool, TempConv, GRU)
            Week 5:     3D CNN Implementation (C3D, I3D)
            Week 6:     Two-Stream Networks (RGB + Optical Flow)
            Week 7:     Video Transformers (ViViT)
            Week 8:     TimeSformer & Divided Attention
            Week 9:     Feature Extraction & Embedding Visualization
            Week 10:    Temporal Attention Visualization & Interpretability
            Week 11:    Real-Time Classification & Webcam Demo
            Week 12:    Video Highlights & Practical Applications
            Week 13:    Research Paper Writing
            Week 14:    Web Application Development & Code Release

            TOTAL: 14 weeks (one semester)

            KEY MILESTONES:
            - Week 2:  Clean dataset, EDA complete, frame extraction pipeline working
            - Week 4:  Baseline 2D models trained with results
            - Week 6:  3D CNN and two-stream models implemented
            - Week 8:  Video transformers trained
            - Week 10: All experiments complete, visualizations ready
            - Week 12: Applications built and tested
            - Week 14: Paper submitted, code published, demo deployed

            DELIVERABLES BY WEEK 14:
            - 8-page conference paper (CVPR format)
            - GitHub repository with full implementation
            - Pre-trained models on Hugging Face/Google Drive
            - Interactive web demo (Streamlit/Gradio)
            - Video demonstrations of applications
            - Comprehensive documentation and tutorials
            - Blog post explaining methodology
            - Presentation slides and poster
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            RECOMMENDED: 2-3 students

            ROLE DISTRIBUTION FOR 3 STUDENTS:

            Student 1: Data Engineer & EDA Specialist
            - Responsibilities:
              * Download and organize video datasets
              * Build video preprocessing pipeline (frame extraction, optical flow)
              * Perform comprehensive EDA with visualizations
              * Create data loaders for PyTorch
              * Handle video augmentation
              * Quality control and dataset documentation
            - Skills: Python, OpenCV, FFmpeg, Data Analysis, Visualization
            - Deliverables: Clean datasets, EDA report, preprocessing scripts

            Student 2: Deep Learning & Model Development Specialist
            - Responsibilities:
              * Implement baseline models (2D CNN + LSTM, 3D CNN)
              * Train and optimize models
              * Hyperparameter tuning and experiment tracking
              * Model evaluation and comparison
              * Create training pipelines
              * Performance benchmarking
            - Skills: PyTorch, Deep Learning, GPU Computing, Experiment Design
            - Deliverables: Trained models, training scripts, benchmark results

            Student 3: Advanced Architectures & Deployment Specialist
            - Responsibilities:
              * Implement video transformers (ViViT, TimeSformer)
              * Create attention visualization tools
              * Build real-time classification demo
              * Develop web application (Streamlit)
              * Feature extraction and embedding analysis
              * Model deployment and optimization
            - Skills: PyTorch, Transformers, Web Development, Visualization
            - Deliverables: Transformer models, visualization tools, web app, demos

            SHARED RESPONSIBILITIES (All Students):
            - Weekly team meetings for integration and planning
            - Collaborative paper writing (divided by sections)
            - Code reviews and documentation
            - Experiment result discussion and analysis
            - Presentation preparation
            - GitHub repository maintenance

            COMMUNICATION STRUCTURE:
            - Weekly progress meetings (90 minutes)
            - Daily standups (15 minutes via Slack/Discord)
            - Shared experiment tracking (Weights & Biases)
            - Google Docs for collaborative writing
            - GitHub for code collaboration with PR reviews
            - Notion/Trello for project management

            FOR 2 STUDENTS:
            - Student 1: Data + EDA + Baselines + Traditional 3D CNNs
            - Student 2: Video Transformers + Visualization + Applications + Deployment

            FOR 4 STUDENTS (If Available):
            - Student 4: Evaluation & Analysis Specialist
              * Cross-dataset evaluation
              * Statistical significance testing
              * Ablation studies
              * Error analysis
              * Benchmark comparisons with SOTA
              * Result visualization and paper figures
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Research Contributions":
            """
            This project offers multiple avenues for impactful research contributions:

            1. METHODOLOGICAL CONTRIBUTIONS:
            - Comprehensive benchmark study comparing 6+ architectures
            - Efficient video transformers with sparse attention
            - Novel temporal attention mechanisms
            - Transfer learning strategies for video

            2. EMPIRICAL CONTRIBUTIONS:
            - Extensive ablation studies on frame sampling, resolution, architectures
            - Temporal pattern analysis showing what models learn
            - Robustness evaluation against compression and degradation
            - Cross-dataset generalization study

            3. FEATURE EXTRACTION CONTRIBUTIONS:
            - Pre-trained video features for all major datasets
            - Video embeddings analysis and visualization
            - Action localization using temporal class activation maps

            4. INTERPRETABILITY CONTRIBUTIONS:
            - Attention visualization framework for video
            - Human evaluation of attention quality
            - Counterfactual analysis for video classification

            5. PRACTICAL APPLICATION CONTRIBUTIONS:
            - Open-source toolkit with pre-trained models
            - Real-time webcam demo
            - Video analysis tools (highlights, summarization, retrieval)

            PUBLICATION TARGETS:
            - CVPR, ICCV, ECCV (computer vision)
            - NeurIPS, ICML (machine learning)
            - ACM Multimedia (video understanding)
            - Workshops and journals

            EXPECTED OUTCOMES:
            - 1 conference paper
            - 1 GitHub repository (500+ stars)
            - Pre-trained models
            - Web demo
            - Blog post
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            TECHNICAL CHALLENGES:
            1. Computational resources (3D CNNs, transformers are memory-intensive)
            2. Dataset size (Kinetics-400 is 450 GB)
            3. Video loading speed (I/O bottleneck)
            4. Training time (video models are slow)
            5. Overfitting (video datasets smaller than image datasets)

            SOLUTIONS:
            - Use gradient accumulation, mixed precision training
            - Start with UCF101, use Kinetics-Mini
            - Pre-extract frames, use faster libraries (decord)
            - Use pre-trained weights, transfer learning
            - Strong augmentation, regularization, early stopping

            See complete issue list and mitigation strategies in full proposal.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Additional Resources":
            """
            DATASET DOWNLOAD LINKS:

            1. UCF101: https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition
            2. HMDB51: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
            3. Kinetics-400-Mini: https://www.kaggle.com/datasets/shivamb/kinetics-400-mini
            4. Something-Something V2: https://developer.qualcomm.com/software/ai-datasets/something-something

            PRE-TRAINED MODELS:
            - torchvision: r3d_18, r2plus1d_18, mc3_18
            - Third-party: I3D, C3D, SlowFast, TimeSformer, ViViT, X3D

            LIBRARIES:
            - torch, torchvision, opencv-python, decord
            - transformers, timm, einops
            - matplotlib, seaborn, tqdm, wandb

            See complete resources in full proposal document.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Dr. Amir Jafari",
        "Proposed by email": "ajafari@gwu.edu",
        "instructor": "Amir Jafari",
        "instructor_email": "ajafari@gwu.edu",
        "collaborator": "None",
        "funding_opportunity": "Open Source Community Project / NSF CAREER / Industry Partnerships",
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
print("\n" + "=" * 80)
print("DATASET DOWNLOAD INSTRUCTIONS")
print("=" * 80)
print("\n1. UCF101 (RECOMMENDED - Start Here):")
print("   Kaggle: https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition")
print("   Download: kaggle datasets download -d matthewjansen/ucf101-action-recognition")
print("\n2. HMDB51:")
print("   Official: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/")
print("   Registration required (instant, free)")
print("\n3. Kinetics-400 Mini (For Prototyping):")
print("   Kaggle: https://www.kaggle.com/datasets/shivamb/kinetics-400-mini")
print("   Download: kaggle datasets download -d shivamb/kinetics-400-mini")
print("\n4. Something-Something V2:")
print("   Official: https://developer.qualcomm.com/software/ai-datasets/something-something")
print("   Registration required (instant approval)")
print("\n" + "=" * 80)
print("All datasets are publicly available!")
print("Total size: ~35 GB (UCF101 + HMDB51 + Kinetics-Mini)")
print("Recommended start: UCF101 (6.5 GB) only")
print("=" * 80)