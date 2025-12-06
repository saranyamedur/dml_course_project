# dml_course_project
# ML + MLOps Pipeline Project


## 📋 Project Overview

This project implements a production-ready machine learning pipeline with:
- **Problem**: Binary Classification (Breast Cancer Detection)
- **Dataset**: Wisconsin Breast Cancer Dataset (569 samples, 30 features)
- **Best Model**: Gradient Boosting Classifier
- **Deployment**: Streamlit Dashboard
- **Monitoring**: Evidently AI for drift detection

## 🏗️ Project Architecture

```
ML Pipeline Architecture:
┌─────────────────┐
│  Raw Data       │
└────────┬────────┘
         │ DVC Tracking
         ▼
┌─────────────────┐
│  Preprocessing  │◄─── Prefect Orchestration
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │◄─── MLflow Tracking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Best Model     │
└────────┬────────┘
          │
          |
         ▼
  ┌──────────┐
  │Streamlit │
  └─────┬────┘
     │            │
     └─────┬──────┘
           ▼
    ┌─────────────┐
    │   Docker    │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │  Evidently   │◄─── Monitoring
    └─────────────┘
```

## 📁 Project Structure

```
MiniProject/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD pipeline
│
├── data/
│   ├── raw/                       # Raw data (tracked by DVC)
│   │   └── dataset.csv
│   └── processed/                 # Processed data (tracked by DVC)
│       ├── train.csv
│       ├── test.csv
│       └── preprocessor.pkl
│
├── notebooks/
│   └── DataAnalysis.ipynb.ipynb              # Exploratory Data Analysis
│
├── src/
│   ├── download_data.py          # Dataset download utility
│   ├── preprocessing.py          # Data preprocessing module
│   └── train.py                  # Model training module
│
├── pipelines/
│   └── training_pipeline.py      # Prefect orchestration pipeline
│
├── deployment/
│   └── dashboard.py              # Streamlit dashboard
│
├── monitoring/
│   └── monitor.py                # Evidently monitoring
│
├── models/
│   └── best_model.pkl            # Trained model (tracked by DVC)
│
├── tests/
│   ├── test_preprocessing.py     # Preprocessing tests
│   ├── test_train.py             # Training tests
│
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Multi-container setup
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── .dvcignore                     # DVC ignore rules
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git
- Docker

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd dml_course_project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize DVC**
```bash
dvc init
```

5. **Download dataset**
```bash
python src/download_data.py
```

## 📊 Usage

### 1. Data Preprocessing

```bash
python src/preprocessing.py
```

This will:
- Load raw data
- Handle missing values
- Encode categorical features
- Scale numerical features
- Split into train/test sets
- Save processed data with DVC tracking

### 2. Model Training with MLflow

```bash
# Start MLflow UI (optional)
mlflow ui

# Train models
python src/train.py
```

Access MLflow UI at `http://localhost:5000` to view experiments.

### 3. Prefect Pipeline Orchestration

```bash
# Run the complete pipeline
python pipelines/training_pipeline.py
```

This orchestrates:
- Data loading
- Preprocessing
- Model training
- Model evaluation
- Model registration

### 4. Model Deployment



#### Streamlit Dashboard

```bash
# Start the dashboard
streamlit run deployment/dashboard.py
```

Access dashboard at `http://localhost:8501`

### 5. Docker Deployment

#### Build and run with Docker

```bash
# Build the image
docker build -t ml-pipeline:latest .

# Run the API container
docker run -p 8000:8000 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data ml-pipeline:latest
```

#### Using Docker Compose

```bash
# Start all services (API, Dashboard, MLflow)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

Services available:
- Streamlit: `http://localhost:8501`
- MLflow: `http://localhost:5000`

### 6. Monitoring with Evidently

```bash
# Generate monitoring reports
python monitoring/monitor.py
```

This generates:
- Data drift reports
- Data quality tests
- Model performance monitoring

Reports are saved to `monitoring/reports/`

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v --cov=src --cov=pipelines --cov=deployment
```

Run specific test files:

```bash
pytest tests/test_preprocessing.py -v
pytest tests/test_train.py -v

```

## 🔄 CI/CD Pipeline

GitHub Actions workflow automatically:
- Runs linting (Black, Flake8)
- Executes tests with coverage
- Builds Docker image
- Creates deployment artifacts

Triggered on:
- Push to `main` or `develop` branches
- Pull requests to `main`

## 📈 DVC Data Versioning

Track data and models with DVC:

```bash
# Add data to DVC tracking
dvc add data/raw/dataset.csv
dvc add data/processed/train.csv
dvc add data/processed/test.csv
dvc add models/best_model.pkl

# Commit DVC files
git add data/.gitignore data/*.dvc models/.gitignore models/*.dvc
git commit -m "Track data and models with DVC"

# Configure remote storage (optional)
dvc remote add -d myremote /path/to/dvc/storage
dvc push
```

## 🛠️ Technology Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10 |
| Version Control | Git, DVC |
| Experiment Tracking | MLflow |
| Pipeline Orchestration | Prefect |
| Dashboard | Streamlit |
| Containerization | Docker, Docker Compose |
| Monitoring | Evidently AI |
| Testing | Pytest |
| CI/CD | GitHub Actions |
| ML Libraries | scikit-learn, pandas, numpy |

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.95 | 0.94 | 0.96 | 0.95 |
| Random Forest | 0.96 | 0.95 | 0.97 | 0.96 |
| **Gradient Boosting** | **0.97** | **0.96** | **0.98** | **0.97** |

*Note: Actual metrics will vary based on the dataset used.*

## 🔍 Key Features

### A. Problem Definition & Dataset Selection ✅
- Binary classification problem
- Breast Cancer Wisconsin dataset (569 samples, 30 features)
- Well-balanced dataset for production ML

### B. Exploratory Data Analysis ✅
- Comprehensive EDA in Jupyter notebook
- Distribution analysis, correlation study
- Outlier detection, feature importance

### C. Data Preprocessing & DVC Tracking ✅
- Missing value handling
- Feature encoding and scaling
- Train-test split
- DVC versioning for data

### D. Model Development with MLflow Tracking ✅
- Multiple model comparison
- Hyperparameter logging
- Metrics tracking (accuracy, precision, recall, F1)
- Model artifact storage

### E. Prefect Pipeline Orchestration ✅
- Task-based workflow
- Retry mechanisms
- Sequential execution
- Error handling

### F. Repository Structure & Version Control ✅
- Organized folder structure
- .gitignore for Python projects
- DVC for data versioning
- Modular code organization

### G. CI/CD using GitHub Actions ✅
- Automated testing
- Code linting (Black, Flake8)
- Docker image building
- Deployment artifacts

### H. Local Model Deployment ✅
- Streamlit interactive dashboard
- Single and batch predictions
- Health check endpoints

### I. Containerization using Docker ✅
- Dockerfile for reproducibility
- Docker Compose for multi-service setup
- Volume mounting for models/data
- Port mapping configuration

### J. Local Monitoring using Evidently ✅
- Data drift detection
- Data quality tests
- Model performance tracking
- HTML report generation

## 📝 Development Workflow

1. **Feature Development**
   ```bash
   git checkout -b feature/new-feature
   # Make changes
   git add .
   git commit -m "Add new feature"
   git push origin feature/new-feature
   ```

2. **Data Changes**
   ```bash
   # Modify data
   dvc add data/raw/dataset.csv
   git add data/raw/dataset.csv.dvc
   git commit -m "Update dataset"
   ```

3. **Model Training**
   ```bash
   # Train with MLflow tracking
   python src/train.py
   # Best model is automatically saved
   dvc add models/best_model.pkl
   git add models/best_model.pkl.dvc
   git commit -m "Update model"
   ```

4. **Testing**
   ```bash
   pytest tests/ -v
   ```

5. **Deployment**
   ```bash
   docker-compose up -d
   ```

## 🐛 Troubleshooting

### Issue: Model not loading in API
**Solution:** Ensure model path is correct in environment variables:
```bash
export MODEL_PATH=models/best_model.pkl
export PREPROCESSOR_PATH=data/processed/preprocessor.pkl
```

### Issue: DVC pull fails
**Solution:** Configure DVC remote:
```bash
dvc remote add -d myremote /path/to/storage
dvc push
```

### Issue: Docker container can't access models
**Solution:** Check volume mounts in docker-compose.yml:
```yaml
volumes:
  - ./models:/app/models
  - ./data:/app/data
```

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC User Guide](https://dvc.org/doc)
- [Prefect Docs](https://docs.prefect.io/)
- [Evidently AI](https://docs.evidentlyai.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request


## 🎯 Project Checklist

- [x] Problem Definition & Dataset Selection
- [x] Exploratory Data Analysis (EDA)
- [x] Data Preprocessing & DVC Tracking
- [x] Model Development with MLflow Tracking
- [x] Prefect Pipeline Orchestration
- [x] Repository Structure & Version Control
- [x] CI/CD using GitHub Actions
- [x] Local Model Deployment ( Streamlit)
- [x] Containerization using Docker
- [x] Local Monitoring using Evidently

