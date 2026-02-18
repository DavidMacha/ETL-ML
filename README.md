# ETL and Machine Learning Platform

A comprehensive, production-ready ETL pipeline management and ML experiment tracking platform with modern web dashboard.

![Dashboard Preview](assets/images/dashboard-preview.png)

## Overview

This platform provides end-to-end management of ETL (Extract, Transform, Load) pipelines and Machine Learning experiments. Originally based on the [ruslanmv/ETL-and-Machine-Learning](https://github.com/ruslanmv/ETL-and-Machine-Learning) project, this version has been completely refactored with:

- Modern Python backend with FastAPI
- React dashboard for real-time monitoring
- Comprehensive documentation and API reference
- Enhanced code quality with type hints and docstrings
- Data validation and quality monitoring
- ML experiment tracking with model versioning

## Features

### Core Features

- **Pipeline Management**: Create, configure, and run ETL pipelines
- **Experiment Tracking**: Track ML experiments with metrics and parameters
- **Model Registry**: Version and manage trained models
- **Data Validation**: Automated data quality checks and profiling
- **Real-time Monitoring**: Dashboard with live pipeline status
- **Comprehensive Logging**: Centralized logging with filtering

### Technical Highlights

- RESTful API with OpenAPI documentation
- MongoDB for scalable data storage
- React with Recharts for data visualization
- Type-safe Python with Pydantic models
- Comprehensive error handling

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB 4.4+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/etl-ml-platform.git
cd etl-ml-platform

# Backend setup
cd backend
pip install -r requirements.txt

# Frontend setup
cd ../frontend
yarn install
```

### Running the Application

```bash
# Start MongoDB (if not running as a service)
mongod --dbpath /data/db

# Start Backend (terminal 1)
cd backend
uvicorn server:app --host 0.0.0.0 --port 8001 --reload

# Start Frontend (terminal 2)
cd frontend
yarn start
```

The dashboard will be available at `http://localhost:3000`

### Seed Sample Data

Navigate to **Settings** in the dashboard and click **Seed Database** to populate with sample data.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │  Dashboard   │  Pipelines   │ Experiments  │ Data Quality │  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                            │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐  │
│  │ Pipeline API │Experiment API│  Model API   │Validation API│  │
│  └──────────────┴──────────────┴──────────────┴──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MongoDB Database                            │
│  ┌────────────┬────────────┬────────────┬────────────────────┐  │
│  │ pipelines  │experiments │   models   │ data_validations   │  │
│  └────────────┴────────────┴────────────┴────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## API Reference

### Base URL
```
http://localhost:8001/api
```

### Endpoints

#### Health Check
```
GET /api/health
```

#### Dashboard
```
GET /api/dashboard/stats      # Get dashboard statistics
GET /api/dashboard/metrics    # Get metrics for charts
GET /api/dashboard/recent-runs # Get recent pipeline runs
```

#### Pipelines
```
GET    /api/pipelines              # List all pipelines
POST   /api/pipelines              # Create new pipeline
GET    /api/pipelines/{id}         # Get pipeline by ID
DELETE /api/pipelines/{id}         # Delete pipeline
POST   /api/pipelines/{id}/run     # Run a pipeline
GET    /api/pipelines/{id}/runs    # Get pipeline runs
```

#### Experiments
```
GET    /api/experiments              # List all experiments
POST   /api/experiments              # Create new experiment
GET    /api/experiments/{id}         # Get experiment by ID
PUT    /api/experiments/{id}/metrics # Update metrics
POST   /api/experiments/{id}/complete # Complete experiment
DELETE /api/experiments/{id}         # Delete experiment
```

#### Models
```
GET  /api/models       # List all models
POST /api/models       # Register new model
GET  /api/models/{id}  # Get model by ID
```

#### Data Validations
```
GET  /api/validations       # List all validations
POST /api/validations       # Create new validation
GET  /api/validations/{id}  # Get validation by ID
```

#### Logs
```
GET  /api/logs  # List logs with optional filtering
POST /api/logs  # Create log entry
```

## Project Structure

```
etl-ml-platform/
├── backend/
│   ├── server.py           # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── .env               # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js         # Main React component
│   │   ├── App.css        # Styles
│   │   └── index.js       # Entry point
│   ├── package.json       # Node dependencies
│   └── tailwind.config.js # Tailwind configuration
├── docs/
│   ├── ARCHITECTURE.md    # Detailed architecture guide
│   ├── API.md            # API documentation
│   └── SETUP.md          # Setup instructions
└── README.md             # This file
```

## Configuration

### Backend Environment Variables

```env
MONGO_URL=mongodb://localhost:27017
DB_NAME=etl_ml_dashboard
```

### Frontend Environment Variables

```env
REACT_APP_BACKEND_URL=http://localhost:8001
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original ETL pipeline concept from [ruslanmv/ETL-and-Machine-Learning](https://github.com/ruslanmv/ETL-and-Machine-Learning)
- Apache Spark for data processing
- Apache Airflow for workflow orchestration
