"""
ETL & ML Dashboard - Backend API Server

A comprehensive FastAPI-based backend for managing ETL pipelines, 
ML experiments, and data quality monitoring.

Features:
- Pipeline management and execution tracking
- ML experiment tracking with model versioning
- Data validation and quality metrics
- Real-time pipeline monitoring
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from enum import Enum
import os
import uuid
import random
from pymongo import MongoClient

# ============================================================================
# Configuration & Database Setup
# ============================================================================

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "etl_ml_dashboard")

client = MongoClient(MONGO_URL)
db = client[DB_NAME]

# Collections
pipelines_collection = db["pipelines"]
pipeline_runs_collection = db["pipeline_runs"]
experiments_collection = db["experiments"]
models_collection = db["models"]
data_validations_collection = db["data_validations"]
logs_collection = db["logs"]

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ETL & ML Dashboard API",
    description="Backend API for ETL pipeline management and ML experiment tracking",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Enums & Models
# ============================================================================

class PipelineStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"

class PipelineStepType(str, Enum):
    EXTRACT = "extract"
    TRANSFORM = "transform"
    LOAD = "load"
    VALIDATE = "validate"
    TRAIN = "train"

class PipelineStep(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: PipelineStepType
    config: Dict[str, Any] = {}
    position: Dict[str, int] = {"x": 0, "y": 0}

class PipelineCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    steps: List[PipelineStep] = []
    schedule: Optional[str] = None

class PipelineResponse(BaseModel):
    id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    status: PipelineStatus
    schedule: Optional[str]
    last_run: Optional[str]
    created_at: str
    updated_at: str
    run_count: int

class PipelineRunResponse(BaseModel):
    id: str
    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    started_at: str
    finished_at: Optional[str]
    duration_seconds: Optional[float]
    steps_completed: int
    total_steps: int
    logs: List[str]
    error: Optional[str]

class ExperimentCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    pipeline_id: Optional[str] = None
    parameters: Dict[str, Any] = {}

class ExperimentResponse(BaseModel):
    id: str
    name: str
    description: str
    pipeline_id: Optional[str]
    status: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    model_version: Optional[str]
    created_at: str
    finished_at: Optional[str]

class ModelResponse(BaseModel):
    id: str
    name: str
    version: str
    experiment_id: str
    algorithm: str
    metrics: Dict[str, Any]
    parameters: Dict[str, Any]
    file_path: Optional[str]
    created_at: str
    status: str

class DataValidationCreate(BaseModel):
    name: str
    dataset_path: str
    rules: List[Dict[str, Any]] = []

class DataValidationResponse(BaseModel):
    id: str
    name: str
    dataset_path: str
    status: str
    rules_passed: int
    rules_failed: int
    total_rules: int
    issues: List[Dict[str, Any]]
    profile: Dict[str, Any]
    created_at: str
    finished_at: Optional[str]

class LogEntry(BaseModel):
    id: str
    timestamp: str
    level: str
    source: str
    message: str
    metadata: Dict[str, Any] = {}

class DashboardStats(BaseModel):
    total_pipelines: int
    active_pipelines: int
    total_experiments: int
    successful_runs_24h: int
    failed_runs_24h: int
    total_models: int
    data_validations_passed: int
    data_validations_failed: int

# ============================================================================
# Helper Functions
# ============================================================================

def generate_id() -> str:
    """Generate a unique ID"""
    return str(uuid.uuid4())[:8]

def get_current_time() -> str:
    """Get current UTC time as ISO string"""
    return datetime.now(timezone.utc).isoformat()

def serialize_doc(doc: Dict) -> Dict:
    """Remove MongoDB _id from document"""
    if doc and "_id" in doc:
        del doc["_id"]
    return doc

# ============================================================================
# API Routes - Health Check
# ============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": get_current_time(),
        "version": "1.0.0",
        "database": "connected" if client else "disconnected"
    }

# ============================================================================
# API Routes - Dashboard
# ============================================================================

@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics"""
    total_pipelines = pipelines_collection.count_documents({})
    active_pipelines = pipeline_runs_collection.count_documents({"status": "running"})
    total_experiments = experiments_collection.count_documents({})
    successful_runs = pipeline_runs_collection.count_documents({"status": "success"})
    failed_runs = pipeline_runs_collection.count_documents({"status": "failed"})
    total_models = models_collection.count_documents({})
    validations_passed = data_validations_collection.count_documents({"status": "passed"})
    validations_failed = data_validations_collection.count_documents({"status": "failed"})
    
    return DashboardStats(
        total_pipelines=total_pipelines,
        active_pipelines=active_pipelines,
        total_experiments=total_experiments,
        successful_runs_24h=successful_runs,
        failed_runs_24h=failed_runs,
        total_models=total_models,
        data_validations_passed=validations_passed,
        data_validations_failed=validations_failed
    )

@app.get("/api/dashboard/recent-runs")
async def get_recent_runs(limit: int = Query(default=10, le=50)):
    """Get recent pipeline runs"""
    runs = list(pipeline_runs_collection.find().sort("started_at", -1).limit(limit))
    return [serialize_doc(run) for run in runs]

@app.get("/api/dashboard/metrics")
async def get_dashboard_metrics():
    """Get metrics for dashboard charts"""
    # Generate sample metrics data for charts
    metrics = {
        "pipeline_runs": [
            {"date": "Mon", "success": 12, "failed": 2},
            {"date": "Tue", "success": 15, "failed": 1},
            {"date": "Wed", "success": 8, "failed": 3},
            {"date": "Thu", "success": 18, "failed": 0},
            {"date": "Fri", "success": 14, "failed": 2},
            {"date": "Sat", "success": 6, "failed": 1},
            {"date": "Sun", "success": 4, "failed": 0},
        ],
        "model_accuracy": [
            {"version": "v1.0", "accuracy": 0.82},
            {"version": "v1.1", "accuracy": 0.85},
            {"version": "v1.2", "accuracy": 0.87},
            {"version": "v1.3", "accuracy": 0.89},
            {"version": "v1.4", "accuracy": 0.91},
        ],
        "data_quality": {
            "completeness": 98.5,
            "accuracy": 97.2,
            "consistency": 99.1,
            "timeliness": 95.8
        }
    }
    return metrics

# ============================================================================
# API Routes - Pipelines
# ============================================================================

@app.get("/api/pipelines", response_model=List[PipelineResponse])
async def list_pipelines():
    """List all pipelines"""
    pipelines = list(pipelines_collection.find())
    return [serialize_doc(p) for p in pipelines]

@app.post("/api/pipelines", response_model=PipelineResponse)
async def create_pipeline(pipeline: PipelineCreate):
    """Create a new pipeline"""
    now = get_current_time()
    pipeline_doc = {
        "id": generate_id(),
        "name": pipeline.name,
        "description": pipeline.description or "",
        "steps": [step.dict() for step in pipeline.steps],
        "status": PipelineStatus.IDLE,
        "schedule": pipeline.schedule,
        "last_run": None,
        "created_at": now,
        "updated_at": now,
        "run_count": 0
    }
    pipelines_collection.insert_one(pipeline_doc)
    return serialize_doc(pipeline_doc)

@app.get("/api/pipelines/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str):
    """Get pipeline by ID"""
    pipeline = pipelines_collection.find_one({"id": pipeline_id})
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return serialize_doc(pipeline)

@app.delete("/api/pipelines/{pipeline_id}")
async def delete_pipeline(pipeline_id: str):
    """Delete a pipeline"""
    result = pipelines_collection.delete_one({"id": pipeline_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return {"message": "Pipeline deleted successfully"}

@app.post("/api/pipelines/{pipeline_id}/run")
async def run_pipeline(pipeline_id: str):
    """Trigger a pipeline run"""
    pipeline = pipelines_collection.find_one({"id": pipeline_id})
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    now = get_current_time()
    run_doc = {
        "id": generate_id(),
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline["name"],
        "status": PipelineStatus.RUNNING,
        "started_at": now,
        "finished_at": None,
        "duration_seconds": None,
        "steps_completed": 0,
        "total_steps": len(pipeline.get("steps", [])),
        "logs": [f"[{now}] Pipeline run started"],
        "error": None
    }
    pipeline_runs_collection.insert_one(run_doc)
    
    # Update pipeline status
    pipelines_collection.update_one(
        {"id": pipeline_id},
        {"$set": {"status": PipelineStatus.RUNNING, "updated_at": now}}
    )
    
    return serialize_doc(run_doc)

@app.get("/api/pipelines/{pipeline_id}/runs")
async def get_pipeline_runs(pipeline_id: str, limit: int = Query(default=20, le=100)):
    """Get runs for a specific pipeline"""
    runs = list(pipeline_runs_collection.find({"pipeline_id": pipeline_id}).sort("started_at", -1).limit(limit))
    return [serialize_doc(run) for run in runs]

# ============================================================================
# API Routes - Pipeline Runs
# ============================================================================

@app.get("/api/runs", response_model=List[PipelineRunResponse])
async def list_runs(limit: int = Query(default=50, le=200)):
    """List all pipeline runs"""
    runs = list(pipeline_runs_collection.find().sort("started_at", -1).limit(limit))
    return [serialize_doc(run) for run in runs]

@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get a specific run by ID"""
    run = pipeline_runs_collection.find_one({"id": run_id})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return serialize_doc(run)

@app.post("/api/runs/{run_id}/complete")
async def complete_run(run_id: str, success: bool = True, error: Optional[str] = None):
    """Mark a run as complete"""
    run = pipeline_runs_collection.find_one({"id": run_id})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    now = get_current_time()
    started = datetime.fromisoformat(run["started_at"].replace("Z", "+00:00"))
    finished = datetime.now(timezone.utc)
    duration = (finished - started).total_seconds()
    
    status = PipelineStatus.SUCCESS if success else PipelineStatus.FAILED
    
    pipeline_runs_collection.update_one(
        {"id": run_id},
        {"$set": {
            "status": status,
            "finished_at": now,
            "duration_seconds": duration,
            "steps_completed": run["total_steps"] if success else run["steps_completed"],
            "error": error,
            "logs": run["logs"] + [f"[{now}] Pipeline run {'completed successfully' if success else 'failed'}"]
        }}
    )
    
    # Update pipeline status and run count
    pipelines_collection.update_one(
        {"id": run["pipeline_id"]},
        {"$set": {"status": status, "last_run": now, "updated_at": now}, "$inc": {"run_count": 1}}
    )
    
    return {"message": f"Run marked as {status}"}

# ============================================================================
# API Routes - Experiments
# ============================================================================

@app.get("/api/experiments", response_model=List[ExperimentResponse])
async def list_experiments():
    """List all experiments"""
    experiments = list(experiments_collection.find().sort("created_at", -1))
    return [serialize_doc(exp) for exp in experiments]

@app.post("/api/experiments", response_model=ExperimentResponse)
async def create_experiment(experiment: ExperimentCreate):
    """Create a new experiment"""
    now = get_current_time()
    exp_doc = {
        "id": generate_id(),
        "name": experiment.name,
        "description": experiment.description or "",
        "pipeline_id": experiment.pipeline_id,
        "status": "running",
        "parameters": experiment.parameters,
        "metrics": {},
        "model_version": None,
        "created_at": now,
        "finished_at": None
    }
    experiments_collection.insert_one(exp_doc)
    return serialize_doc(exp_doc)

@app.get("/api/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment by ID"""
    exp = experiments_collection.find_one({"id": experiment_id})
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return serialize_doc(exp)

@app.put("/api/experiments/{experiment_id}/metrics")
async def update_experiment_metrics(experiment_id: str, metrics: Dict[str, Any]):
    """Update experiment metrics"""
    exp = experiments_collection.find_one({"id": experiment_id})
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiments_collection.update_one(
        {"id": experiment_id},
        {"$set": {"metrics": metrics}}
    )
    return {"message": "Metrics updated successfully"}

@app.post("/api/experiments/{experiment_id}/complete")
async def complete_experiment(experiment_id: str, metrics: Dict[str, Any] = {}):
    """Mark experiment as complete"""
    exp = experiments_collection.find_one({"id": experiment_id})
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    now = get_current_time()
    model_version = f"v{len(list(models_collection.find({'experiment_id': experiment_id}))) + 1}.0"
    
    experiments_collection.update_one(
        {"id": experiment_id},
        {"$set": {
            "status": "completed",
            "metrics": {**exp.get("metrics", {}), **metrics},
            "model_version": model_version,
            "finished_at": now
        }}
    )
    return {"message": "Experiment completed", "model_version": model_version}

@app.delete("/api/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str):
    """Delete an experiment"""
    result = experiments_collection.delete_one({"id": experiment_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"message": "Experiment deleted successfully"}

# ============================================================================
# API Routes - Models
# ============================================================================

@app.get("/api/models", response_model=List[ModelResponse])
async def list_models():
    """List all models"""
    models = list(models_collection.find().sort("created_at", -1))
    return [serialize_doc(m) for m in models]

@app.post("/api/models")
async def create_model(
    name: str,
    experiment_id: str,
    algorithm: str,
    metrics: Dict[str, Any] = {},
    parameters: Dict[str, Any] = {}
):
    """Register a new model"""
    # Get version number
    existing = list(models_collection.find({"name": name}))
    version = f"v{len(existing) + 1}.0"
    
    now = get_current_time()
    model_doc = {
        "id": generate_id(),
        "name": name,
        "version": version,
        "experiment_id": experiment_id,
        "algorithm": algorithm,
        "metrics": metrics,
        "parameters": parameters,
        "file_path": f"/models/{name}/{version}/model.pkl",
        "created_at": now,
        "status": "registered"
    }
    models_collection.insert_one(model_doc)
    return serialize_doc(model_doc)

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get model by ID"""
    model = models_collection.find_one({"id": model_id})
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return serialize_doc(model)

# ============================================================================
# API Routes - Data Validation
# ============================================================================

@app.get("/api/validations", response_model=List[DataValidationResponse])
async def list_validations():
    """List all data validations"""
    validations = list(data_validations_collection.find().sort("created_at", -1))
    return [serialize_doc(v) for v in validations]

@app.post("/api/validations", response_model=DataValidationResponse)
async def create_validation(validation: DataValidationCreate):
    """Create a new data validation"""
    now = get_current_time()
    
    # Simulate validation results
    rules_total = len(validation.rules) if validation.rules else random.randint(5, 15)
    rules_passed = random.randint(int(rules_total * 0.7), rules_total)
    rules_failed = rules_total - rules_passed
    
    issues = []
    if rules_failed > 0:
        issue_types = ["Missing values", "Type mismatch", "Out of range", "Duplicate records", "Invalid format"]
        for i in range(rules_failed):
            issues.append({
                "rule": f"Rule_{i+1}",
                "type": random.choice(issue_types),
                "severity": random.choice(["low", "medium", "high"]),
                "affected_rows": random.randint(1, 100),
                "description": f"Validation issue detected in dataset"
            })
    
    validation_doc = {
        "id": generate_id(),
        "name": validation.name,
        "dataset_path": validation.dataset_path,
        "status": "passed" if rules_failed == 0 else "failed",
        "rules_passed": rules_passed,
        "rules_failed": rules_failed,
        "total_rules": rules_total,
        "issues": issues,
        "profile": {
            "total_rows": random.randint(10000, 100000),
            "total_columns": random.randint(10, 50),
            "missing_cells": random.randint(0, 500),
            "duplicate_rows": random.randint(0, 100),
            "memory_size_mb": round(random.uniform(10, 500), 2)
        },
        "created_at": now,
        "finished_at": now
    }
    data_validations_collection.insert_one(validation_doc)
    return serialize_doc(validation_doc)

@app.get("/api/validations/{validation_id}")
async def get_validation(validation_id: str):
    """Get validation by ID"""
    validation = data_validations_collection.find_one({"id": validation_id})
    if not validation:
        raise HTTPException(status_code=404, detail="Validation not found")
    return serialize_doc(validation)

# ============================================================================
# API Routes - Logs
# ============================================================================

@app.get("/api/logs")
async def list_logs(
    limit: int = Query(default=100, le=500),
    level: Optional[str] = None,
    source: Optional[str] = None
):
    """List logs with optional filtering"""
    query = {}
    if level:
        query["level"] = level
    if source:
        query["source"] = source
    
    logs = list(logs_collection.find(query).sort("timestamp", -1).limit(limit))
    return [serialize_doc(log) for log in logs]

@app.post("/api/logs")
async def create_log(level: str, source: str, message: str, metadata: Dict[str, Any] = {}):
    """Create a new log entry"""
    log_doc = {
        "id": generate_id(),
        "timestamp": get_current_time(),
        "level": level,
        "source": source,
        "message": message,
        "metadata": metadata
    }
    logs_collection.insert_one(log_doc)
    return serialize_doc(log_doc)

# ============================================================================
# API Routes - Seed Data
# ============================================================================

@app.post("/api/seed")
async def seed_data():
    """Seed the database with sample data"""
    now = get_current_time()
    
    # Clear existing data
    pipelines_collection.delete_many({})
    pipeline_runs_collection.delete_many({})
    experiments_collection.delete_many({})
    models_collection.delete_many({})
    data_validations_collection.delete_many({})
    logs_collection.delete_many({})
    
    # Seed pipelines
    pipelines = [
        {
            "id": "pip-001",
            "name": "HMP Data ETL Pipeline",
            "description": "Extract, transform and load HMP accelerometer sensor data",
            "steps": [
                {"id": "step-1", "name": "Extract CSV Data", "type": "extract", "config": {"source": "data/data.csv"}, "position": {"x": 100, "y": 100}},
                {"id": "step-2", "name": "Convert to Parquet", "type": "transform", "config": {"format": "parquet"}, "position": {"x": 300, "y": 100}},
                {"id": "step-3", "name": "Validate Data", "type": "validate", "config": {"rules": ["not_null", "type_check"]}, "position": {"x": 500, "y": 100}},
                {"id": "step-4", "name": "Load to Database", "type": "load", "config": {"target": "mongodb"}, "position": {"x": 700, "y": 100}},
            ],
            "status": "success",
            "schedule": "0 */6 * * *",
            "last_run": now,
            "created_at": now,
            "updated_at": now,
            "run_count": 45
        },
        {
            "id": "pip-002",
            "name": "ML Training Pipeline",
            "description": "Train Random Forest classifier for activity recognition",
            "steps": [
                {"id": "step-1", "name": "Load Parquet Data", "type": "extract", "config": {"source": "data/data.parquet"}, "position": {"x": 100, "y": 100}},
                {"id": "step-2", "name": "Feature Engineering", "type": "transform", "config": {"features": ["x", "y", "z"]}, "position": {"x": 300, "y": 100}},
                {"id": "step-3", "name": "Train Model", "type": "train", "config": {"algorithm": "RandomForest"}, "position": {"x": 500, "y": 100}},
                {"id": "step-4", "name": "Export PMML", "type": "load", "config": {"format": "pmml"}, "position": {"x": 700, "y": 100}},
            ],
            "status": "idle",
            "schedule": None,
            "last_run": now,
            "created_at": now,
            "updated_at": now,
            "run_count": 12
        },
        {
            "id": "pip-003",
            "name": "Data Quality Check",
            "description": "Daily data quality validation pipeline",
            "steps": [
                {"id": "step-1", "name": "Load Data", "type": "extract", "config": {}, "position": {"x": 100, "y": 100}},
                {"id": "step-2", "name": "Check Completeness", "type": "validate", "config": {}, "position": {"x": 300, "y": 100}},
                {"id": "step-3", "name": "Check Consistency", "type": "validate", "config": {}, "position": {"x": 500, "y": 100}},
            ],
            "status": "failed",
            "schedule": "0 0 * * *",
            "last_run": now,
            "created_at": now,
            "updated_at": now,
            "run_count": 30
        }
    ]
    
    for pipeline in pipelines:
        pipelines_collection.insert_one(pipeline)
    
    # Seed pipeline runs
    statuses = ["success", "success", "success", "success", "failed", "success", "running"]
    for i in range(15):
        pipeline = random.choice(pipelines)
        status = random.choice(statuses)
        run = {
            "id": f"run-{i+1:03d}",
            "pipeline_id": pipeline["id"],
            "pipeline_name": pipeline["name"],
            "status": status,
            "started_at": now,
            "finished_at": now if status != "running" else None,
            "duration_seconds": random.uniform(30, 300) if status != "running" else None,
            "steps_completed": len(pipeline["steps"]) if status == "success" else random.randint(0, len(pipeline["steps"])-1),
            "total_steps": len(pipeline["steps"]),
            "logs": [f"[{now}] Step {j+1} completed" for j in range(random.randint(1, len(pipeline["steps"])))],
            "error": "Connection timeout to data source" if status == "failed" else None
        }
        pipeline_runs_collection.insert_one(run)
    
    # Seed experiments
    algorithms = ["RandomForest", "GradientBoosting", "LogisticRegression", "SVM", "NeuralNetwork"]
    for i in range(10):
        exp = {
            "id": f"exp-{i+1:03d}",
            "name": f"Activity Recognition Experiment {i+1}",
            "description": f"Training {algorithms[i % len(algorithms)]} model for activity classification",
            "pipeline_id": "pip-002",
            "status": "completed" if i < 8 else "running",
            "parameters": {
                "algorithm": algorithms[i % len(algorithms)],
                "n_estimators": random.randint(50, 200),
                "max_depth": random.randint(5, 20),
                "learning_rate": round(random.uniform(0.01, 0.1), 3)
            },
            "metrics": {
                "accuracy": round(random.uniform(0.82, 0.95), 4),
                "precision": round(random.uniform(0.80, 0.94), 4),
                "recall": round(random.uniform(0.78, 0.93), 4),
                "f1_score": round(random.uniform(0.79, 0.94), 4),
                "training_time_seconds": round(random.uniform(60, 600), 2)
            } if i < 8 else {},
            "model_version": f"v{i+1}.0" if i < 8 else None,
            "created_at": now,
            "finished_at": now if i < 8 else None
        }
        experiments_collection.insert_one(exp)
        
        # Create model for completed experiments
        if i < 8:
            model = {
                "id": f"model-{i+1:03d}",
                "name": "activity_classifier",
                "version": f"v{i+1}.0",
                "experiment_id": exp["id"],
                "algorithm": algorithms[i % len(algorithms)],
                "metrics": exp["metrics"],
                "parameters": exp["parameters"],
                "file_path": f"/models/activity_classifier/v{i+1}.0/model.pkl",
                "created_at": now,
                "status": "registered"
            }
            models_collection.insert_one(model)
    
    # Seed data validations
    validation_names = ["Input Data Validation", "Training Data Validation", "Feature Data Validation"]
    for i in range(5):
        rules_total = random.randint(8, 15)
        rules_passed = random.randint(int(rules_total * 0.6), rules_total)
        rules_failed = rules_total - rules_passed
        
        validation = {
            "id": f"val-{i+1:03d}",
            "name": validation_names[i % len(validation_names)],
            "dataset_path": f"/data/dataset_{i+1}.parquet",
            "status": "passed" if rules_failed == 0 else "failed",
            "rules_passed": rules_passed,
            "rules_failed": rules_failed,
            "total_rules": rules_total,
            "issues": [
                {"rule": f"Rule_{j}", "type": random.choice(["Missing values", "Type mismatch", "Out of range"]), "severity": random.choice(["low", "medium", "high"]), "affected_rows": random.randint(1, 50), "description": "Validation issue"}
                for j in range(rules_failed)
            ],
            "profile": {
                "total_rows": random.randint(10000, 100000),
                "total_columns": random.randint(10, 50),
                "missing_cells": random.randint(0, 500),
                "duplicate_rows": random.randint(0, 100),
                "memory_size_mb": round(random.uniform(10, 500), 2)
            },
            "created_at": now,
            "finished_at": now
        }
        data_validations_collection.insert_one(validation)
    
    # Seed logs
    log_levels = ["INFO", "INFO", "INFO", "WARNING", "ERROR", "DEBUG"]
    sources = ["Pipeline", "Experiment", "Validation", "System"]
    messages = [
        "Pipeline execution started",
        "Data extraction completed",
        "Transform step finished",
        "Model training in progress",
        "Validation passed",
        "Connection established",
        "Memory usage at 75%",
        "Retrying failed operation",
        "Data validation completed",
        "Model exported successfully"
    ]
    
    for i in range(50):
        log = {
            "id": f"log-{i+1:03d}",
            "timestamp": now,
            "level": random.choice(log_levels),
            "source": random.choice(sources),
            "message": random.choice(messages),
            "metadata": {"run_id": f"run-{random.randint(1, 15):03d}"}
        }
        logs_collection.insert_one(log)
    
    return {
        "message": "Database seeded successfully",
        "pipelines": len(pipelines),
        "runs": 15,
        "experiments": 10,
        "models": 8,
        "validations": 5,
        "logs": 50
    }

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
