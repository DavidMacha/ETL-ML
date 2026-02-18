import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import {
  LayoutDashboard,
  GitBranch,
  FlaskConical,
  ShieldCheck,
  ScrollText,
  Settings,
  Play,
  CheckCircle2,
  XCircle,
  Clock,
  Activity,
  Database,
  Cpu,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  ChevronRight,
  Loader2,
  Box,
  Trash2,
  Eye,
  AlertTriangle
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';

const API_URL = process.env.REACT_APP_BACKEND_URL || '';

// ============================================================================
// API Functions
// ============================================================================

const api = {
  async get(endpoint) {
    const res = await fetch(`${API_URL}${endpoint}`);
    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return res.json();
  },
  async post(endpoint, data = {}) {
    const res = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return res.json();
  },
  async delete(endpoint) {
    const res = await fetch(`${API_URL}${endpoint}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return res.json();
  }
};

// ============================================================================
// Sidebar Component
// ============================================================================

const Sidebar = ({ activePage, setActivePage }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'pipelines', label: 'Pipelines', icon: GitBranch },
    { id: 'experiments', label: 'Experiments', icon: FlaskConical },
    { id: 'validations', label: 'Data Quality', icon: ShieldCheck },
    { id: 'logs', label: 'Logs', icon: ScrollText },
    { id: 'settings', label: 'Settings', icon: Settings },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <Database size={20} color="#fafafa" />
        </div>
        <span className="sidebar-title">ETL & ML</span>
      </div>
      
      <nav className="sidebar-nav">
        {navItems.map(item => (
          <button
            key={item.id}
            data-testid={`nav-${item.id}`}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => setActivePage(item.id)}
          >
            <item.icon size={18} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
};

// ============================================================================
// Status Badge Component
// ============================================================================

const StatusBadge = ({ status }) => {
  const getStatusIcon = () => {
    switch (status) {
      case 'success': return <CheckCircle2 size={12} />;
      case 'failed': return <XCircle size={12} />;
      case 'running': return <Loader2 size={12} className="animate-spin" />;
      case 'pending': return <Clock size={12} />;
      default: return <Activity size={12} />;
    }
  };

  return (
    <span className={`status-badge ${status}`} data-testid={`status-badge-${status}`}>
      {getStatusIcon()}
      {status}
    </span>
  );
};

// ============================================================================
// Dashboard Page
// ============================================================================

const DashboardPage = () => {
  const [stats, setStats] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [recentRuns, setRecentRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [statsData, metricsData, runsData] = await Promise.all([
        api.get('/api/dashboard/stats'),
        api.get('/api/dashboard/metrics'),
        api.get('/api/dashboard/recent-runs?limit=5')
      ]);
      setStats(statsData);
      setMetrics(metricsData);
      setRecentRuns(runsData);
    } catch (err) {
      console.error('Failed to load dashboard:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const COLORS = ['#10b981', '#ef4444'];

  if (loading) {
    return (
      <div className="empty-state">
        <Loader2 size={48} className="animate-spin" />
        <p className="mt-4">Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 className="page-title">Dashboard</h1>
            <p className="page-description">Overview of your ETL pipelines and ML experiments</p>
          </div>
          <button className="btn btn-secondary" data-testid="refresh-dashboard" onClick={loadData}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>
      </div>

      <div className="dashboard-grid">
        {/* Stats Cards */}
        <div className="stat-card" data-testid="stat-total-pipelines">
          <div className="stat-label">Total Pipelines</div>
          <div className="stat-value">{stats?.total_pipelines || 0}</div>
          <div className="stat-change positive">
            <TrendingUp size={12} /> Active: {stats?.active_pipelines || 0}
          </div>
        </div>

        <div className="stat-card" data-testid="stat-experiments">
          <div className="stat-label">Experiments</div>
          <div className="stat-value">{stats?.total_experiments || 0}</div>
          <div className="stat-change positive">
            <FlaskConical size={12} /> Models: {stats?.total_models || 0}
          </div>
        </div>

        <div className="stat-card" data-testid="stat-success-runs">
          <div className="stat-label">Successful Runs</div>
          <div className="stat-value">{stats?.successful_runs_24h || 0}</div>
          <div className="stat-change positive">
            <TrendingUp size={12} /> Last 24h
          </div>
        </div>

        <div className="stat-card" data-testid="stat-failed-runs">
          <div className="stat-label">Failed Runs</div>
          <div className="stat-value">{stats?.failed_runs_24h || 0}</div>
          <div className="stat-change negative">
            <TrendingDown size={12} /> Last 24h
          </div>
        </div>

        {/* Pipeline Runs Chart */}
        <div className="chart-card" data-testid="chart-pipeline-runs">
          <h3 className="chart-title">Pipeline Runs (Last 7 Days)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={metrics?.pipeline_runs || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="date" stroke="#a1a1aa" fontSize={12} />
              <YAxis stroke="#a1a1aa" fontSize={12} />
              <Tooltip 
                contentStyle={{ background: '#121212', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px' }}
                labelStyle={{ color: '#fafafa' }}
              />
              <Bar dataKey="success" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="failed" fill="#ef4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Model Accuracy Chart */}
        <div className="chart-card" data-testid="chart-model-accuracy">
          <h3 className="chart-title">Model Accuracy Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={metrics?.model_accuracy || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
              <XAxis dataKey="version" stroke="#a1a1aa" fontSize={12} />
              <YAxis stroke="#a1a1aa" fontSize={12} domain={[0.8, 1]} />
              <Tooltip 
                contentStyle={{ background: '#121212', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px' }}
                labelStyle={{ color: '#fafafa' }}
                formatter={(value) => [(value * 100).toFixed(1) + '%', 'Accuracy']}
              />
              <Line type="monotone" dataKey="accuracy" stroke="#2563eb" strokeWidth={2} dot={{ fill: '#2563eb', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Runs */}
        <div className="list-card" data-testid="recent-runs-list">
          <div className="list-header">
            <h3 className="list-title">Recent Runs</h3>
            <ChevronRight size={16} className="text-muted-foreground" />
          </div>
          <div className="list-content">
            {recentRuns.length === 0 ? (
              <div className="empty-state">
                <p>No recent runs</p>
              </div>
            ) : (
              recentRuns.map(run => (
                <div key={run.id} className="list-item">
                  <StatusBadge status={run.status} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 500, fontSize: '0.875rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {run.pipeline_name}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                      {run.duration_seconds ? `${run.duration_seconds.toFixed(1)}s` : 'Running...'}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Data Quality */}
        <div className="list-card" data-testid="data-quality-card">
          <div className="list-header">
            <h3 className="list-title">Data Quality Metrics</h3>
          </div>
          <div className="metrics-grid">
            {metrics?.data_quality && Object.entries(metrics.data_quality).map(([key, value]) => (
              <div key={key} className="metric-item">
                <div className="metric-label">{key}</div>
                <div className="metric-value">{value}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Validation Status */}
        <div className="list-card" data-testid="validation-status-card">
          <div className="list-header">
            <h3 className="list-title">Validation Status</h3>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={[
                    { name: 'Passed', value: stats?.data_validations_passed || 0 },
                    { name: 'Failed', value: stats?.data_validations_failed || 0 }
                  ]}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={70}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {[0, 1].map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Pipelines Page
// ============================================================================

const PipelinesPage = () => {
  const [pipelines, setPipelines] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPipeline, setSelectedPipeline] = useState(null);
  const [runs, setRuns] = useState([]);

  const loadPipelines = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.get('/api/pipelines');
      setPipelines(data);
    } catch (err) {
      console.error('Failed to load pipelines:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  const loadPipelineRuns = useCallback(async (pipelineId) => {
    try {
      const data = await api.get(`/api/pipelines/${pipelineId}/runs`);
      setRuns(data);
    } catch (err) {
      console.error('Failed to load runs:', err);
    }
  }, []);

  useEffect(() => {
    loadPipelines();
  }, [loadPipelines]);

  useEffect(() => {
    if (selectedPipeline) {
      loadPipelineRuns(selectedPipeline.id);
    }
  }, [selectedPipeline, loadPipelineRuns]);

  const runPipeline = async (pipelineId) => {
    try {
      await api.post(`/api/pipelines/${pipelineId}/run`);
      loadPipelines();
      if (selectedPipeline?.id === pipelineId) {
        loadPipelineRuns(pipelineId);
      }
    } catch (err) {
      console.error('Failed to run pipeline:', err);
    }
  };

  const deletePipeline = async (pipelineId) => {
    try {
      await api.delete(`/api/pipelines/${pipelineId}`);
      loadPipelines();
      if (selectedPipeline?.id === pipelineId) {
        setSelectedPipeline(null);
      }
    } catch (err) {
      console.error('Failed to delete pipeline:', err);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Pipelines</h1>
        <p className="page-description">Manage and monitor your ETL pipelines</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: selectedPipeline ? '1fr 1fr' : '1fr', gap: '1.5rem' }}>
        <div className="table-container" data-testid="pipelines-table">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Steps</th>
                <th>Runs</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
                  </td>
                </tr>
              ) : pipelines.length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
                    No pipelines found
                  </td>
                </tr>
              ) : (
                pipelines.map(pipeline => (
                  <tr key={pipeline.id} style={{ cursor: 'pointer' }} onClick={() => setSelectedPipeline(pipeline)}>
                    <td>
                      <div style={{ fontWeight: 500 }}>{pipeline.name}</div>
                      <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>{pipeline.description}</div>
                    </td>
                    <td><StatusBadge status={pipeline.status} /></td>
                    <td>{pipeline.steps?.length || 0}</td>
                    <td>{pipeline.run_count}</td>
                    <td onClick={e => e.stopPropagation()}>
                      <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button 
                          className="btn-icon" 
                          data-testid={`run-pipeline-${pipeline.id}`}
                          onClick={() => runPipeline(pipeline.id)}
                          title="Run Pipeline"
                        >
                          <Play size={16} />
                        </button>
                        <button 
                          className="btn-icon" 
                          data-testid={`view-pipeline-${pipeline.id}`}
                          onClick={() => setSelectedPipeline(pipeline)}
                          title="View Details"
                        >
                          <Eye size={16} />
                        </button>
                        <button 
                          className="btn-icon" 
                          data-testid={`delete-pipeline-${pipeline.id}`}
                          onClick={() => deletePipeline(pipeline.id)}
                          title="Delete Pipeline"
                          style={{ color: '#ef4444' }}
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {selectedPipeline && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="table-container" style={{ padding: '1.5rem' }} data-testid="pipeline-details">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>{selectedPipeline.name}</h3>
                  <p style={{ fontSize: '0.875rem', color: '#a1a1aa' }}>{selectedPipeline.description}</p>
                </div>
                <StatusBadge status={selectedPipeline.status} />
              </div>
              
              <div style={{ marginTop: '1.5rem' }}>
                <h4 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '1rem' }}>Pipeline Steps</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {selectedPipeline.steps?.map((step, idx) => (
                    <div key={step.id} style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: '0.75rem',
                      padding: '0.75rem',
                      background: 'rgba(255,255,255,0.02)',
                      borderRadius: '6px',
                      borderLeft: '3px solid #2563eb'
                    }}>
                      <div style={{ 
                        width: '24px', 
                        height: '24px', 
                        borderRadius: '50%', 
                        background: '#2563eb', 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'center',
                        fontSize: '0.75rem',
                        fontWeight: 600
                      }}>
                        {idx + 1}
                      </div>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 500, fontSize: '0.875rem' }}>{step.name}</div>
                        <div style={{ fontSize: '0.75rem', color: '#a1a1aa', textTransform: 'capitalize' }}>{step.type}</div>
                      </div>
                      <span className={`status-badge ${step.type}`}>{step.type}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="table-container" data-testid="pipeline-runs">
              <div style={{ padding: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
                <h4 style={{ fontSize: '0.875rem', fontWeight: 600 }}>Run History</h4>
              </div>
              <div style={{ maxHeight: '300px', overflow: 'auto' }}>
                {runs.length === 0 ? (
                  <div style={{ padding: '2rem', textAlign: 'center', color: '#a1a1aa' }}>No runs yet</div>
                ) : (
                  runs.map(run => (
                    <div key={run.id} style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'space-between',
                      padding: '0.75rem 1rem',
                      borderBottom: '1px solid rgba(255,255,255,0.05)'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <StatusBadge status={run.status} />
                        <span style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                          {run.steps_completed}/{run.total_steps} steps
                        </span>
                      </div>
                      <span style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                        {run.duration_seconds ? `${run.duration_seconds.toFixed(1)}s` : 'Running...'}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Experiments Page
// ============================================================================

const ExperimentsPage = () => {
  const [experiments, setExperiments] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('experiments');

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [expData, modelData] = await Promise.all([
        api.get('/api/experiments'),
        api.get('/api/models')
      ]);
      setExperiments(expData);
      setModels(modelData);
    } catch (err) {
      console.error('Failed to load experiments:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const deleteExperiment = async (expId) => {
    try {
      await api.delete(`/api/experiments/${expId}`);
      loadData();
    } catch (err) {
      console.error('Failed to delete experiment:', err);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Experiments</h1>
        <p className="page-description">Track ML experiments and model versions</p>
      </div>

      <div className="tabs">
        <button 
          className={`tab ${activeTab === 'experiments' ? 'active' : ''}`}
          data-testid="tab-experiments"
          onClick={() => setActiveTab('experiments')}
        >
          Experiments
        </button>
        <button 
          className={`tab ${activeTab === 'models' ? 'active' : ''}`}
          data-testid="tab-models"
          onClick={() => setActiveTab('models')}
        >
          Models
        </button>
      </div>

      {activeTab === 'experiments' && (
        <div className="table-container" data-testid="experiments-table">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Algorithm</th>
                <th>Status</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>Model Version</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
                  </td>
                </tr>
              ) : experiments.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
                    No experiments found
                  </td>
                </tr>
              ) : (
                experiments.map(exp => (
                  <tr key={exp.id}>
                    <td>
                      <div style={{ fontWeight: 500 }}>{exp.name}</div>
                      <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>{exp.id}</div>
                    </td>
                    <td>{exp.parameters?.algorithm || '-'}</td>
                    <td><StatusBadge status={exp.status === 'completed' ? 'success' : exp.status} /></td>
                    <td>
                      {exp.metrics?.accuracy ? (
                        <span style={{ color: '#10b981', fontWeight: 500 }}>
                          {(exp.metrics.accuracy * 100).toFixed(2)}%
                        </span>
                      ) : '-'}
                    </td>
                    <td>
                      {exp.metrics?.f1_score ? (
                        <span>{(exp.metrics.f1_score * 100).toFixed(2)}%</span>
                      ) : '-'}
                    </td>
                    <td>
                      {exp.model_version ? (
                        <span className="status-badge success">{exp.model_version}</span>
                      ) : '-'}
                    </td>
                    <td>
                      <button 
                        className="btn-icon"
                        data-testid={`delete-experiment-${exp.id}`}
                        onClick={() => deleteExperiment(exp.id)}
                        style={{ color: '#ef4444' }}
                      >
                        <Trash2 size={16} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {activeTab === 'models' && (
        <div className="table-container" data-testid="models-table">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Version</th>
                <th>Algorithm</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
                  </td>
                </tr>
              ) : models.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
                    No models found
                  </td>
                </tr>
              ) : (
                models.map(model => (
                  <tr key={model.id}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <Box size={16} className="text-primary" />
                        <span style={{ fontWeight: 500 }}>{model.name}</span>
                      </div>
                    </td>
                    <td>
                      <span className="status-badge success">{model.version}</span>
                    </td>
                    <td>{model.algorithm}</td>
                    <td>
                      <span style={{ color: '#10b981', fontWeight: 500 }}>
                        {model.metrics?.accuracy ? (model.metrics.accuracy * 100).toFixed(2) + '%' : '-'}
                      </span>
                    </td>
                    <td>{model.metrics?.precision ? (model.metrics.precision * 100).toFixed(2) + '%' : '-'}</td>
                    <td>{model.metrics?.recall ? (model.metrics.recall * 100).toFixed(2) + '%' : '-'}</td>
                    <td><StatusBadge status="success" /></td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Data Quality / Validations Page
// ============================================================================

const ValidationsPage = () => {
  const [validations, setValidations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedValidation, setSelectedValidation] = useState(null);

  const loadValidations = useCallback(async () => {
    try {
      setLoading(true);
      const data = await api.get('/api/validations');
      setValidations(data);
    } catch (err) {
      console.error('Failed to load validations:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadValidations();
  }, [loadValidations]);

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Data Quality</h1>
        <p className="page-description">Monitor data validation results and quality metrics</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: selectedValidation ? '1fr 1fr' : '1fr', gap: '1.5rem' }}>
        <div className="table-container" data-testid="validations-table">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Dataset</th>
                <th>Status</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem' }}>
                    <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
                  </td>
                </tr>
              ) : validations.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
                    No validations found
                  </td>
                </tr>
              ) : (
                validations.map(val => (
                  <tr key={val.id} style={{ cursor: 'pointer' }} onClick={() => setSelectedValidation(val)}>
                    <td style={{ fontWeight: 500 }}>{val.name}</td>
                    <td style={{ fontSize: '0.75rem', color: '#a1a1aa', fontFamily: 'JetBrains Mono, monospace' }}>
                      {val.dataset_path}
                    </td>
                    <td><StatusBadge status={val.status === 'passed' ? 'success' : 'failed'} /></td>
                    <td style={{ color: '#10b981' }}>{val.rules_passed}</td>
                    <td style={{ color: val.rules_failed > 0 ? '#ef4444' : '#a1a1aa' }}>{val.rules_failed}</td>
                    <td onClick={e => e.stopPropagation()}>
                      <button 
                        className="btn-icon"
                        data-testid={`view-validation-${val.id}`}
                        onClick={() => setSelectedValidation(val)}
                      >
                        <Eye size={16} />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {selectedValidation && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <div className="table-container" style={{ padding: '1.5rem' }} data-testid="validation-details">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.5rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>{selectedValidation.name}</h3>
                  <p style={{ fontSize: '0.75rem', color: '#a1a1aa', fontFamily: 'JetBrains Mono' }}>
                    {selectedValidation.dataset_path}
                  </p>
                </div>
                <StatusBadge status={selectedValidation.status === 'passed' ? 'success' : 'failed'} />
              </div>

              <div className="metrics-grid">
                <div className="metric-item">
                  <div className="metric-label">Total Rows</div>
                  <div className="metric-value">{selectedValidation.profile?.total_rows?.toLocaleString()}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Columns</div>
                  <div className="metric-value">{selectedValidation.profile?.total_columns}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Missing Cells</div>
                  <div className="metric-value">{selectedValidation.profile?.missing_cells}</div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Size</div>
                  <div className="metric-value">{selectedValidation.profile?.memory_size_mb} MB</div>
                </div>
              </div>

              <div style={{ marginTop: '1.5rem' }}>
                <div style={{ marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.875rem' }}>Validation Progress</span>
                  <span style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                    {selectedValidation.rules_passed}/{selectedValidation.total_rules} rules passed
                  </span>
                </div>
                <div className="progress-bar">
                  <div 
                    className={`progress-fill ${selectedValidation.rules_failed > 0 ? 'error' : 'success'}`}
                    style={{ width: `${(selectedValidation.rules_passed / selectedValidation.total_rules) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {selectedValidation.issues?.length > 0 && (
              <div className="table-container" data-testid="validation-issues">
                <div style={{ padding: '1rem', borderBottom: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertTriangle size={16} className="text-warning" />
                  <h4 style={{ fontSize: '0.875rem', fontWeight: 600 }}>Issues ({selectedValidation.issues.length})</h4>
                </div>
                <div style={{ maxHeight: '250px', overflow: 'auto' }}>
                  {selectedValidation.issues.map((issue, idx) => (
                    <div key={idx} style={{ 
                      padding: '0.75rem 1rem',
                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '0.75rem'
                    }}>
                      <span className={`status-badge ${issue.severity === 'high' ? 'failed' : issue.severity === 'medium' ? 'pending' : 'idle'}`}>
                        {issue.severity}
                      </span>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 500, fontSize: '0.875rem' }}>{issue.type}</div>
                        <div style={{ fontSize: '0.75rem', color: '#a1a1aa' }}>
                          {issue.affected_rows} affected rows
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Logs Page
// ============================================================================

const LogsPage = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');

  const loadLogs = useCallback(async () => {
    try {
      setLoading(true);
      const endpoint = filter === 'all' ? '/api/logs' : `/api/logs?level=${filter.toUpperCase()}`;
      const data = await api.get(endpoint);
      setLogs(data);
    } catch (err) {
      console.error('Failed to load logs:', err);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    loadLogs();
  }, [loadLogs]);

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 className="page-title">Logs</h1>
            <p className="page-description">System logs and activity history</p>
          </div>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {['all', 'info', 'warning', 'error'].map(level => (
              <button
                key={level}
                className={`btn ${filter === level ? 'btn-primary' : 'btn-secondary'}`}
                data-testid={`filter-${level}`}
                onClick={() => setFilter(level)}
              >
                {level.charAt(0).toUpperCase() + level.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="log-viewer" data-testid="log-viewer">
        {loading ? (
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <Loader2 className="animate-spin" size={24} style={{ margin: '0 auto' }} />
          </div>
        ) : logs.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: '#a1a1aa' }}>
            No logs found
          </div>
        ) : (
          logs.map(log => (
            <div key={log.id} className="log-entry">
              <span className="log-timestamp">
                {new Date(log.timestamp).toLocaleTimeString()}
              </span>
              <span className={`log-level ${log.level.toLowerCase()}`}>
                [{log.level}]
              </span>
              <span className="log-source">{log.source}</span>
              <span className="log-message">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Settings Page
// ============================================================================

const SettingsPage = () => {
  const [seeding, setSeeding] = useState(false);
  const [seedResult, setSeedResult] = useState(null);

  const seedDatabase = async () => {
    try {
      setSeeding(true);
      const result = await api.post('/api/seed');
      setSeedResult(result);
    } catch (err) {
      console.error('Failed to seed database:', err);
      setSeedResult({ error: err.message });
    } finally {
      setSeeding(false);
    }
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Settings</h1>
        <p className="page-description">Application configuration and administration</p>
      </div>

      <div style={{ display: 'grid', gap: '1.5rem', maxWidth: '600px' }}>
        <div className="table-container" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>Database</h3>
          <p style={{ fontSize: '0.875rem', color: '#a1a1aa', marginBottom: '1rem' }}>
            Seed the database with sample data for testing and demonstration purposes.
          </p>
          <button 
            className="btn btn-primary"
            data-testid="seed-database-btn"
            onClick={seedDatabase}
            disabled={seeding}
          >
            {seeding ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Seeding...
              </>
            ) : (
              <>
                <Database size={16} />
                Seed Database
              </>
            )}
          </button>
          
          {seedResult && (
            <div style={{ 
              marginTop: '1rem', 
              padding: '1rem', 
              background: seedResult.error ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)', 
              borderRadius: '6px',
              fontSize: '0.875rem'
            }}>
              {seedResult.error ? (
                <span style={{ color: '#ef4444' }}>Error: {seedResult.error}</span>
              ) : (
                <div>
                  <div style={{ color: '#10b981', fontWeight: 500, marginBottom: '0.5rem' }}>
                    Database seeded successfully!
                  </div>
                  <div style={{ color: '#a1a1aa', fontSize: '0.75rem' }}>
                    Created: {seedResult.pipelines} pipelines, {seedResult.runs} runs, {seedResult.experiments} experiments, {seedResult.models} models
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="table-container" style={{ padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '0.5rem' }}>About</h3>
          <div style={{ fontSize: '0.875rem', color: '#a1a1aa' }}>
            <p style={{ marginBottom: '0.5rem' }}>ETL & ML Dashboard v1.0.0</p>
            <p>A comprehensive platform for managing ETL pipelines, tracking ML experiments, and monitoring data quality.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Main App Component
// ============================================================================

function App() {
  const [activePage, setActivePage] = useState('dashboard');

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return <DashboardPage />;
      case 'pipelines':
        return <PipelinesPage />;
      case 'experiments':
        return <ExperimentsPage />;
      case 'validations':
        return <ValidationsPage />;
      case 'logs':
        return <LogsPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <DashboardPage />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar activePage={activePage} setActivePage={setActivePage} />
      <main className="main-content" data-testid="main-content">
        {renderPage()}
      </main>
    </div>
  );
}

export default App;
