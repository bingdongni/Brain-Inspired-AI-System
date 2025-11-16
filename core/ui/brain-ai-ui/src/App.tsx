import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { SystemArchitecture } from './pages/SystemArchitecture';
import { TrainingInterface } from './pages/TrainingInterface';
import { Tutorials } from './pages/Tutorials';
import { BrainStateMonitor } from './components/BrainStateMonitor';
import { PerformanceDashboard } from './components/PerformanceDashboard';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/architecture" element={<SystemArchitecture />} />
            <Route path="/training" element={<TrainingInterface />} />
            <Route path="/tutorials" element={<Tutorials />} />
            <Route path="/monitor" element={<BrainStateMonitor />} />
            <Route path="/performance" element={<PerformanceDashboard />} />
          </Routes>
        </Layout>
      </div>
    </Router>
  );
}

export default App;
