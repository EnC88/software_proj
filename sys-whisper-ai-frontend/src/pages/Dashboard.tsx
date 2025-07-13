
import React from 'react';
import StatsOverview from '@/components/StatsOverview';

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">System Analytics Dashboard</h1>
          <p className="text-gray-600">Monitor system compatibility metrics and usage statistics</p>
        </div>
        
        <StatsOverview />
      </div>
    </div>
  );
};

export default Dashboard;
