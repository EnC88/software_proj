
import React, { useState } from 'react';
import ChatInterface from "@/components/ChatInterface";
import SystemConfiguration from "@/components/SystemConfiguration";
import StatsOverview from "@/components/StatsOverview";

interface SystemConfig {
  operatingSystem: string;
  database: string;
  webServers: string[];
}

const Index = () => {
  const [systemConfig, setSystemConfig] = useState<SystemConfig>({
    operatingSystem: '',
    database: '',
    webServers: []
  });

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-6 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-light text-gray-900 mb-4 tracking-tight">
            System Compatibility Assistant
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Enterprise-grade system compatibility analysis and recommendations
          </p>
        </div>
        
        {/* Statistics Overview */}
        <div className="mb-8">
          <StatsOverview />
        </div>

        {/* Main Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 max-w-6xl mx-auto">
          <div className="lg:col-span-2">
            <SystemConfiguration 
              config={systemConfig}
              onConfigChange={setSystemConfig}
            />
          </div>
          <div className="lg:col-span-3">
            <ChatInterface systemConfig={systemConfig} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
