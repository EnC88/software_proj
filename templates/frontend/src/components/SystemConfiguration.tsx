
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DropdownMenu, DropdownMenuCheckboxItem, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Computer, Database, Server, Settings, ChevronDown } from 'lucide-react';

interface SystemConfig {
  operatingSystem: string;
  database: string;
  webServers: string[];
}

interface SystemConfigurationProps {
  config: SystemConfig;
  onConfigChange: (config: SystemConfig) => void;
}

const SystemConfiguration: React.FC<SystemConfigurationProps> = ({ config, onConfigChange }) => {
  const operatingSystems = [
    'Windows 10',
    'Windows 11',
    'Windows Server 2019',
    'Windows Server 2022',
    'Ubuntu 20.04 LTS',
    'Ubuntu 22.04 LTS',
    'Red Hat Enterprise Linux 8',
    'Red Hat Enterprise Linux 9',
    'CentOS 7',
    'CentOS 8',
    'macOS Monterey',
    'macOS Ventura',
    'macOS Sonoma'
  ];

  const databases = [
    'MySQL 5.7',
    'MySQL 8.0',
    'PostgreSQL 13',
    'PostgreSQL 14',
    'PostgreSQL 15',
    'Oracle Database 19c',
    'Oracle Database 21c',
    'Microsoft SQL Server 2019',
    'Microsoft SQL Server 2022',
    'MongoDB 5.0',
    'MongoDB 6.0',
    'Redis 6',
    'Redis 7'
  ];

  const webServers = [
    'Apache HTTP Server 2.4',
    'Nginx 1.20',
    'Nginx 1.22',
    'Microsoft IIS 10',
    'Tomcat 9',
    'Tomcat 10',
    'Node.js 16',
    'Node.js 18',
    'Node.js 20'
  ];

  const handleWebServerChange = (webServer: string, checked: boolean) => {
    const newConfig = {
      ...config,
      webServers: checked 
        ? [...config.webServers, webServer]
        : config.webServers.filter(ws => ws !== webServer)
    };
    onConfigChange(newConfig);
  };

  const handleOperatingSystemChange = (value: string) => {
    onConfigChange({
      ...config,
      operatingSystem: value
    });
  };

  const handleDatabaseChange = (value: string) => {
    onConfigChange({
      ...config,
      database: value
    });
  };

  return (
    <Card className="h-fit shadow-lg border border-gray-200/50">
      <CardHeader className="bg-gradient-to-r from-slate-900 to-slate-800 text-white">
        <CardTitle className="text-lg font-semibold flex items-center gap-3">
          <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center">
            <Settings className="w-4 h-4" />
          </div>
          System Configuration
        </CardTitle>
        <p className="text-slate-300 text-sm mt-1">
          Optional - helps provide targeted recommendations
        </p>
      </CardHeader>
      <CardContent className="p-6 space-y-6">
        {/* Operating System */}
        <div className="space-y-2">
          <Label className="text-sm font-medium text-gray-700 flex items-center gap-2">
            <Computer className="w-4 h-4 text-blue-600" />
            Operating System
          </Label>
          <Select 
            value={config.operatingSystem} 
            onValueChange={handleOperatingSystemChange}
          >
            <SelectTrigger className="w-full h-10 bg-white border-gray-200 hover:border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500">
              <SelectValue placeholder="Select operating system" />
            </SelectTrigger>
            <SelectContent className="bg-white border border-gray-200 shadow-lg z-50">
              {operatingSystems.map((os) => (
                <SelectItem key={os} value={os} className="hover:bg-gray-50">
                  {os}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Database */}
        <div className="space-y-2">
          <Label className="text-sm font-medium text-gray-700 flex items-center gap-2">
            <Database className="w-4 h-4 text-green-600" />
            Database
          </Label>
          <Select 
            value={config.database} 
            onValueChange={handleDatabaseChange}
          >
            <SelectTrigger className="w-full h-10 bg-white border-gray-200 hover:border-gray-300 focus:border-blue-500 focus:ring-1 focus:ring-blue-500">
              <SelectValue placeholder="Select database" />
            </SelectTrigger>
            <SelectContent className="bg-white border border-gray-200 shadow-lg z-50">
              {databases.map((db) => (
                <SelectItem key={db} value={db} className="hover:bg-gray-50">
                  {db}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Web Servers */}
        <div className="space-y-2">
          <Label className="text-sm font-medium text-gray-700 flex items-center gap-2">
            <Server className="w-4 h-4 text-orange-600" />
            Web Servers
          </Label>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                className="w-full h-10 justify-between bg-white border-gray-200 hover:border-gray-300 hover:bg-gray-50"
              >
                <span className="text-gray-700">
                  {config.webServers.length === 0 
                    ? "Select web servers" 
                    : `${config.webServers.length} selected`
                  }
                </span>
                <ChevronDown className="h-4 w-4 text-gray-500" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-80 bg-white border border-gray-200 shadow-lg z-50" align="start">
              {webServers.map((webServer) => (
                <DropdownMenuCheckboxItem
                  key={webServer}
                  checked={config.webServers.includes(webServer)}
                  onCheckedChange={(checked) => handleWebServerChange(webServer, checked)}
                  className="hover:bg-gray-50 cursor-pointer"
                >
                  {webServer}
                </DropdownMenuCheckboxItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {/* Selection Summary */}
        {(config.operatingSystem || config.database || config.webServers.length > 0) && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-6">
            <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              Current Configuration
            </h4>
            <div className="space-y-2 text-sm">
              {config.operatingSystem && (
                <div className="flex items-center gap-2 text-gray-600">
                  <Computer className="w-3 h-3" />
                  <span className="font-medium">OS:</span> {config.operatingSystem}
                </div>
              )}
              {config.database && (
                <div className="flex items-center gap-2 text-gray-600">
                  <Database className="w-3 h-3" />
                  <span className="font-medium">DB:</span> {config.database}
                </div>
              )}
              {config.webServers.length > 0 && (
                <div className="flex items-start gap-2 text-gray-600">
                  <Server className="w-3 h-3 mt-0.5" />
                  <div className="flex-1">
                    <span className="font-medium">Servers:</span>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {config.webServers.map((server) => (
                        <span key={server} className="inline-block bg-white px-2 py-0.5 rounded text-xs border border-gray-200">
                          {server}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SystemConfiguration;
