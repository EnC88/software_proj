
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DropdownMenu, DropdownMenuCheckboxItem, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Computer, Database, Server, User, ChevronDown, Save, Loader2 } from 'lucide-react';

interface SystemConfig {
  operatingSystem: string;
  database: string;
  webServers: string[];
  useInChat: {
    os: boolean;
    database: boolean;
    webServers: boolean;
  };
}

const Profile = () => {
  const [config, setConfig] = useState<SystemConfig>({
    operatingSystem: '',
    database: '',
    webServers: [],
    useInChat: {
      os: true,
      database: true,
      webServers: true
    }
  });

  const [operatingSystems, setOperatingSystems] = useState<string[]>([]);
  const [databases, setDatabases] = useState<string[]>([]);
  const [webServers, setWebServers] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch options from API
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        setLoading(true);
        setError(null);

        // Fetch all options in parallel
        const [osResponse, dbResponse, wsResponse] = await Promise.all([
          fetch('/api/options/operating-systems'),
          fetch('/api/options/databases'),
          fetch('/api/options/web-servers')
        ]);

        if (!osResponse.ok || !dbResponse.ok || !wsResponse.ok) {
          throw new Error('Failed to fetch options from API');
        }

        const [osData, dbData, wsData] = await Promise.all([
          osResponse.json(),
          dbResponse.json(),
          wsResponse.json()
        ]);

        setOperatingSystems(osData.operating_systems || []);
        setDatabases(dbData.databases || []);
        setWebServers(wsData.web_servers || []);
      } catch (err) {
        console.error('Error fetching options:', err);
        setError('Failed to load configuration options. Please try again.');
        
        // Fallback to hardcoded options if API fails
        setOperatingSystems([
          'Windows 10', 'Windows 11', 'Windows Server 2019', 'Windows Server 2022',
          'Ubuntu 20.04 LTS', 'Ubuntu 22.04 LTS', 'Red Hat Enterprise Linux 8',
          'Red Hat Enterprise Linux 9', 'CentOS 7', 'CentOS 8',
          'macOS Monterey', 'macOS Ventura', 'macOS Sonoma'
        ]);
        setDatabases([
          'MySQL 5.7', 'MySQL 8.0', 'PostgreSQL 13', 'PostgreSQL 14', 'PostgreSQL 15',
          'Oracle Database 19c', 'Oracle Database 21c', 'Microsoft SQL Server 2019',
          'Microsoft SQL Server 2022', 'MongoDB 5.0', 'MongoDB 6.0', 'Redis 6', 'Redis 7'
        ]);
        setWebServers([
          'Apache HTTP Server 2.4', 'Nginx 1.20', 'Nginx 1.22', 'Microsoft IIS 10',
          'Tomcat 9', 'Tomcat 10', 'Node.js 16', 'Node.js 18', 'Node.js 20'
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchOptions();
  }, []);

  const handleWebServerChange = (webServer: string, checked: boolean) => {
    setConfig(prev => ({
      ...prev,
      webServers: checked 
        ? [...prev.webServers, webServer]
        : prev.webServers.filter(ws => ws !== webServer)
    }));
  };

  const handleSave = () => {
    // Save configuration logic here
    console.log('Saving configuration:', config);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-center h-64">
            <div className="flex items-center gap-3 text-gray-600">
              <Loader2 className="w-6 h-6 animate-spin" />
              <span>Loading configuration options...</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Profile Settings</h1>
          <p className="text-gray-600">Configure your system preferences for personalized chat recommendations</p>
          {error && (
            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-yellow-800 text-sm">{error}</p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* System Configuration */}
          <Card className="shadow-lg border border-gray-200/50">
            <CardHeader className="bg-gradient-to-r from-slate-900 to-slate-800 text-white">
              <CardTitle className="text-lg font-semibold flex items-center gap-3">
                <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center">
                  <Computer className="w-4 h-4" />
                </div>
                System Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6 space-y-6">
              {/* Operating System */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium text-gray-700 flex items-center gap-2">
                    <Computer className="w-4 h-4 text-blue-600" />
                    Operating System
                  </Label>
                  <div className="flex items-center gap-2">
                    <Label htmlFor="os-toggle" className="text-sm text-gray-600">Use in chat</Label>
                    <Switch
                      id="os-toggle"
                      checked={config.useInChat.os}
                      onCheckedChange={(checked) => setConfig(prev => ({
                        ...prev,
                        useInChat: { ...prev.useInChat, os: checked }
                      }))}
                    />
                  </div>
                </div>
                <Select 
                  value={config.operatingSystem} 
                  onValueChange={(value) => setConfig(prev => ({ ...prev, operatingSystem: value }))}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select operating system" />
                  </SelectTrigger>
                  <SelectContent>
                    {operatingSystems.map((os) => (
                      <SelectItem key={os} value={os}>{os}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Database */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium text-gray-700 flex items-center gap-2">
                    <Database className="w-4 h-4 text-green-600" />
                    Database
                  </Label>
                  <div className="flex items-center gap-2">
                    <Label htmlFor="db-toggle" className="text-sm text-gray-600">Use in chat</Label>
                    <Switch
                      id="db-toggle"
                      checked={config.useInChat.database}
                      onCheckedChange={(checked) => setConfig(prev => ({
                        ...prev,
                        useInChat: { ...prev.useInChat, database: checked }
                      }))}
                    />
                  </div>
                </div>
                <Select 
                  value={config.database} 
                  onValueChange={(value) => setConfig(prev => ({ ...prev, database: value }))}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select database" />
                  </SelectTrigger>
                  <SelectContent>
                    {databases.map((db) => (
                      <SelectItem key={db} value={db}>{db}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Web Servers */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label className="text-sm font-medium text-gray-700 flex items-center gap-2">
                    <Server className="w-4 h-4 text-orange-600" />
                    Web Servers
                  </Label>
                  <div className="flex items-center gap-2">
                    <Label htmlFor="ws-toggle" className="text-sm text-gray-600">Use in chat</Label>
                    <Switch
                      id="ws-toggle"
                      checked={config.useInChat.webServers}
                      onCheckedChange={(checked) => setConfig(prev => ({
                        ...prev,
                        useInChat: { ...prev.useInChat, webServers: checked }
                      }))}
                    />
                  </div>
                </div>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" className="w-full justify-between">
                      <span>
                        {config.webServers.length === 0 
                          ? "Select web servers" 
                          : `${config.webServers.length} selected`
                        }
                      </span>
                      <ChevronDown className="h-4 w-4" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-80">
                    {webServers.map((webServer) => (
                      <DropdownMenuCheckboxItem
                        key={webServer}
                        checked={config.webServers.includes(webServer)}
                        onCheckedChange={(checked) => handleWebServerChange(webServer, checked)}
                      >
                        {webServer}
                      </DropdownMenuCheckboxItem>
                    ))}
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>

              <Button onClick={handleSave} className="w-full bg-slate-800 hover:bg-slate-700">
                <Save className="w-4 h-4 mr-2" />
                Save Configuration
              </Button>
            </CardContent>
          </Card>

          {/* Configuration Preview */}
          <Card className="shadow-lg border border-gray-200/50">
            <CardHeader className="bg-gradient-to-r from-blue-600 to-blue-700 text-white">
              <CardTitle className="text-lg font-semibold flex items-center gap-3">
                <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center">
                  <User className="w-4 h-4" />
                </div>
                Configuration Preview
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="text-sm text-gray-600 mb-4">
                  This configuration will be used to provide personalized recommendations in your chats.
                </div>

                {config.operatingSystem && config.useInChat.os && (
                  <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
                    <Computer className="w-4 h-4 text-blue-600" />
                    <div>
                      <div className="font-medium text-blue-900">Operating System</div>
                      <div className="text-sm text-blue-700">{config.operatingSystem}</div>
                    </div>
                  </div>
                )}

                {config.database && config.useInChat.database && (
                  <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
                    <Database className="w-4 h-4 text-green-600" />
                    <div>
                      <div className="font-medium text-green-900">Database</div>
                      <div className="text-sm text-green-700">{config.database}</div>
                    </div>
                  </div>
                )}

                {config.webServers.length > 0 && config.useInChat.webServers && (
                  <div className="flex items-start gap-3 p-3 bg-orange-50 rounded-lg">
                    <Server className="w-4 h-4 text-orange-600 mt-0.5" />
                    <div className="flex-1">
                      <div className="font-medium text-orange-900">Web Servers</div>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {config.webServers.map((server) => (
                          <span key={server} className="inline-block bg-white px-2 py-1 rounded text-xs border border-orange-200 text-orange-700">
                            {server}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {(!config.operatingSystem && !config.database && config.webServers.length === 0) && (
                  <div className="text-center py-8 text-gray-500">
                    <Computer className="w-12 h-12 mx-auto mb-3 text-gray-300" />
                    <p>Configure your system preferences above to see them here</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Profile;
