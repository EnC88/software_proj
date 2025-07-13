
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, ResponsiveContainer, XAxis, YAxis, CartesianGrid } from 'recharts';
import { TrendingUp, TrendingDown, Users, Database, Server, Activity } from 'lucide-react';

// Sample data for charts
const compatibilityData = [
  { name: 'Windows', value: 85, color: '#3b82f6' },
  { name: 'Linux', value: 92, color: '#10b981' },
  { name: 'macOS', value: 78, color: '#f59e0b' },
];

const monthlyQueries = [
  { month: 'Jan', queries: 1200, resolved: 1050 },
  { month: 'Feb', queries: 1450, resolved: 1320 },
  { month: 'Mar', queries: 1680, resolved: 1580 },
  { month: 'Apr', queries: 1820, resolved: 1750 },
  { month: 'May', queries: 2100, resolved: 2020 },
  { month: 'Jun', queries: 2350, resolved: 2280 },
];

const systemDistribution = [
  { name: 'Database Issues', value: 35, color: '#ef4444' },
  { name: 'OS Compatibility', value: 28, color: '#3b82f6' },
  { name: 'Web Server', value: 22, color: '#10b981' },
  { name: 'General', value: 15, color: '#f59e0b' },
];

const chartConfig = {
  queries: { label: "Queries", color: "hsl(var(--chart-1))" },
  resolved: { label: "Resolved", color: "hsl(var(--chart-2))" },
};

const StatsOverview = () => {
  const stats = [
    {
      title: 'Total Queries',
      value: '12,847',
      change: '+12.5%',
      trend: 'up',
      icon: Activity,
      description: 'vs last month'
    },
    {
      title: 'Active Systems',
      value: '1,247',
      change: '+3.2%',
      trend: 'up',
      icon: Server,
      description: 'being monitored'
    },
    {
      title: 'Compatibility Score',
      value: '94.2%',
      change: '+2.1%',
      trend: 'up',
      icon: Database,
      description: 'average score'
    },
    {
      title: 'Response Time',
      value: '1.2s',
      change: '-8.3%',
      trend: 'down',
      icon: Users,
      description: 'average response'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Key Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <Card key={index} className="border border-gray-200/50 shadow-sm hover:shadow-md transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                  <p className="text-3xl font-bold text-gray-900">{stat.value}</p>
                  <div className="flex items-center space-x-2">
                    <div className={`flex items-center text-sm ${
                      stat.trend === 'up' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {stat.trend === 'up' ? (
                        <TrendingUp className="w-4 h-4 mr-1" />
                      ) : (
                        <TrendingDown className="w-4 h-4 mr-1" />
                      )}
                      {stat.change}
                    </div>
                    <span className="text-sm text-gray-500">{stat.description}</span>
                  </div>
                </div>
                <div className="w-12 h-12 bg-slate-100 rounded-lg flex items-center justify-center">
                  <stat.icon className="w-6 h-6 text-slate-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Monthly Queries Trend */}
        <Card className="border border-gray-200/50 shadow-sm">
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-gray-900">
              Query Volume Trends
            </CardTitle>
            <p className="text-sm text-gray-600">Monthly queries and resolution rates</p>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-80">
              <LineChart data={monthlyQueries}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="month" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line 
                  type="monotone" 
                  dataKey="queries" 
                  stroke="#3b82f6" 
                  strokeWidth={3}
                  dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="resolved" 
                  stroke="#10b981" 
                  strokeWidth={3}
                  dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                />
              </LineChart>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* System Compatibility */}
        <Card className="border border-gray-200/50 shadow-sm">
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-gray-900">
              OS Compatibility Rates
            </CardTitle>
            <p className="text-sm text-gray-600">Compatibility scores by operating system</p>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-80">
              <BarChart data={compatibilityData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="name" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {compatibilityData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* Issue Distribution */}
        <Card className="border border-gray-200/50 shadow-sm">
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-gray-900">
              Issue Distribution
            </CardTitle>
            <p className="text-sm text-gray-600">Breakdown of query categories</p>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig} className="h-80">
              <PieChart>
                <Pie
                  data={systemDistribution}
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  dataKey="value"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                >
                  {systemDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <ChartTooltip content={<ChartTooltipContent />} />
              </PieChart>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* Performance Metrics */}
        <Card className="border border-gray-200/50 shadow-sm">
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-gray-900">
              System Performance
            </CardTitle>
            <p className="text-sm text-gray-600">Key performance indicators</p>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-600">Uptime</span>
                <span className="text-sm font-bold text-green-600">99.97%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{ width: '99.97%' }}></div>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-600">Query Success Rate</span>
                <span className="text-sm font-bold text-blue-600">96.8%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-500 h-2 rounded-full" style={{ width: '96.8%' }}></div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-600">User Satisfaction</span>
                <span className="text-sm font-bold text-orange-600">94.2%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-orange-500 h-2 rounded-full" style={{ width: '94.2%' }}></div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default StatsOverview;
