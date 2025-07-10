
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Send, MessageCircle, Database, Server, Monitor, Loader2 } from 'lucide-react';
import { apiService, AnalyzeRequest, AnalyzeResponse } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  category?: 'os' | 'database' | 'webserver' | 'general';
  analysisResults?: AnalyzeResponse['results'];
}

interface ChatInterfaceProps {
  systemConfig?: {
    operatingSystem?: string;
    database?: string;
    webServers?: string[];
  };
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ systemConfig }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your System Compatibility Assistant. I can help you analyze software compatibility with your infrastructure. What would you like to know?',
      isUser: false,
      timestamp: new Date(),
      category: 'general'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const quickActions = [
    { 
      text: 'Check OS compatibility', 
      icon: Monitor, 
      category: 'os' as const,
      color: 'bg-blue-50 border-blue-200 text-blue-700 hover:bg-blue-100'
    },
    { 
      text: 'Database requirements', 
      icon: Database, 
      category: 'database' as const,
      color: 'bg-green-50 border-green-200 text-green-700 hover:bg-green-100'
    },
    { 
      text: 'Web server setup', 
      icon: Server, 
      category: 'webserver' as const,
      color: 'bg-orange-50 border-orange-200 text-orange-700 hover:bg-orange-100'
    },
    {
      text: 'Upgrade Apache to 2.4.50',
      icon: Server,
      category: 'webserver' as const,
      color: 'bg-purple-50 border-purple-200 text-purple-700 hover:bg-purple-100'
    },
    {
      text: 'Install MySQL 8.0',
      icon: Database,
      category: 'database' as const,
      color: 'bg-emerald-50 border-emerald-200 text-emerald-700 hover:bg-emerald-100'
    }
  ];

  const detectCategory = (text: string): Message['category'] => {
    const lowerText = text.toLowerCase();
    if (lowerText.includes('os') || lowerText.includes('operating system') || lowerText.includes('windows') || lowerText.includes('linux') || lowerText.includes('mac')) {
      return 'os';
    }
    if (lowerText.includes('database') || lowerText.includes('mysql') || lowerText.includes('postgresql') || lowerText.includes('oracle')) {
      return 'database';
    }
    if (lowerText.includes('web server') || lowerText.includes('apache') || lowerText.includes('nginx') || lowerText.includes('iis')) {
      return 'webserver';
    }
    return 'general';
  };

  const formatAnalysisResults = (results: AnalyzeResponse['results']): string => {
    if (!results || results.length === 0) {
      return 'No analysis results available.';
    }

    let formattedText = '';
    
    results.forEach((result, index) => {
      const { request, compatibility, affected_models, conflicts, warnings, recommendations } = result;
      
      formattedText += `**Analysis ${index + 1}: ${request.action} ${request.software_name} ${request.version || ''}**\n\n`;
      formattedText += `**Status:** ${compatibility.status} (${(compatibility.confidence * 100).toFixed(1)}% confidence)\n\n`;
      
      if (affected_models.length > 0) {
        formattedText += '**Affected Models:**\n';
        affected_models.forEach(model => {
          formattedText += `- ${model.model} (${model.environments.join(', ')})\n`;
        });
        formattedText += '\n';
      }
      
      if (conflicts.length > 0) {
        formattedText += '**Conflicts:**\n';
        conflicts.forEach(conflict => {
          formattedText += `- ${conflict}\n`;
        });
        formattedText += '\n';
      }
      
      if (warnings.length > 0) {
        formattedText += '**Warnings:**\n';
        warnings.forEach(warning => {
          formattedText += `- ${warning}\n`;
        });
        formattedText += '\n';
      }
      
      if (recommendations.length > 0) {
        formattedText += '**Recommendations:**\n';
        recommendations.forEach(rec => {
          formattedText += `- ${rec}\n`;
        });
        formattedText += '\n';
      }
      
      if (index < results.length - 1) {
        formattedText += '---\n\n';
      }
    });
    
    return formattedText;
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      isUser: true,
      timestamp: new Date(),
      category: detectCategory(inputValue)
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Prepare request data
      const requestData: AnalyzeRequest = {
        query: inputValue,
        sessionId: `session_${Date.now()}`,
        userOS: navigator.platform || 'unknown'
      };

      // Add system configuration if available
      if (systemConfig) {
        if (systemConfig.operatingSystem) {
          requestData.operatingSystem = systemConfig.operatingSystem;
        }
        if (systemConfig.database) {
          requestData.database = systemConfig.database;
        }
        if (systemConfig.webServers && systemConfig.webServers.length > 0) {
          requestData.webServers = systemConfig.webServers;
        }
      }

      // Call API
      const response = await apiService.analyzeCompatibility(requestData);
      
      // Create AI response message
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: formatAnalysisResults(response.results),
        isUser: false,
        timestamp: new Date(),
        category: detectCategory(inputValue),
        analysisResults: response.results
      };

      setMessages(prev => [...prev, aiResponse]);

    } catch (error) {
      console.error('Analysis failed:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: `Sorry, I encountered an error while analyzing your request. Please try again or rephrase your question.`,
        isUser: false,
        timestamp: new Date(),
        category: 'general'
      };

      setMessages(prev => [...prev, errorMessage]);
      
      toast({
        title: "Analysis Failed",
        description: "There was an error processing your request. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuickAction = (action: typeof quickActions[0]) => {
    setInputValue(action.text);
  };

  const getCategoryIcon = (category: Message['category']) => {
    switch (category) {
      case 'os': return <Monitor className="w-3 h-3" />;
      case 'database': return <Database className="w-3 h-3" />;
      case 'webserver': return <Server className="w-3 h-3" />;
      default: return <MessageCircle className="w-3 h-3" />;
    }
  };

  const getCategoryColor = (category: Message['category']) => {
    switch (category) {
      case 'os': return 'bg-blue-100 text-blue-700';
      case 'database': return 'bg-green-100 text-green-700';
      case 'webserver': return 'bg-orange-100 text-orange-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <Card className="h-fit shadow-lg border border-gray-200/50">
      <CardHeader className="bg-gradient-to-r from-slate-900 to-slate-800 text-white">
        <CardTitle className="text-lg font-semibold flex items-center gap-3">
          <div className="w-8 h-8 bg-white/10 rounded-lg flex items-center justify-center">
            <MessageCircle className="w-4 h-4" />
          </div>
          System Compatibility Assistant
        </CardTitle>
        <p className="text-slate-300 text-sm mt-1">
          Ask questions about OS, databases, and web servers
        </p>
      </CardHeader>
      <CardContent className="p-0">
        {/* Quick Actions */}
        <div className="p-4 border-b border-gray-100">
          <p className="text-xs font-medium text-gray-500 mb-3 uppercase tracking-wide">Quick Start</p>
          <div className="grid grid-cols-1 gap-2">
            {quickActions.map((action, index) => (
              <Button
                key={index}
                variant="outline"
                onClick={() => handleQuickAction(action)}
                className={`justify-start h-auto p-3 border text-left ${action.color} transition-all duration-200`}
              >
                <action.icon className="w-4 h-4 mr-3 flex-shrink-0" />
                <span className="font-medium">{action.text}</span>
              </Button>
            ))}
          </div>
        </div>

        {/* Messages */}
        <div className="h-96 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-3 ${
                  message.isUser
                    ? 'bg-slate-800 text-white'
                    : 'bg-white border border-gray-200 text-gray-800'
                }`}
              >
                {!message.isUser && message.category && (
                  <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium mb-2 ${getCategoryColor(message.category)}`}>
                    {getCategoryIcon(message.category)}
                    {message.category.toUpperCase()}
                  </div>
                )}
                <div className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</div>
                <p className={`text-xs mt-2 ${message.isUser ? 'text-slate-300' : 'text-gray-500'}`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </div>
          ))}
          
          {/* Loading indicator */}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white border border-gray-200 text-gray-800 rounded-lg px-4 py-3">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm">Analyzing compatibility...</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div className="p-4 border-t border-gray-100">
          <div className="flex gap-2">
            <Input
              placeholder="Ask about system compatibility..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendMessage()}
              className="flex-1 border-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              disabled={isLoading}
            />
            <Button 
              onClick={handleSendMessage} 
              disabled={!inputValue.trim() || isLoading}
              className="bg-slate-800 hover:bg-slate-700 text-white px-4"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ChatInterface;
