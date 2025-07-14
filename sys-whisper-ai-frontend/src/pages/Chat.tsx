
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { MessageCircle, Send, History, Settings, Monitor, Database, Server } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useUserConfig } from '@/hooks/useUserConfig';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  category?: 'os' | 'database' | 'webserver' | 'general';
}

interface ChatSession {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  messageCount: number;
}

const Chat = () => {
  const { shouldIncludeInChat, getPersonalizedContext, getActiveConfig } = useUserConfig();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your System Compatibility Assistant. I can help you with questions about operating systems, databases, and web servers. Configure your system preferences in your profile for personalized recommendations.',
      isUser: false,
      timestamp: new Date(),
      category: 'general'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [showHistory, setShowHistory] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Mock chat history
  const chatHistory: ChatSession[] = [
    {
      id: '1',
      title: 'Windows Server Setup',
      lastMessage: 'Thanks for the help with IIS configuration!',
      timestamp: new Date(Date.now() - 86400000),
      messageCount: 12
    },
    {
      id: '2', 
      title: 'MySQL Compatibility',
      lastMessage: 'What about PostgreSQL vs MySQL?',
      timestamp: new Date(Date.now() - 172800000),
      messageCount: 8
    },
    {
      id: '3',
      title: 'Linux Migration',
      lastMessage: 'Ubuntu vs CentOS for production?',
      timestamp: new Date(Date.now() - 259200000),
      messageCount: 15
    }
  ];

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
    }
  ];

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      isUser: true,
      timestamp: new Date(),
      category: 'general' // Let backend handle categorization
    };

    setMessages(prev => [...prev, userMessage]);
    setIsAnalyzing(true);

    // Get active user configuration
    const activeConfig = getActiveConfig();
    
    // Prepare parameters for backend API
    const apiParams = {
      request: inputValue,
      os: activeConfig.operatingSystem || null,
      database: activeConfig.database || null,
      webServers: activeConfig.webServers || [],
      environment: null // Could be added to user config if needed
    };

    try {
      // Send to backend API for analysis
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(apiParams),
      });

      if (response.ok) {
        const analysisResult = await response.json();
        
        // Use the backend's actual analysis results
        let aiResponseText = "I couldn't analyze that request. Please try asking about software compatibility.";
        
        if (analysisResult.results && analysisResult.results.length > 0) {
          const result = analysisResult.results[0];
          
          if (result.is_compatible !== undefined) {
            const status = result.is_compatible ? "✅ Compatible" : "❌ Not Compatible";
            const confidencePercent = Math.round((result.confidence || 0) * 100);
            
            aiResponseText = `## Analysis Result\n\n**Status:** ${status}\n**Confidence:** ${confidencePercent}%\n**Request:** ${result.request}`;
            
            // Add affected servers info
            if (result.affected_servers && result.affected_servers.length > 0) {
              aiResponseText += `\n\n**Affected Servers:** ${result.affected_servers.length} server(s) found`;
            }
            
            // Add conflicts
            if (result.conflicts && result.conflicts.length > 0) {
              aiResponseText += `\n\n**❌ Conflicts Found:**\n${result.conflicts.map(conflict => `• ${conflict}`).join('\n')}`;
            }
            
            // Add warnings
            if (result.warnings && result.warnings.length > 0) {
              aiResponseText += `\n\n**⚠️ Warnings:**\n${result.warnings.map(warning => `• ${warning}`).join('\n')}`;
            }
            
            // Add recommendations
            if (result.recommendations && result.recommendations.length > 0) {
              aiResponseText += `\n\n**💡 Recommendations:**\n${result.recommendations.map(rec => `• ${rec}`).join('\n')}`;
            }
            
            // Add alternative versions
            if (result.alternative_versions && result.alternative_versions.length > 0) {
              aiResponseText += `\n\n**🔄 Alternative Versions:**\n${result.alternative_versions.map(version => `• ${version}`).join('\n')}`;
            }
          }
        }
        
        const aiResponse: Message = {
          id: (Date.now() + 1).toString(),
          text: aiResponseText,
          isUser: false,
          timestamp: new Date(),
          category: 'general'
        };
        setMessages(prev => [...prev, aiResponse]);
      } else {
        // Fallback if API fails
        const aiResponse: Message = {
          id: (Date.now() + 1).toString(),
          text: "Sorry, I couldn't process your request. Please try again.",
          isUser: false,
          timestamp: new Date(),
          category: 'general'
        };
        setMessages(prev => [...prev, aiResponse]);
      }
    } catch (error) {
      console.error('Error calling analysis API:', error);
      // Fallback on error
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: "Sorry, I encountered an error. Please try again.",
        isUser: false,
        timestamp: new Date(),
        category: 'general'
      };
      setMessages(prev => [...prev, aiResponse]);
    } finally {
      setIsAnalyzing(false);
    }

    setInputValue('');
  };

  const getCategoryIcon = (category: Message['category']) => {
    switch (category) {
      case 'os': return <Monitor className="w-3 h-3" />;
      case 'database': return <Database className="w-3 h-3" />;
      case 'webserver': return <Server className="w-3 h-3" />;
      case 'general': return <MessageCircle className="w-3 h-3" />;
      default: return <MessageCircle className="w-3 h-3" />;
    }
  };

  const getCategoryColor = (category: Message['category']) => {
    switch (category) {
      case 'os': return 'bg-blue-100 text-blue-700';
      case 'database': return 'bg-green-100 text-green-700';
      case 'webserver': return 'bg-orange-100 text-orange-700';
      case 'general': return 'bg-gray-100 text-gray-700';
      default: return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Chat History Sidebar */}
      {showHistory && (
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-100">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">Chat History</h3>
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => setShowHistory(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ×
              </Button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {chatHistory.map((session) => (
              <div key={session.id} className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors">
                <h4 className="font-medium text-gray-900 text-sm">{session.title}</h4>
                <p className="text-xs text-gray-600 mt-1 truncate">{session.lastMessage}</p>
                <div className="flex justify-between items-center mt-2">
                  <span className="text-xs text-gray-500">
                    {session.messageCount} messages
                  </span>
                  <span className="text-xs text-gray-500">
                    {session.timestamp.toLocaleDateString()}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <h1 className="text-2xl font-bold text-gray-900">System Chat</h1>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowHistory(!showHistory)}
                className="flex items-center gap-2"
              >
                <History className="w-4 h-4" />
                History
              </Button>
            </div>
            <Link to="/profile">
              <Button variant="outline" className="flex items-center gap-2">
                <Settings className="w-4 h-4" />
                Configure System
              </Button>
            </Link>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white border-b border-gray-100 p-4">
          <p className="text-sm font-medium text-gray-700 mb-3">Quick Actions</p>
          <div className="flex gap-3">
            {quickActions.map((action, index) => (
              <Button
                key={index}
                variant="outline"
                onClick={() => setInputValue(action.text)}
                className={`${action.color} border transition-all duration-200`}
              >
                <action.icon className="w-4 h-4 mr-2" />
                {action.text}
              </Button>
            ))}
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[70%] rounded-xl px-4 py-3 ${
                  message.isUser
                    ? 'bg-slate-800 text-white'
                    : 'bg-white border border-gray-200 text-gray-800 shadow-sm'
                }`}
              >
                {!message.isUser && message.category && (
                  <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium mb-2 ${getCategoryColor(message.category)}`}>
                    {getCategoryIcon(message.category)}
                    {message.category.toUpperCase()}
                  </div>
                )}
                <p className="text-sm leading-relaxed">{message.text}</p>
                <p className={`text-xs mt-2 ${message.isUser ? 'text-slate-300' : 'text-gray-500'}`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="flex gap-3 max-w-4xl mx-auto">
            <Input
              placeholder="Ask about system compatibility..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              className="flex-1 border-gray-300 focus:border-slate-500 focus:ring-1 focus:ring-slate-500"
            />
            <Button 
              onClick={handleSendMessage} 
              disabled={!inputValue.trim() || isAnalyzing}
              className="bg-slate-800 hover:bg-slate-700 text-white px-6"
            >
              {isAnalyzing ? (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Analyzing...
                </div>
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
