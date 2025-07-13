
import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { MessageCircle, BarChart3, User, Settings, Send, Database, Server, Monitor, PanelLeft } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

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

const SystemChatStandalone = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I\'m your System Compatibility Assistant. I can help you with questions about operating systems, databases, and web servers. What would you like to know?',
      isUser: false,
      timestamp: new Date(),
      category: 'general'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const location = useLocation();

  const navigationItems = [
    { title: 'Chat', url: '/', icon: MessageCircle },
    { title: 'Dashboard', url: '/dashboard', icon: BarChart3 },
    { title: 'Profile', url: '/profile', icon: User },
  ];

  const chatSessions: ChatSession[] = [
    {
      id: '1',
      title: 'Windows Server Setup',
      lastMessage: 'Thanks for the help with IIS configuration!',
      timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000),
      messageCount: 12
    },
    {
      id: '2',
      title: 'MySQL Compatibility',
      lastMessage: 'What about PostgreSQL vs MySQL?',
      timestamp: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000),
      messageCount: 8
    },
    {
      id: '3',
      title: 'Linux Migration',
      lastMessage: 'Ubuntu vs CentOS for production?',
      timestamp: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000),
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

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  const getNavClasses = (path: string) => 
    isActive(path) ? "bg-slate-800 text-white" : "hover:bg-slate-100 text-gray-700";

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

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      isUser: true,
      timestamp: new Date(),
      category: detectCategory(inputValue)
    };

    setMessages(prev => [...prev, userMessage]);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: `I understand you're asking about ${userMessage.category}. Let me help you with that. Based on your question about "${inputValue}", here's what I recommend...`,
        isUser: false,
        timestamp: new Date(),
        category: userMessage.category
      };
      setMessages(prev => [...prev, aiResponse]);
    }, 1000);

    setInputValue('');
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
    <div className="min-h-screen flex w-full bg-gray-50">
      {/* Sidebar */}
      <div className={`${sidebarCollapsed ? 'w-14' : 'w-64'} bg-white border-r transition-all duration-300 flex flex-col`}>
        {/* Sidebar Header */}
        <div className="p-4 border-b">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-slate-800 rounded-lg flex items-center justify-center">
              <Settings className="w-4 h-4 text-white" />
            </div>
            {!sidebarCollapsed && (
              <div>
                <h2 className="text-lg font-semibold text-gray-900">SysCompat</h2>
                <p className="text-xs text-gray-500">System Assistant</p>
              </div>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="mt-2 ml-auto h-7 w-7"
          >
            <PanelLeft className="w-4 h-4" />
          </Button>
        </div>

        {/* Navigation */}
        <div className="p-4">
          <div className="mb-4">
            <h3 className="px-4 py-2 text-xs font-medium text-gray-500 uppercase tracking-wide">
              Navigation
            </h3>
            <nav className="space-y-1">
              {navigationItems.map((item) => (
                <NavLink
                  key={item.title}
                  to={item.url}
                  className={`flex items-center gap-3 px-4 py-2 mx-2 rounded-lg transition-colors ${getNavClasses(item.url)}`}
                >
                  <item.icon className="w-4 h-4 flex-shrink-0" />
                  {!sidebarCollapsed && <span className="font-medium">{item.title}</span>}
                </NavLink>
              ))}
            </nav>
          </div>

          {/* Chat History */}
          {!sidebarCollapsed && (
            <div>
              <h3 className="px-4 py-2 text-xs font-medium text-gray-500 uppercase tracking-wide">
                Recent Chats
              </h3>
              <div className="space-y-2">
                {chatSessions.map((session) => (
                  <div
                    key={session.id}
                    className="mx-2 p-3 rounded-lg bg-gray-50 hover:bg-gray-100 cursor-pointer transition-colors"
                  >
                    <h4 className="font-medium text-sm text-gray-900 truncate">{session.title}</h4>
                    <p className="text-xs text-gray-500 mt-1 truncate">{session.lastMessage}</p>
                    <div className="flex justify-between items-center mt-2">
                      <span className="text-xs text-gray-400">{session.messageCount} messages</span>
                      <span className="text-xs text-gray-400">
                        {session.timestamp.toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">System Chat</h1>
              <p className="text-gray-500 text-sm mt-1">Get help with system compatibility questions</p>
            </div>
            <div className="flex items-center gap-3">
              <Button variant="outline" size="sm">
                <Settings className="w-4 h-4 mr-2" />
                Configure System
              </Button>
            </div>
          </div>
        </div>

        {/* Chat Interface */}
        <div className="flex-1 p-6">
          <Card className="h-full shadow-lg border border-gray-200/50 flex flex-col">
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
            <CardContent className="p-0 flex-1 flex flex-col">
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
              <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
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
                      <p className="text-sm leading-relaxed">{message.text}</p>
                      <p className={`text-xs mt-2 ${message.isUser ? 'text-slate-300' : 'text-gray-500'}`}>
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                ))}
              </div>

              {/* Input */}
              <div className="p-4 border-t border-gray-100">
                <div className="flex gap-2">
                  <Input
                    placeholder="Ask about system compatibility..."
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    className="flex-1 border-gray-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
                  />
                  <Button 
                    onClick={handleSendMessage} 
                    disabled={!inputValue.trim()}
                    className="bg-slate-800 hover:bg-slate-700 text-white px-4"
                  >
                    <Send className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
};

export default SystemChatStandalone;
