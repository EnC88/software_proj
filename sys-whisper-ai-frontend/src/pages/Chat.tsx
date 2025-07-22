
import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { MessageCircle, Send, History, Settings, Monitor, Database, Server, Trash, ChevronDown, ChevronRight, BarChart3 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useUserConfig } from '@/hooks/useUserConfig';
import { getOrCreateUserId } from '@/lib/utils';
import FilteringGraph from '@/components/FilteringGraph';

interface FilteringStep {
  stage: string;
  count: number;
  description: string;
  label: string;
}

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  category?: 'os' | 'database' | 'webserver' | 'general';
  filteringSteps?: FilteringStep[];
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
  const [sessionId, setSessionId] = useState('default_session');
  const [sessions, setSessions] = useState<{ [key: string]: Message[] }>({});
  const [sessionTitles, setSessionTitles] = useState<{ [key: string]: string }>({});
  const [expandedSteps, setExpandedSteps] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const [longWait, setLongWait] = useState(false);
  const [selectionMode, setSelectionMode] = useState(false);
  const [selectedSessions, setSelectedSessions] = useState<Set<string>>(new Set());
  
  // Helper function to extract content from potential TaskResult objects
  const extractContent = (data: any): string => {
    if (typeof data === 'string') {
      return data;
    }
    if (data && typeof data === 'object') {
      if (data.content) {
        return String(data.content);
      }
      if (data.messages && Array.isArray(data.messages)) {
        // Find the TextMessage from the agent
        for (const message of data.messages) {
          if (message.type === 'TextMessage' && message.source === 'RecommendationValidationAgent' && message.content) {
            return String(message.content);
          }
        }
        // Fallback to any TextMessage
        for (const message of data.messages) {
          if (message.type === 'TextMessage' && message.content) {
            return String(message.content);
          }
        }
      }
    }
    return String(data);
  };

  // Fetch chat history on load
  useEffect(() => {
    // Always start a new session on page load
    const newSessionId = crypto.randomUUID();
    setSessionId(newSessionId);
    setMessages([
      {
        id: '1',
        text: "Hello! I'm your System Compatibility Assistant. I can help you with questions about operating systems, databases, and web servers. Configure your system preferences in your profile for personalized recommendations.",
        isUser: false,
        timestamp: new Date(),
        category: 'general'
      }
    ]);
    // Fetch all sessions for the sidebar/history
    const userId = getOrCreateUserId();
    fetch(`/api/user/chat?user_id=${encodeURIComponent(userId)}`)
      .then(res => res.json())
      .then(data => {
        const sessionsRaw = data.sessions || {};
        const sessions: { [key: string]: Message[] } = {};
        Object.keys(sessionsRaw).forEach(sessionKey => {
          sessions[sessionKey] = (sessionsRaw[sessionKey] as any[]).map(msg => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }));
        });
        setSessions(sessions);
        // Set sessionTitles from backend if available
        if (data.session_titles) {
          setSessionTitles(data.session_titles);
        }
      });
  }, []);

  // Scroll to bottom on new message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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
    const userId = getOrCreateUserId();
    let newSessionId = sessionId;
    let isNewSession = false;
    if (messages.length === 0 || (messages.length === 1 && messages[0].isUser === false)) {
      newSessionId = crypto.randomUUID();
      setSessionId(newSessionId);
      isNewSession = true;
    }
    const messageId = Date.now().toString();
    const userMessage: Message = {
      id: messageId,
      text: inputValue,
      isUser: true,
      timestamp: new Date(),
      category: 'general'
    };
    setMessages(prev => [...prev, userMessage]);
    setIsAnalyzing(true);
    setLongWait(false);
    // Save user message to backend
    await fetch('/api/user/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: userId,
        session_id: newSessionId,
        message_id: messageId,
        message_data: {
          text: inputValue,
          isUser: true,
          timestamp: new Date().toISOString(),
          category: 'general'
        }
      })
    });
    // If this is a new session, generate a title
    if (isNewSession) {
      try {
        const resp = await fetch('/api/generate_title', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: inputValue })
        });
        const data = await resp.json();
        let title = data.title;
        // If title is an object with a 'content' field, use it
        if (title && typeof title === 'object' && 'content' in title) {
          title = title.content;
        } else if (typeof title === 'string') {
          // Find all content='...' matches
          const contentMatches = [...title.matchAll(/content=['"]([^'\"]+)['"]/gi)];
          if (contentMatches.length > 0) {
            // Use the last match (most likely the LLM's title)
            title = contentMatches[contentMatches.length - 1][1];
          } else {
            // Fallback: remove any trailing metadata or JSON
            title = title.split('\n')[0].trim();
          }
        } else {
          // Fallback
          title = inputValue.slice(0, 30);
        }
        // Save title to backend
        await fetch(
          `/api/user/chat_title?user_id=${encodeURIComponent(userId)}&session_id=${encodeURIComponent(newSessionId)}&title=${encodeURIComponent(title)}`,
          { method: 'POST' }
        );
        setSessionTitles(prev => ({ ...prev, [newSessionId]: title }));
      } catch {
        setSessionTitles(prev => ({ ...prev, [newSessionId]: inputValue.slice(0, 30) }));
      }
    }
    // Fetch updated chat history to update sidebar
    fetch(`/api/user/chat?user_id=${encodeURIComponent(userId)}`)
      .then(res => res.json())
      .then(data => {
        const sessionsRaw = data.sessions || {};
        const sessions: { [key: string]: Message[] } = {};
        Object.keys(sessionsRaw).forEach(sessionKey => {
          sessions[sessionKey] = (sessionsRaw[sessionKey] as any[]).map(msg => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }));
        });
        setSessions(sessions);
        // Set sessionTitles from backend if available
        if (data.session_titles) {
          setSessionTitles(data.session_titles);
        }
      });

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

    // --- Custom timeout and long-wait logic ---
    let timeoutId: number | null = null;
    let longWaitId: number | null = null;
    try {
      // Show long-wait message after 10s
      longWaitId = window.setTimeout(() => setLongWait(true), 10000);
      // Custom fetch with timeout (90s)
      const fetchWithTimeout = (resource: RequestInfo, options: RequestInit = {}) => {
        return new Promise<Response>((resolve, reject) => {
          timeoutId = window.setTimeout(() => reject(new Error('timeout')), 90000);
          fetch(resource, options)
            .then(res => {
              if (timeoutId) window.clearTimeout(timeoutId);
              resolve(res);
            })
            .catch(err => {
              if (timeoutId) window.clearTimeout(timeoutId);
              reject(err);
            });
        });
      };
      // Send to new RAG endpoint
      const response = await fetchWithTimeout('/api/rag_query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query: inputValue,
          os: activeConfig.operatingSystem || null,
          database: activeConfig.database || null,
          webServers: activeConfig.webServers || []
        }),
      });
      let aiResponseText = "Sorry, I couldn't process your request. Please try again.";
      let filteringSteps: FilteringStep[] | undefined;
      if (response.ok) {
        const data = await response.json();
        console.log("RAG API response:", data); // Debug log
        
        // Extract filtering steps from backend response
        if (data.filtering_steps && Array.isArray(data.filtering_steps)) {
          filteringSteps = data.filtering_steps;
        }
        
        // Only show validated recommendations if available, otherwise show the original result
        if (data.validated_recommendations && data.validated_recommendations !== 'No recommendations to validate.') {
          const extractedRecommendations = extractContent(data.validated_recommendations);
          if (extractedRecommendations.trim()) {
            aiResponseText = `\u2728 Enhanced Recommendations:\n${extractedRecommendations}`;
          } else {
            aiResponseText = data.result;
          }
        } else {
        aiResponseText = data.result;
        }
      }
      const aiMessageId = (Date.now() + 1).toString();
      const aiResponse: Message = {
        id: aiMessageId,
        text: aiResponseText,
        isUser: false,
        timestamp: new Date(),
        category: 'general',
        filteringSteps: filteringSteps
      };
      setMessages(prev => [...prev, aiResponse]);
      // Save AI response to backend
      fetch('/api/user/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: newSessionId,
          message_id: aiMessageId,
          message_data: {
            text: aiResponseText,
            isUser: false,
            timestamp: new Date().toISOString(),
            category: 'general',
            filteringSteps: filteringSteps
          }
        })
      });
    } catch (error: any) {
      let errorText = 'Sorry, I encountered an error. Please try again.';
      if (error && error.message === 'timeout') {
        errorText = 'The system is still processing your request. Please wait or try again in a moment.';
      }
      const aiMessageId = (Date.now() + 1).toString();
      const aiResponse: Message = {
        id: aiMessageId,
        text: errorText,
        isUser: false,
        timestamp: new Date(),
        category: 'general'
      };
      setMessages(prev => [...prev, aiResponse]);
      // Save error AI response to backend
      fetch('/api/user/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: newSessionId,
          message_id: aiMessageId,
          message_data: {
            text: errorText,
            isUser: false,
            timestamp: new Date().toISOString(),
            category: 'general'
          }
        })
      });
    } finally {
      if (timeoutId) window.clearTimeout(timeoutId);
      if (longWaitId) window.clearTimeout(longWaitId);
      setIsAnalyzing(false);
      setLongWait(false);
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

  const toggleFilteringSteps = (messageId: string) => {
    setExpandedSteps(prev => ({
      ...prev,
      [messageId]: !prev[messageId]
    }));
  };

  // In the chat sidebar, allow switching sessions (optional for now)
  const handleSessionChange = (newSessionId: string) => {
    setSessionId(newSessionId);
    setMessages((sessions[newSessionId] || []).map(msg => ({
      ...msg,
      timestamp: new Date(msg.timestamp)
    })));
  };

  const handleDeleteSession = async (sessionIdToDelete: string) => {
    const userId = getOrCreateUserId();
    if (!window.confirm('Delete this chat session? This cannot be undone.')) return;
    await fetch(`/api/user/chat_session?user_id=${encodeURIComponent(userId)}&session_id=${encodeURIComponent(sessionIdToDelete)}`, {
      method: 'DELETE'
    });
    setSessions(prev => {
      const newSessions = { ...prev };
      delete newSessions[sessionIdToDelete];
      return newSessions;
    });
    setSessionTitles(prev => {
      const newTitles = { ...prev };
      delete newTitles[sessionIdToDelete];
      return newTitles;
    });
    if (sessionId === sessionIdToDelete) {
      const remaining = Object.keys(sessions).filter(id => id !== sessionIdToDelete);
      if (remaining.length > 0) {
        handleSessionChange(remaining[0]);
      } else {
        const newSessionId = crypto.randomUUID();
        setSessionId(newSessionId);
        setMessages([
          {
            id: '1',
            text: "Hello! I'm your System Compatibility Assistant. I can help you with questions about operating systems, databases, and web servers. Configure your system preferences in your profile for personalized recommendations.",
            isUser: false,
            timestamp: new Date(),
            category: 'general'
          }
        ]);
      }
    }
  };

  // Replace chatHistory with real sessions in sidebar
  const chatHistory: ChatSession[] = Object.entries(sessions)
    .map(([id, msgsRaw]) => {
      const msgs = msgsRaw as Message[];
      const lastMsg = msgs[msgs.length - 1];
      // Find the first user message in the session
      const firstUserMsg = msgs.find(m => m.isUser)?.text || '';
      return {
        id,
        title: sessionTitles[id] || firstUserMsg.slice(0, 30) || 'New Chat',
        lastMessage: lastMsg?.text || '',
        timestamp: lastMsg ? new Date(lastMsg.timestamp) : new Date(),
        messageCount: msgs.length
      };
    })
    // Only show sessions with a title or a user message
    .filter(session => !!session.title && session.title !== '')
    .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime()); // Sort by most recent

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
            {/* New Chat Button */}
            <Button
              variant="outline"
              size="sm"
              className="w-full mt-4 mb-2 text-blue-700 border-blue-200 hover:bg-blue-50"
              onClick={() => {
                const newSessionId = crypto.randomUUID();
                setSessionId(newSessionId);
                setMessages([
                  {
                    id: '1',
                    text: "Hello! I'm your System Compatibility Assistant. I can help you with questions about operating systems, databases, and web servers. Configure your system preferences in your profile for personalized recommendations.",
                    isUser: false,
                    timestamp: new Date(),
                    category: 'general'
                  }
                ]);
              }}
            >
              + New Chat
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {chatHistory.map((session) => (
              <div
                key={session.id}
                className={`p-3 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors ${session.id === sessionId ? 'ring-2 ring-blue-400' : ''}`}
                onClick={() => handleSessionChange(session.id)}
              >
                <div className="flex justify-between items-center">
                  <h4 className="font-medium text-gray-900 text-sm">{session.title}</h4>
                  {/* Delete button */}
                  <button
                    className="ml-2 text-gray-400 hover:text-red-600"
                    onClick={e => {
                      e.stopPropagation();
                      handleDeleteSession(session.id);
                    }}
                    title="Delete chat"
                  >
                    <Trash className="w-4 h-4" />
                  </button>
                </div>
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
            <div ref={messagesEndRef} />
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4 flex-shrink-0">
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
        <div className="bg-white border-b border-gray-100 p-4 flex-shrink-0">
          <p className="text-sm font-medium text-gray-700 mb-3">Quick Actions</p>
          <div className="flex gap-3 flex-wrap">
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
        <div className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0">
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
                
                {/* 
                {!message.isUser && message.filteringSteps && (
                  <div className="mt-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleFilteringSteps(message.id)}
                      className="text-xs px-2 py-1 h-auto text-gray-600 hover:text-gray-800 hover:bg-gray-100 transition-colors"
                    >
                      <BarChart3 className="w-3 h-3 mr-1" />
                      View filtering process
                      {expandedSteps[message.id] ? 
                        <ChevronDown className="w-3 h-3 ml-1" /> : 
                        <ChevronRight className="w-3 h-3 ml-1" />
                      }
                    </Button>
                    
                    {expandedSteps[message.id] && (
                      <div className="mt-3">
                        <div className="text-xs font-medium text-gray-700 mb-3">How I narrowed down to your recommendations:</div>
                        <FilteringGraph messageId={message.id} filteringSteps={message.filteringSteps} />
                      </div>
                    )}
                  </div>
                )}
                */}
                
                <p className={`text-xs mt-2 ${message.isUser ? 'text-slate-300' : 'text-gray-500'}`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </p>
              </div>
            </div>
          ))}
          {longWait && (
            <div className="flex items-center gap-2 text-sm text-orange-600 mt-2">
              <div className="w-4 h-4 border-2 border-orange-600 border-t-transparent rounded-full animate-spin"></div>
              Still working... this may take a while for complex queries.
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 p-4 flex-shrink-0">
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
