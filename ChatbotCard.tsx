
import React, { useState } from 'react';

// All CSS styles embedded
const styles = `
  .chat-container {
    display: flex;
    height: 100vh;
    background-color: #f9fafb;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  }

  .sidebar {
    width: 320px;
    background: white;
    border-right: 1px solid #e5e7eb;
    display: flex;
    flex-direction: column;
    transition: width 0.3s ease;
  }

  .sidebar.hidden {
    width: 0;
    overflow: hidden;
  }

  .sidebar-header {
    padding: 1rem;
    border-bottom: 1px solid #f3f4f6;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .sidebar-title {
    font-size: 1.125rem;
    font-weight: 600;
    color: #111827;
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 1.5rem;
    color: #6b7280;
    cursor: pointer;
    padding: 0.25rem;
  }

  .close-btn:hover {
    color: #374151;
  }

  .sidebar-content {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
  }

  .history-item {
    padding: 0.75rem;
    background: #f9fafb;
    border-radius: 0.5rem;
    margin-bottom: 0.75rem;
    cursor: pointer;
    transition: background-color 0.2s ease;
  }

  .history-item:hover {
    background: #f3f4f6;
  }

  .history-title {
    font-weight: 500;
    color: #111827;
    font-size: 0.875rem;
    margin-bottom: 0.25rem;
  }

  .history-message {
    font-size: 0.75rem;
    color: #6b7280;
    margin-bottom: 0.5rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .history-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #9ca3af;
  }

  .main-area {
    flex: 1;
    display: flex;
    flex-direction: column;
  }

  .header {
    background: white;
    border-bottom: 1px solid #e5e7eb;
    padding: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
  }

  .header-title {
    font-size: 1.5rem;
    font-weight: bold;
    color: #111827;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border: 1px solid #d1d5db;
    border-radius: 0.375rem;
    background: white;
    color: #374151;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .btn:hover {
    background: #f9fafb;
    border-color: #9ca3af;
  }

  .btn-primary {
    background: #1f2937;
    color: white;
    border-color: #1f2937;
  }

  .btn-primary:hover {
    background: #374151;
  }

  .quick-actions {
    background: white;
    border-bottom: 1px solid #f3f4f6;
    padding: 1rem;
  }

  .quick-actions-title {
    font-size: 0.875rem;
    font-weight: 500;
    color: #374151;
    margin-bottom: 0.75rem;
  }

  .quick-actions-grid {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
  }

  .quick-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border: 1px solid;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .quick-btn-os {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #1d4ed8;
  }

  .quick-btn-os:hover {
    background: #dbeafe;
  }

  .quick-btn-db {
    background: #f0fdf4;
    border-color: #bbf7d0;
    color: #15803d;
  }

  .quick-btn-db:hover {
    background: #dcfce7;
  }

  .quick-btn-server {
    background: #fff7ed;
    border-color: #fed7aa;
    color: #c2410c;
  }

  .quick-btn-server:hover {
    background: #ffedd5;
  }

  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .message {
    display: flex;
    max-width: 70%;
  }

  .message.user {
    justify-content: flex-end;
    margin-left: auto;
  }

  .message.assistant {
    justify-content: flex-start;
  }

  .message-content {
    padding: 0.75rem 1rem;
    border-radius: 0.75rem;
    position: relative;
  }

  .message.user .message-content {
    background: #1f2937;
    color: white;
  }

  .message.assistant .message-content {
    background: white;
    border: 1px solid #e5e7eb;
    color: #111827;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  }

  .category-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
  }

  .category-os {
    background: #dbeafe;
    color: #1d4ed8;
  }

  .category-database {
    background: #dcfce7;
    color: #15803d;
  }

  .category-webserver {
    background: #ffedd5;
    color: #c2410c;
  }

  .category-general {
    background: #f3f4f6;
    color: #374151;
  }

  .message-text {
    font-size: 0.875rem;
    line-height: 1.5;
  }

  .message-time {
    font-size: 0.75rem;
    margin-top: 0.5rem;
    opacity: 0.7;
  }

  .input-area {
    background: white;
    border-top: 1px solid #e5e7eb;
    padding: 1rem;
  }

  .input-container {
    display: flex;
    gap: 0.75rem;
    max-width: 64rem;
    margin: 0 auto;
  }

  .input-field {
    flex: 1;
    padding: 0.75rem 1rem;
    border: 1px solid #d1d5db;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    outline: none;
    transition: border-color 0.2s ease;
  }

  .input-field:focus {
    border-color: #1f2937;
    box-shadow: 0 0 0 1px #1f2937;
  }

  .send-btn {
    padding: 0.75rem 1.5rem;
    background: #1f2937;
    color: white;
    border: none;
    border-radius: 0.375rem;
    cursor: pointer;
    transition: background-color 0.2s ease;
  }

  .send-btn:hover:not(:disabled) {
    background: #374151;
  }

  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .icon {
    width: 1rem;
    height: 1rem;
  }
`;

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
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your System Compatibility Assistant. I can help you with questions about operating systems, databases, and web servers. Configure your system preferences in your profile for personalized recommendations.",
      isUser: false,
      timestamp: new Date(),
      category: 'general'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [showHistory, setShowHistory] = useState(false);

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

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);

    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        text: `I understand your question about "${inputValue}". Based on your system configuration, here's what I recommend...`,
        isUser: false,
        timestamp: new Date(),
        category: 'general'
      };
      setMessages(prev => [...prev, aiResponse]);
    }, 1000);

    setInputValue('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  const handleQuickAction = (text: string) => {
    setInputValue(text);
  };

  const getCategoryIcon = (category: Message['category']) => {
    switch (category) {
      case 'os': return '🖥️';
      case 'database': return '🗄️';
      case 'webserver': return '🖥️';
      default: return '💬';
    }
  };

  const getCategoryClass = (category: Message['category']) => {
    switch (category) {
      case 'os': return 'category-os';
      case 'database': return 'category-database';
      case 'webserver': return 'category-webserver';
      default: return 'category-general';
    }
  };

  return (
    <div>
      <style>{styles}</style>
      <div className="chat-container">
        {/* Chat History Sidebar */}
        {showHistory && (
          <div className="sidebar">
            <div className="sidebar-header">
              <h3 className="sidebar-title">Chat History</h3>
              <button 
                className="close-btn"
                onClick={() => setShowHistory(false)}
              >
                ×
              </button>
            </div>
            <div className="sidebar-content">
              {chatHistory.map((session) => (
                <div key={session.id} className="history-item">
                  <h4 className="history-title">{session.title}</h4>
                  <p className="history-message">{session.lastMessage}</p>
                  <div className="history-meta">
                    <span>{session.messageCount} messages</span>
                    <span>{session.timestamp.toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Main Chat Area */}
        <div className="main-area">
          {/* Header */}
          <div className="header">
            <div className="header-left">
              <h1 className="header-title">System Chat</h1>
              <button
                className="btn"
                onClick={() => setShowHistory(!showHistory)}
              >
                🕒 History
              </button>
            </div>
            <button className="btn">
              ⚙️ Configure System
            </button>
          </div>

          {/* Quick Actions */}
          <div className="quick-actions">
            <p className="quick-actions-title">Quick Actions</p>
            <div className="quick-actions-grid">
              <button
                className="quick-btn quick-btn-os"
                onClick={() => handleQuickAction('Check OS compatibility')}
              >
                🖥️ Check OS compatibility
              </button>
              <button
                className="quick-btn quick-btn-db"
                onClick={() => handleQuickAction('Database requirements')}
              >
                🗄️ Database requirements
              </button>
              <button
                className="quick-btn quick-btn-server"
                onClick={() => handleQuickAction('Web server setup')}
              >
                🖥️ Web server setup
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="chat-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`message ${message.isUser ? 'user' : 'assistant'}`}
              >
                <div className="message-content">
                  {!message.isUser && message.category && (
                    <div className={`category-badge ${getCategoryClass(message.category)}`}>
                      {getCategoryIcon(message.category)}
                      {message.category.toUpperCase()}
                    </div>
                  )}
                  <p className="message-text">{message.text}</p>
                  <p className="message-time">
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Input */}
          <div className="input-area">
            <div className="input-container">
              <input
                type="text"
                className="input-field"
                placeholder="Ask about system compatibility..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
              />
              <button 
                className="send-btn"
                onClick={handleSendMessage}
                disabled={!inputValue.trim()}
              >
                📤
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemChatStandalone;
