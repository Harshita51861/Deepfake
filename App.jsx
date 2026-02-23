// frontend/src/App.jsx

import { useState, useEffect, useRef } from 'react';
import './App.css';
import ChatMessage from './components/ChatMessage';
import MemoryPanel from './components/MemoryPanel';
import StatsPanel from './components/StatsPanel';
import { sendMessage, getMemories, getStats } from './api/chatApi';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [memories, setMemories] = useState([]);
  const [stats, setStats] = useState(null);
  const [showMemories, setShowMemories] = useState(false);
  const [userId, setUserId] = useState('default-user-001');
  const [userName, setUserName] = useState('User');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Load initial data
    loadMemories();
    loadStats();
    
    // Add welcome message
    setMessages([{
      type: 'bot',
      text: 'Hello! I\'m your memory-enhanced AI assistant. I can remember your preferences, facts, and commitments across our conversations. Tell me about yourself!',
      timestamp: new Date().toISOString()
    }]);
  }, []);

  const loadMemories = async () => {
    try {
      const data = await getMemories(userId);
      setMemories(data.memories || []);
    } catch (error) {
      console.error('Error loading memories:', error);
    }
  };

  const loadStats = async () => {
    try {
      const data = await getStats(userId);
      setStats(data);
      if (data.user_name && data.user_name !== 'User') {
        setUserName(data.user_name);
      }
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = {
      type: 'user',
      text: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await sendMessage(input, userId);
      
      const botMessage = {
        type: 'bot',
        text: response.response,
        timestamp: new Date().toISOString(),
        memoriesUsed: response.relevant_memories || [],
        memoriesCreated: response.memories_created || [],
        memoryUsed: response.memory_used
      };

      setMessages(prev => [...prev, botMessage]);
      
      // Update user name if changed
      if (response.user_name && response.user_name !== 'User') {
        setUserName(response.user_name);
      }

      // Reload memories and stats
      await loadMemories();
      await loadStats();

    } catch (error) {
      const errorMessage = {
        type: 'bot',
        text: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date().toISOString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error('Error sending message:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([{
      type: 'bot',
      text: 'Chat cleared! Your memories are still intact. How can I help you?',
      timestamp: new Date().toISOString()
    }]);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <div className="logo">
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M16 4C9.373 4 4 9.373 4 16C4 22.627 9.373 28 16 28C22.627 28 28 22.627 28 16C28 9.373 22.627 4 16 4ZM16 6C21.546 6 26 10.454 26 16C26 21.546 21.546 26 16 26C10.454 26 6 21.546 6 16C6 10.454 10.454 6 16 6ZM12 11V13H14V11H12ZM18 11V13H20V11H18ZM10 16C10 18.209 11.791 20 14 20H18C20.209 20 22 18.209 22 16H10Z" fill="currentColor"/>
              </svg>
              <span className="logo-text">Memory Chatbot</span>
            </div>
            <div className="user-badge">
              <span className="user-icon">👤</span>
              <span className="user-name">{userName}</span>
            </div>
          </div>
          
          <div className="header-right">
            <button 
              className={`btn-icon ${showMemories ? 'active' : ''}`}
              onClick={() => setShowMemories(!showMemories)}
              title="Toggle Memories"
            >
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M10 3C6.134 3 3 6.134 3 10C3 13.866 6.134 17 10 17C13.866 17 17 13.866 17 10C17 6.134 13.866 3 10 3ZM10 5C12.785 5 15 7.215 15 10C15 12.785 12.785 15 10 15C7.215 15 5 12.785 5 10C5 7.215 7.215 5 10 5Z" fill="currentColor"/>
              </svg>
              <span className="badge">{memories.length}</span>
            </button>
            <button className="btn-icon" onClick={clearChat} title="Clear Chat">
              <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M8 3V4H3V6H4V16C4 17.105 4.895 18 6 18H14C15.105 18 16 17.105 16 16V6H17V4H12V3H8ZM6 6H14V16H6V6ZM8 8V14H10V8H8ZM12 8V14H14V8H12Z" fill="currentColor"/>
              </svg>
            </button>
          </div>
        </div>
        
        {stats && (
          <div className="header-stats">
            <div className="stat">
              <span className="stat-label">Memories:</span>
              <span className="stat-value">{stats.stats?.active_memories || 0}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Turn:</span>
              <span className="stat-value">{stats.current_turn || 1}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Confidence:</span>
              <span className="stat-value">{(stats.stats?.avg_confidence || 0).toFixed(2)}</span>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <div className="app-content">
        {/* Chat Area */}
        <main className="chat-container">
          <div className="messages">
            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))}
            {loading && (
              <div className="message bot-message">
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="input-container">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message... (Press Enter to send)"
              rows="1"
              disabled={loading}
            />
            <button 
              onClick={handleSend} 
              disabled={!input.trim() || loading}
              className="send-button"
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z" fill="currentColor"/>
              </svg>
            </button>
          </div>
        </main>

        {/* Sidebar */}
        {showMemories && (
          <aside className="sidebar">
            <div className="sidebar-header">
              <h2>Memory Bank</h2>
              <button 
                className="btn-close"
                onClick={() => setShowMemories(false)}
              >×</button>
            </div>
            
            <StatsPanel stats={stats} />
            <MemoryPanel memories={memories} onRefresh={loadMemories} />
          </aside>
        )}
      </div>
    </div>
  );
}

export default App;
