import { useState, useEffect } from 'react';
import { STORAGE_KEYS, API_ENDPOINTS } from '@/lib/constants';

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

export const useUserConfig = () => {
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
  const [loading, setLoading] = useState(true);

  // Load configuration from localStorage and API
  useEffect(() => {
    const loadConfig = async () => {
      try {
        setLoading(true);
        
        // Try to load from localStorage first
        const savedConfig = localStorage.getItem(STORAGE_KEYS.USER_SYSTEM_CONFIG);
        console.log('Loading from localStorage, key:', STORAGE_KEYS.USER_SYSTEM_CONFIG);
        console.log('Found saved config:', savedConfig);
        let hasLocalConfig = false;
        
        if (savedConfig) {
          try {
            const parsedConfig = JSON.parse(savedConfig);
            console.log('Parsed config:', parsedConfig);
            // Only set config if it has meaningful data
            if (parsedConfig.operatingSystem || parsedConfig.database || parsedConfig.webServers.length > 0) {
              setConfig(parsedConfig);
              hasLocalConfig = true;
              console.log('Set config from localStorage');
            } else {
              console.log('Config has no meaningful data, not setting');
            }
          } catch (parseError) {
            console.error('Error parsing saved config:', parseError);
            // Remove invalid config from localStorage
            localStorage.removeItem(STORAGE_KEYS.USER_SYSTEM_CONFIG);
          }
        }

        // Only try to load from API if we don't have local config
        if (!hasLocalConfig) {
          try {
            const response = await fetch(API_ENDPOINTS.USER_CONFIG);
            if (response.ok) {
              const apiConfig = await response.json();
              // Only use API config if it has meaningful data
              if (apiConfig.operatingSystem || apiConfig.database || apiConfig.webServers.length > 0) {
                setConfig(apiConfig);
                // Update localStorage with API data
                localStorage.setItem(STORAGE_KEYS.USER_SYSTEM_CONFIG, JSON.stringify(apiConfig));
              }
            }
          } catch (err) {
            console.warn('Failed to load from API, using localStorage:', err);
          }
        }
      } catch (err) {
        console.error('Error loading user configuration:', err);
      } finally {
        setLoading(false);
      }
    };

    loadConfig();
  }, []);

  // Get active configuration for chat (only enabled components)
  const getActiveConfig = () => {
    const activeConfig: Partial<SystemConfig> = {};
    
    if (config.useInChat.os && config.operatingSystem) {
      activeConfig.operatingSystem = config.operatingSystem;
    }
    
    if (config.useInChat.database && config.database) {
      activeConfig.database = config.database;
    }
    
    if (config.useInChat.webServers && config.webServers.length > 0) {
      activeConfig.webServers = config.webServers;
    }
    
    return activeConfig;
  };

  // Check if a specific category should be included in chat
  const shouldIncludeInChat = (category: 'os' | 'database' | 'webserver') => {
    switch (category) {
      case 'os':
        return config.useInChat.os && !!config.operatingSystem;
      case 'database':
        return config.useInChat.database && !!config.database;
      case 'webserver':
        return config.useInChat.webServers && config.webServers.length > 0;
      default:
        return false;
    }
  };

  // Get personalized context for chat responses
  const getPersonalizedContext = () => {
    const activeConfig = getActiveConfig();
    const contextParts = [];
    
    if (activeConfig.operatingSystem) {
      contextParts.push(`Operating System: ${activeConfig.operatingSystem}`);
    }
    
    if (activeConfig.database) {
      contextParts.push(`Database: ${activeConfig.database}`);
    }
    
    if (activeConfig.webServers && activeConfig.webServers.length > 0) {
      contextParts.push(`Web Servers: ${activeConfig.webServers.join(', ')}`);
    }
    
    return contextParts.length > 0 
      ? `Based on your system configuration (${contextParts.join(', ')}), here's what I recommend:`
      : "Here's what I recommend:";
  };

  // Save configuration to localStorage and optionally to API
  const saveConfig = async (newConfig: SystemConfig) => {
    try {
      // Update local state
      setConfig(newConfig);
      
      // Save to localStorage
      const configString = JSON.stringify(newConfig);
      localStorage.setItem(STORAGE_KEYS.USER_SYSTEM_CONFIG, configString);
      console.log('Saved to localStorage with key:', STORAGE_KEYS.USER_SYSTEM_CONFIG);
      console.log('Saved config string:', configString);
      
      // Optionally save to API
      try {
        const response = await fetch(API_ENDPOINTS.USER_CONFIG, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(newConfig),
        });
        
        if (!response.ok) {
          throw new Error('Failed to save to server');
        }
      } catch (err) {
        console.warn('Failed to save to server, but saved locally:', err);
      }
      
      return true;
    } catch (err) {
      console.error('Error saving configuration:', err);
      return false;
    }
  };

  return {
    config,
    setConfig,
    loading,
    getActiveConfig,
    shouldIncludeInChat,
    getPersonalizedContext,
    saveConfig
  };
}; 