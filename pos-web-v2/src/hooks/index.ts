import { useEffect, useRef, useState, useCallback } from 'react';
import type { VoiceInputState } from '@types/index';

// 语音输入Hook
export function useVoiceInput() {
  const [state, setState] = useState<VoiceInputState>({
    isListening: false,
    transcript: '',
    interimTranscript: '',
    confidence: 0,
  });
  
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  
  useEffect(() => {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
      setState((s) => ({ ...s, error: '浏览器不支持语音识别' }));
      return;
    }
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'zh-CN';
    recognition.maxAlternatives = 1;
    
    recognition.onstart = () => {
      setState((s) => ({ ...s, isListening: true, error: undefined }));
    };
    
    recognition.onresult = (event: SpeechRecognitionEvent) => {
      let finalTranscript = '';
      let interimTranscript = '';
      
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalTranscript += transcript;
        } else {
          interimTranscript += transcript;
        }
      }
      
      setState((s) => ({
        ...s,
        transcript: s.transcript + finalTranscript,
        interimTranscript,
        confidence: event.results[0]?.[0]?.confidence || 0,
      }));
    };
    
    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      setState((s) => ({ ...s, error: event.error, isListening: false }));
    };
    
    recognition.onend = () => {
      setState((s) => ({ ...s, isListening: false }));
    };
    
    recognitionRef.current = recognition;
    
    return () => {
      recognition.stop();
    };
  }, []);
  
  const startListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.start();
    }
  }, []);
  
  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
  }, []);
  
  const reset = useCallback(() => {
    setState({
      isListening: false,
      transcript: '',
      interimTranscript: '',
      confidence: 0,
    });
  }, []);
  
  return {
    ...state,
    startListening,
    stopListening,
    reset,
  };
}

// 网络状态Hook
export function useNetworkStatus() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  
  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);
    
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);
  
  return isOnline;
}

// 防抖Hook
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);
  
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);
  
  return debouncedValue;
}

// 本地存储Hook
export function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T | ((val: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(error);
      return initialValue;
    }
  });
  
  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(error);
    }
  }, [key, storedValue]);
  
  return [storedValue, setValue];
}

// 无限滚动Hook
export function useInfiniteScroll(
  callback: () => void,
  hasMore: boolean,
  isLoading: boolean
) {
  const observerRef = useRef<IntersectionObserver | null>(null);
  const targetRef = useRef<HTMLDivElement | null>(null);
  
  useEffect(() => {
    if (isLoading || !hasMore) return;
    
    observerRef.current = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasMore && !isLoading) {
          callback();
        }
      },
      { threshold: 0.5 }
    );
    
    if (targetRef.current) {
      observerRef.current.observe(targetRef.current);
    }
    
    return () => {
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, [callback, hasMore, isLoading]);
  
  return targetRef;
}

// 地理位置Hook
export function useGeolocation() {
  const [location, setLocation] = useState<GeolocationPosition | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    if (!navigator.geolocation) {
      setError('浏览器不支持地理位置');
      return;
    }
    
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setLocation(position);
        setError(null);
      },
      (err) => {
        setError(err.message);
      }
    );
  }, []);
  
  return { location, error };
}

// 主题Hook
export function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window === 'undefined') return 'light';
    
    const stored = localStorage.getItem('pos_theme');
    if (stored === 'dark' || stored === 'light') return stored;
    
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  });
  
  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove('light', 'dark');
    root.classList.add(theme);
    localStorage.setItem('pos_theme', theme);
  }, [theme]);
  
  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  }, []);
  
  return { theme, setTheme, toggleTheme };
}

// 快捷键Hook
export function useHotkey(key: string, callback: () => void, modifiers?: { ctrl?: boolean; alt?: boolean; shift?: boolean }) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== key.toLowerCase()) return;
      
      if (modifiers?.ctrl && !event.ctrlKey) return;
      if (modifiers?.alt && !event.altKey) return;
      if (modifiers?.shift && !event.shiftKey) return;
      
      event.preventDefault();
      callback();
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [key, callback, modifiers]);
}

// 屏幕尺寸Hook
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);
  
  useEffect(() => {
    const media = window.matchMedia(query);
    if (media.matches !== matches) {
      setMatches(media.matches);
    }
    
    const listener = () => setMatches(media.matches);
    media.addEventListener('change', listener);
    return () => media.removeEventListener('change', listener);
  }, [matches, query]);
  
  return matches;
}
