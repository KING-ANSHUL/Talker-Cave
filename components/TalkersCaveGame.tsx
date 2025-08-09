

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { GoogleGenAI, GenerateContentResponse, Type } from '@google/genai';
import { TALKERS_CAVE_SCENES, TALKERS_CAVE_SCENE_IMAGES, TALKERS_CAVE_CHARACTER_IMAGES } from '../constants';
import { MicrophoneIcon } from './icons/MicrophoneIcon';
import { SoundIcon } from './icons/SoundIcon';
import { PracticeSoundIcon } from './icons/PracticeSoundIcon';

declare global {
  interface SpeechRecognition {
    lang: string;
    continuous: boolean;
    interimResults: boolean;
    maxAlternatives: number;
    start(): void;
    stop(): void;
    abort(): void;
    addEventListener(type: string, listener: EventListenerOrEventListenerObject): void;
    removeEventListener(type: string, listener: EventListenerOrEventListenerObject): void;
  }

  interface Window {
    SpeechRecognition: { new(): SpeechRecognition };
    webkitSpeechRecognition: { new(): SpeechRecognition };
  }
  interface SpeechRecognitionEvent extends Event {
    readonly resultIndex: number;
    readonly results: SpeechRecognitionResultList;
  }
  interface SpeechRecognitionResultList {
    readonly length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
  }
  interface SpeechRecognitionResult {
    readonly isFinal: boolean;
    readonly length: number;
    item(index: number): SpeechRecognitionAlternative;
    [index: number]: SpeechRecognitionAlternative;
  }
  interface SpeechRecognitionAlternative {
    readonly transcript: string;
    readonly confidence: number;
  }
  interface SpeechRecognitionErrorEvent extends Event {
    readonly error: string;
    readonly message: string;
  }
}

type Step = 'SCENE' | 'CHARACTER' | 'LOADING_SCRIPT' | 'GAME' | 'PRACTICE_PREP' | 'PRACTICE' | 'COMPLETE';
type Scene = keyof typeof TALKERS_CAVE_SCENES;
type ScriptLine = { character: string; line: string };
type Mistake = { said: string; expected: string };
type PracticeWord = { word: string; phonemes: string[] };

interface TalkersCaveGameProps {
  onComplete: () => void;
  userGrade: number;
  currentLevel: number;
  onBackToGrades: () => void;
}

const getDifficultyDescription = (grade: number, level: number): string => {
  if (level <= 5) return `at a foundational level for grade ${grade}, using very simple words and short sentences`;
  if (level <= 15) return `at the core of a grade ${grade} level`;
  if (level <= 30) return `at the upper end of a grade ${grade} level, introducing slightly more complex sentences and vocabulary`;
  if (level <= 50) return `at a level that slightly exceeds grade ${grade}, preparing them for the next grade level`;
  const nextGrade = Math.min(10, grade + 1);
  return `at a level suitable for grade ${nextGrade}, blending in more advanced content`;
};

const cleanWord = (word: string) => word.trim().toLowerCase().replace(/[.,?!]/g, '');

const getPhoneticBreakdown = async (word: string): Promise<string[]> => {
    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
        const systemInstruction = `You are a linguistic expert specializing in English phonetics for children. Your task is to break down a given word into its distinct phonetic syllables. 
**RULES:**
- The syllables should be simple, intuitive, and easy for a child to read and pronounce. For example, for "together", respond with ["tu", "geh", "dhuh"].
- Your response MUST be a single, valid JSON array of strings.
- Do NOT use markdown code fences or any other text outside the JSON array.
- If the word is a single syllable, return an array with that one word.`;

        const prompt = `Break down the word "${word}" into phonetic syllables.`;
        const responseSchema = { type: Type.ARRAY, items: { type: Type.STRING } };

        const response: GenerateContentResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: { systemInstruction, responseMimeType: 'application/json', responseSchema },
        });
        
        let jsonStr = response.text.trim();
        const firstBracket = jsonStr.indexOf('[');
        const lastBracket = jsonStr.lastIndexOf(']');
        if (firstBracket !== -1 && lastBracket > firstBracket) {
            jsonStr = jsonStr.substring(firstBracket, lastBracket + 1);
        }
        
        const result = JSON.parse(jsonStr);
        if (Array.isArray(result) && result.every(item => typeof item === 'string')) return result;
        return [word];
    } catch (e) {
        console.error(`Phonetic breakdown failed for "${word}":`, e);
        return [word];
    }
};

const analyzeReadingWithAI = async (spokenText: string, targetText: string): Promise<Mistake[]> => {
    if (!spokenText.trim()) return targetText.split(' ').map(word => ({ said: '', expected: word }));

    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
        const systemInstruction = `You are an expert English pronunciation analyst for children. Your task is to compare a 'target text' with a 'spoken text' from a speech-to-text service and identify mistakes. The 'spoken text' is from a child, so your analysis must be extremely lenient and encouraging.

**Core Principle: Focus on communication, not perfection.**

**Rules for Analysis:**
1.  **Extreme Leniency:** Be very forgiving. Accept pronunciations that are close enough to be understood. Do not penalize for common childhood speech patterns (e.g., "w" for "r").
2.  **Ignore Filler Words:** Completely disregard filler words like "um," "uh," "like," etc.
3.  **Handle Speech-to-Text Errors:** The transcription may be imperfect. If the spoken word sounds very similar to the target word (e.g., 'see' vs 'sea', 'two' vs 'to'), and makes sense, count it as correct.
4.  **Identify Key Mistakes Only:** Only flag a mistake if a word is:
    *   **Omitted:** The word was completely skipped. In this case, \`said\` should be \`""\`.
    *   **Mispronounced Unrecognizably:** The spoken word is completely different and cannot be understood as the target word.
    *   **Inserted:** An extra word was added that wasn't in the script. In this case, \`expected\` should be \`""\`.
5.  **No Mistakes is Good:** If the child's attempt is understandable, return an empty array \`[]\`.

**Response Format:**
- Your response MUST be a single, valid JSON array of objects.
- Each object must have two properties: "said" (string) and "expected" (string).
- Do NOT use markdown code fences or any other text.`;
        const prompt = `Target Text: "${targetText}"\nSpoken Text: "${spokenText}"`;
        const responseSchema = {
            type: Type.ARRAY,
            items: {
                type: Type.OBJECT,
                properties: { said: { type: Type.STRING }, expected: { type: Type.STRING } },
                required: ['said', 'expected'],
            },
        };

        const response: GenerateContentResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash',
            contents: prompt,
            config: { systemInstruction, responseMimeType: 'application/json', responseSchema },
        });
        
        let jsonStr = response.text.trim();
        const fenceRegex = /```(?:json)?\s*([\s\S]*?)\s*```/;
        const fenceMatch = jsonStr.match(fenceRegex);
        if (fenceMatch && fenceMatch[1]) jsonStr = fenceMatch[1].trim();
        else {
            const firstBracket = jsonStr.indexOf('[');
            const lastBracket = jsonStr.lastIndexOf(']');
            if (firstBracket !== -1 && lastBracket > firstBracket) {
                jsonStr = jsonStr.substring(firstBracket, lastBracket + 1);
            }
        }
        
        const result = JSON.parse(jsonStr);
        return Array.isArray(result) ? result : [];
    } catch (e) {
        console.error("AI analysis failed:", e);
        return []; 
    }
};

export const TalkersCaveGame: React.FC<TalkersCaveGameProps> = ({ onComplete, userGrade, currentLevel, onBackToGrades }) => {
  const [step, setStep] = useState<Step>('SCENE');
  const [selectedScene, setSelectedScene] = useState<Scene | null>(null);
  const [centeredScene, setCenteredScene] = useState<Scene>('Shopkeeper and Customer');
  const [selectedCharacter, setSelectedCharacter] = useState<string | null>(null);
  const [script, setScript] = useState<ScriptLine[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [currentTurn, setCurrentTurn] = useState(0);
  const [isAiSpeaking, setIsAiSpeaking] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [isRecognitionActive, setIsRecognitionActive] = useState(false);
  const [recognitionError, setRecognitionError] = useState<string | null>(null);
  const [matchedWordCount, setMatchedWordCount] = useState(0);
  const [mistakes, setMistakes] = useState<Mistake[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [practiceWords, setPracticeWords] = useState<PracticeWord[]>([]);
  const [currentPracticeWordIndex, setCurrentPracticeWordIndex] = useState(0);
  const [practiceStatus, setPracticeStatus] = useState<'IDLE' | 'LISTENING' | 'SUCCESS' | 'TRY_AGAIN'>('IDLE');
  const [practiceTranscript, setPracticeTranscript] = useState('');

  const speechTimeout = useRef<number | null>(null);
  const hasProcessedTurn = useRef(false);

  // --- Practice Mode Speech Recognition Refactoring ---
  const practiceRecognizer = useRef<SpeechRecognition | null>(null);
  const practiceResultHandler = useRef<(event: SpeechRecognitionEvent) => void>();

  useEffect(() => {
    practiceResultHandler.current = (event: SpeechRecognitionEvent) => {
        const transcript = event.results[0][0].transcript;
        setPracticeTranscript(transcript);

        if (practiceWords.length > 0 && currentPracticeWordIndex < practiceWords.length) {
            const targetWord = practiceWords[currentPracticeWordIndex].word;
            if (cleanWord(transcript).includes(cleanWord(targetWord))) {
                setPracticeStatus('SUCCESS');
                setTimeout(() => {
                    if (currentPracticeWordIndex < practiceWords.length - 1) {
                        setCurrentPracticeWordIndex(prev => prev + 1);
                        setPracticeStatus('IDLE');
                        setPracticeTranscript('');
                    } else {
                        onComplete();
                    }
                }, 1500);
            } else {
                setPracticeStatus('TRY_AGAIN');
            }
        }
    };
  }, [practiceWords, currentPracticeWordIndex, onComplete]);

  useEffect(() => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) return;
    
    practiceRecognizer.current = new SpeechRecognitionAPI();
    practiceRecognizer.current.lang = 'en-IN';
    practiceRecognizer.current.continuous = false;
    practiceRecognizer.current.interimResults = false;
    const recognizer = practiceRecognizer.current;

    const handleResult = (event: SpeechRecognitionEvent) => practiceResultHandler.current?.(event);
    const handleError = (event: SpeechRecognitionErrorEvent) => {
        if (event.error !== 'no-speech' && event.error !== 'aborted') {
            setRecognitionError(`Mic error: "${event.error}".`);
        }
        setPracticeStatus('IDLE');
    };
    const handleEnd = () => {
        setPracticeStatus(currentStatus => (currentStatus === 'LISTENING' ? 'IDLE' : currentStatus));
    };

    recognizer.addEventListener('result', handleResult as EventListener);
    recognizer.addEventListener('error', handleError as EventListener);
    recognizer.addEventListener('end', handleEnd);

    return () => {
        recognizer.removeEventListener('result', handleResult as EventListener);
        recognizer.removeEventListener('error', handleError as EventListener);
        recognizer.removeEventListener('end', handleEnd);
        try { recognizer.abort(); } catch(e) {}
    };
  }, []); // This effect runs only once to set up the recognizer instance.


  useEffect(() => {
    if (!window.speechSynthesis) return;
    const loadVoices = () => setVoices(window.speechSynthesis.getVoices());
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
    return () => { window.speechSynthesis.onvoiceschanged = null; };
  }, []);

  const speechRecognizer = useMemo<SpeechRecognition | null>(() => {
    const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognitionAPI) return null;
    const recognizer = new SpeechRecognitionAPI();
    recognizer.lang = 'en-IN';
    recognizer.maxAlternatives = 1;
    return recognizer;
  }, []);
  
  useEffect(() => {
    if (!speechRecognizer) setRecognitionError('Speech recognition is not supported in this browser. Try Chrome or Edge.');
  }, [speechRecognizer]);

  const processUserTurn = useCallback(async (transcript: string) => {
    if (hasProcessedTurn.current) return;
    hasProcessedTurn.current = true;
    if (speechRecognizer) try { speechRecognizer.stop(); } catch(e){}
    
    setIsAnalyzing(true);
    const targetLine = script[currentTurn].line;
    const currentMistakes = await analyzeReadingWithAI(transcript, targetLine);
    setIsAnalyzing(false);

    const allMistakes = [...mistakes, ...currentMistakes];
    setMistakes(allMistakes);
    
    if (currentTurn < script.length - 1) {
      setCurrentTurn(prev => prev + 1);
    } else {
      setStep(allMistakes.length > 0 ? 'PRACTICE_PREP' : 'COMPLETE');
    }
  }, [script, currentTurn, mistakes, speechRecognizer]);

  useEffect(() => {
    if (!speechRecognizer || step !== 'GAME') return;
    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;

    const handleResult = (event: SpeechRecognitionEvent) => {
      if (hasProcessedTurn.current) return;
      if (speechTimeout.current) clearTimeout(speechTimeout.current);

      let fullTranscript = '';
      for (let i = event.resultIndex; i < event.results.length; ++i) fullTranscript += event.results[i][0].transcript;
      setMatchedWordCount(fullTranscript.split(' ').map(cleanWord).filter(w => w).length);

      speechTimeout.current = window.setTimeout(() => {
        if (fullTranscript.trim()) processUserTurn(fullTranscript.trim());
      }, 2500);
    };
    const handleError = (event: SpeechRecognitionErrorEvent) => {
      if (speechTimeout.current) clearTimeout(speechTimeout.current);
      const errorType = event.error;
      if (['aborted', 'no-speech'].includes(errorType)) { setIsRecognitionActive(false); return; }
      if (errorType === 'not-allowed' || errorType === 'service-not-allowed') setRecognitionError('Microphone access denied.'); 
      else setRecognitionError(`Mic error: "${errorType}".`);
      setIsRecognitionActive(false);
    };
    const handleStart = () => setIsRecognitionActive(true);
    const handleEnd = () => setIsRecognitionActive(false);

    speechRecognizer.addEventListener('result', handleResult as EventListener);
    speechRecognizer.addEventListener('start', handleStart);
    speechRecognizer.addEventListener('end', handleEnd);
    speechRecognizer.addEventListener('error', handleError as EventListener);
    return () => {
      speechRecognizer.removeEventListener('result', handleResult as EventListener);
      speechRecognizer.removeEventListener('start', handleStart);
      speechRecognizer.removeEventListener('end', handleEnd);
      speechRecognizer.removeEventListener('error', handleError as EventListener);
      if (speechTimeout.current) clearTimeout(speechTimeout.current);
    };
  }, [speechRecognizer, step, processUserTurn]);
  
  const speak = useCallback((text: string, onEndCallback: () => void) => {
    if (!window.speechSynthesis || voices.length === 0) {
      setTimeout(onEndCallback, text.length * 50);
      return;
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const englishVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google')) || voices.find(v => v.lang.startsWith('en'));
    if (englishVoice) utterance.voice = englishVoice;
    utterance.onstart = () => setIsAiSpeaking(true);
    utterance.onend = () => { setIsAiSpeaking(false); onEndCallback(); };
    utterance.onerror = () => { setIsAiSpeaking(false); onEndCallback(); };
    window.speechSynthesis.speak(utterance);
  }, [voices]);

  useEffect(() => {
    if (step !== 'PRACTICE_PREP') return;
    const preparePractice = async () => {
        const wordsToPractice = [...new Set(mistakes.map(m => m.expected).filter(Boolean))];
        if (wordsToPractice.length === 0) { onComplete(); return; }
        const phoneticData = await Promise.all(wordsToPractice.map(async (word) => ({ word, phonemes: await getPhoneticBreakdown(word) })));
        setPracticeWords(phoneticData.filter(p => p.phonemes.length > 0));
        setCurrentPracticeWordIndex(0);
        setPracticeStatus('IDLE');
        setPracticeTranscript('');
        setStep('PRACTICE');
    };
    preparePractice();
  }, [step, mistakes, onComplete]);

  const startPracticeRecognition = useCallback(() => {
    const recognizer = practiceRecognizer.current;
    if (!recognizer || practiceStatus !== 'IDLE') {
        console.warn(`Recognition start blocked. Status: ${practiceStatus}`);
        return;
    }
    try {
        setPracticeStatus('LISTENING');
        setPracticeTranscript('');
        recognizer.start();
    } catch (e) {
        console.error("Could not start practice recognition:", e);
        setRecognitionError("Mic failed to start. Please try again.");
        setPracticeStatus('IDLE');
    }
  }, [practiceStatus]);

  const generateScript = useCallback(async (scene: Scene, character: string) => {
    setError(null); setStep('LOADING_SCRIPT');
    const aiCharacter = TALKERS_CAVE_SCENES[scene].find(c => c !== character);
    const difficulty = getDifficultyDescription(userGrade, currentLevel);
    const systemInstruction = `You are a script writing API for a kids' game. Your only purpose is to generate a conversation script for children learning English.
**REQUIREMENTS**
- The conversation should be natural, easy to follow, and exactly 8 turns long.
- The theme should be fun and engaging for kids.
- The AI character must speak first.
- CRITICAL: The difficulty must be ${difficulty}.
**RESPONSE FORMAT**
- Your entire response MUST be a single, valid JSON array.
- Do NOT include markdown code fences.
- The array must contain exactly 8 objects.
- Each object must have two string properties: "character" and "line".`;
    const prompt = `Generate a simple, 8-line script for kids for the scene "${scene}" with the user playing as "${character}" and the AI playing as "${aiCharacter}". The conversation's difficulty must be ${difficulty}.`;
    const scriptSchema = { type: Type.ARRAY, items: { type: Type.OBJECT, properties: { character: { type: Type.STRING }, line: { type: Type.STRING } }, required: ['character', 'line'] } };
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY as string });
      const response: GenerateContentResponse = await ai.models.generateContent({ model: 'gemini-2.5-flash', contents: prompt, config: { systemInstruction, responseMimeType: "application/json", responseSchema: scriptSchema } });
      let jsonStr = response.text.trim();
      const fenceRegex = /```(?:json)?\s*([\s\S]*?)\s*```/;
      const fenceMatch = jsonStr.match(fenceRegex);
      if (fenceMatch && fenceMatch[1]) jsonStr = fenceMatch[1].trim();
      else {
        const firstBracket = jsonStr.indexOf('[');
        const lastBracket = jsonStr.lastIndexOf(']');
        if (firstBracket !== -1 && lastBracket > firstBracket) jsonStr = jsonStr.substring(firstBracket, lastBracket + 1);
      }
      const parsedScript = JSON.parse(jsonStr);
      if (Array.isArray(parsedScript) && parsedScript.every(item => 'character' in item && 'line' in item)) { setScript(parsedScript); setStep('GAME'); setCurrentTurn(0); } 
      else { throw new Error('Received invalid script format from API.'); }
    } catch (e) {
      console.error(e); setError('Sorry, I couldn\'t create a script. Please try again.'); setStep('CHARACTER');
    }
  }, [userGrade, currentLevel]);

  const startRecognition = useCallback(() => {
    if (!speechRecognizer || isRecognitionActive) return;
    try {
        hasProcessedTurn.current = false;
        setMatchedWordCount(0);
        setRecognitionError(null);
        speechRecognizer.start();
    } catch(e: any) {
        if (e.name !== 'InvalidStateError') { console.error("Could not start recognition:", e); setRecognitionError("Failed to start microphone."); }
    }
  }, [speechRecognizer, isRecognitionActive]);

  useEffect(() => {
    if (step !== 'GAME' || !script.length || currentTurn >= script.length || recognitionError || isAiSpeaking) return;
    const currentLine = script[currentTurn];
    const isUserTurn = currentLine.character === selectedCharacter;
    
    if (isUserTurn) {
        const startMicTimeout = setTimeout(() => {
            startRecognition();
        }, 700); 
        return () => clearTimeout(startMicTimeout);
    } else {
        if (speechRecognizer) try { speechRecognizer.stop(); } catch (e) {}
        const handleAiTurnEnd = () => {
            if (currentTurn < script.length - 1) setCurrentTurn(prev => prev + 1);
            else setStep(mistakes.length > 0 ? 'PRACTICE_PREP' : 'COMPLETE');
        };
        const timeoutId = setTimeout(() => speak(currentLine.line, handleAiTurnEnd), 700);
        return () => clearTimeout(timeoutId);
    }
  }, [step, script, currentTurn, selectedCharacter, speak, speechRecognizer, recognitionError, isAiSpeaking, mistakes, startRecognition]);

  useEffect(() => () => {
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    if (speechRecognizer) speechRecognizer.abort();
    if (speechTimeout.current) clearTimeout(speechTimeout.current);
  }, [speechRecognizer]);

  const handleSceneSelect = (scene: Scene) => { setSelectedScene(scene); setStep('CHARACTER'); };
  const handleCharacterSelect = (character: string) => { setMistakes([]); setSelectedCharacter(character); generateScript(selectedScene!, character); };
  const handleBackToScenes = () => { setStep('SCENE'); setSelectedCharacter(null); setSelectedScene(null); setCenteredScene('Shopkeeper and Customer'); };

  const pronounceWord = (text: string) => {
    if (!window.speechSynthesis || voices.length === 0) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const englishVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google')) || voices.find(v => v.lang.startsWith('en'));
    if (englishVoice) utterance.voice = englishVoice;
    window.speechSynthesis.speak(utterance);
  };

  const renderTitle = () => {
    switch (step) {
      case 'SCENE': return 'Select Scene';
      case 'CHARACTER': return 'Select Character';
      case 'LOADING_SCRIPT': return 'Creating Your Story...';
      case 'GAME': return '';
      case 'COMPLETE': return 'Great Job!';
      case 'PRACTICE_PREP': return 'Getting Practice Ready...';
      case 'PRACTICE': return "Let's Practice!";
      default: return '';
    }
  };

  const renderContent = () => {
    switch (step) {
      case 'SCENE': {
        const scenes = Object.keys(TALKERS_CAVE_SCENES) as Scene[];
        return <div className="flex-grow flex flex-col justify-center items-center w-full h-full overflow-hidden">
            <div className="flex w-full h-full items-center justify-center gap-4 md:gap-8 px-4">
              {scenes.map((scene) => (
                  <button key={scene} onClick={() => handleSceneSelect(scene)} onMouseEnter={() => setCenteredScene(scene)} aria-label={`Select scene: ${scene}`}
                    className={`relative w-60 md:w-72 aspect-[16/10] flex-shrink-0 overflow-hidden rounded-2xl shadow-lg transition-all duration-500 ease-in-out transform group ${centeredScene === scene ? 'scale-110 opacity-100 shadow-cyan-500/40 z-10' : 'scale-90 opacity-60'} hover:!scale-110 hover:!opacity-100 hover:shadow-cyan-400/50`}>
                    <img src={TALKERS_CAVE_SCENE_IMAGES[scene]} alt={scene} className="absolute inset-0 w-full h-full object-cover transition-transform duration-300 group-hover:scale-105" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
                    <h3 className="absolute bottom-4 left-4 right-4 text-white font-bold text-lg md:text-xl text-left truncate" style={{ textShadow: '1px 1px 4px rgba(0,0,0,0.8)' }}>{scene}</h3>
                  </button>
              ))}
            </div>
        </div>;
      }
      case 'CHARACTER':
        if (!selectedScene) return null;
        return <div className="flex flex-col items-center justify-center h-full">
            {error && <p className="text-center text-red-400 mb-4">{error}</p>}
            <div className="flex justify-center items-end gap-4 md:gap-8 flex-wrap">
              {TALKERS_CAVE_SCENES[selectedScene].map((character) => (
                  <button key={character} onClick={() => handleCharacterSelect(character)} className="flex flex-col items-center gap-4 transition-transform transform hover:scale-105 group">
                    <div className="w-36 h-72 md:w-48 md:h-96"><img src={TALKERS_CAVE_CHARACTER_IMAGES[character]} alt={character} className="w-full h-full object-contain" /></div>
                    <span className="text-lg md:text-xl font-bold px-4 py-2 rounded-lg bg-indigo-600 group-hover:bg-indigo-500 transition-colors">{character}</span>
                  </button>
                ))}
            </div>
        </div>;

      case 'LOADING_SCRIPT': return <div className="text-center text-slate-300 animate-pulse text-2xl">Please wait...</div>;
      case 'PRACTICE_PREP': return <div className="text-center text-slate-300 animate-pulse text-2xl">Analyzing words for practice...</div>;

      case 'GAME': {
        if (!script.length || !selectedCharacter || !selectedScene) return null;
        const [characterOnLeft, characterOnRight] = TALKERS_CAVE_SCENES[selectedScene];
        const currentLine = script[currentTurn];
        const isLeftCharacterSpeaking = currentLine.character === characterOnLeft;
        const isUserTurn = currentLine.character === selectedCharacter;

        return (
          <div className='w-full h-full relative flex flex-col overflow-hidden'>
            <div className="flex-grow relative flex items-end justify-center px-4 overflow-hidden">
              <div className={`absolute bottom-0 left-0 md:left-[5%] w-1/2 md:w-2/5 h-2/3 md:h-4/5 transition-transform duration-500 ${isLeftCharacterSpeaking ? 'scale-110' : 'scale-100'}`}><img src={TALKERS_CAVE_CHARACTER_IMAGES[characterOnLeft]} alt={characterOnLeft} className="w-full h-full object-contain"/></div>
              <div className={`absolute bottom-0 right-0 md:right-[5%] w-1/2 md:w-2/5 h-2/3 md:h-4/5 transition-transform duration-500 ${!isLeftCharacterSpeaking ? 'scale-110' : 'scale-100'}`}><img src={TALKERS_CAVE_CHARACTER_IMAGES[characterOnRight]} alt={characterOnRight} className="w-full h-full object-contain"/></div>
              <div className={`absolute top-[8%] w-4/5 md:w-2/5 max-w-lg transition-all duration-300 ease-out ${!isLeftCharacterSpeaking ? 'right-[5%] md:right-[15%]' : 'left-[5%] md:left-[15%]'}`}>
                <div className={`relative bg-white text-slate-900 p-4 rounded-2xl shadow-2xl ${!isLeftCharacterSpeaking ? 'rounded-br-none' : 'rounded-bl-none'}`}>
                  {isUserTurn ? ( <p className="text-lg font-medium leading-relaxed">{currentLine.line.split(' ').map((word, wordIndex) => <span key={wordIndex} className={`transition-colors duration-200 ${wordIndex < matchedWordCount ? 'text-green-600 font-bold' : 'text-slate-800'}`}>{word}{' '}</span>)}</p>
                  ) : <p className="text-lg font-medium">{currentLine.line}</p>}
                  <div className={`absolute bottom-0 h-0 w-0 border-solid border-transparent border-t-white ${!isLeftCharacterSpeaking ? 'right-4 border-r-[15px] border-l-0 border-t-[15px] -mb-[15px]' : 'left-4 border-l-[15px] border-r-0 border-t-[15px] -mb-[15px]'}`}></div>
                </div>
              </div>
            </div>
            <div className="h-16 flex-shrink-0 bg-slate-900/50 flex items-center justify-center text-slate-300 relative">
              {recognitionError ? <p className="text-red-400 font-semibold">{recognitionError}</p> : (isUserTurn ? (
                <div className="flex items-center justify-center text-lg">
                  {isAnalyzing ? (
                    <p className="animate-pulse text-cyan-400">Analyzing your speech...</p>
                  ) : isRecognitionActive ? (
                    <div className="flex items-center gap-3">
                      <MicrophoneIcon className="text-cyan-400 animate-pulse w-6 h-6" />
                      <span className="text-cyan-400">Listening...</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-3">
                      <MicrophoneIcon className="text-slate-300 w-6 h-6" />
                      <span className="text-slate-300">Your turn to speak...</span>
                    </div>
                  )}
                </div>
              ) : <p className="text-lg animate-pulse">{isAiSpeaking ? 'AI is speaking...' : 'AI is thinking...'}</p>)}
            </div>
          </div>
        );
      }
      case 'PRACTICE': {
          if (practiceWords.length === 0 || currentPracticeWordIndex >= practiceWords.length) {
              return <div className="text-center flex flex-col items-center justify-center h-full p-4">
                  <p className="text-xl text-slate-300 mb-8">No words to practice. Well done!</p>
                  <button onClick={onComplete} className="px-8 py-4 bg-green-600 text-white font-bold rounded-lg text-xl hover:bg-green-700 transition-transform transform hover:scale-105">Finish</button>
              </div>;
          }
          const practiceItem = practiceWords[currentPracticeWordIndex];
          const getStatusMessage = () => {
              switch (practiceStatus) {
                  case 'LISTENING': return <p className="text-cyan-400 animate-pulse">Listening...</p>;
                  case 'SUCCESS': return <p className="text-green-400 font-bold">Great job!</p>;
                  case 'TRY_AGAIN': return <p className="text-red-400">Not quite. You said: <span className="font-bold">{practiceTranscript}</span>. Try again!</p>;
                  default: return <p className="text-slate-300">Click the mic and say the word.</p>;
              }
          };
          return <div className="w-full h-full flex flex-col items-center justify-center p-4 sm:p-8 animate-fade-in">
              <div className="w-full max-w-2xl text-center">
                  <div className="flex justify-center flex-wrap gap-4 mb-12">
                      {practiceItem.phonemes.map((phoneme, index) => (
                          <div key={index} className="bg-white rounded-2xl p-4 flex flex-col items-center justify-between gap-4 shadow-lg w-32">
                              <span className="text-purple-600 font-bold text-4xl sm:text-5xl" style={{minHeight: '48px'}}>{phoneme}</span>
                              <button onClick={() => pronounceWord(phoneme)} className="p-1" aria-label={`Listen to ${phoneme}`}>
                                  <PracticeSoundIcon />
                              </button>
                          </div>
                      ))}
                  </div>

                  <div className="flex flex-col items-center gap-4">
                      <button onClick={startPracticeRecognition} disabled={practiceStatus !== 'IDLE'}
                          className={`w-28 h-28 sm:w-32 sm:h-32 rounded-full flex items-center justify-center transition-all duration-300 transform 
                              ${practiceStatus === 'LISTENING' ? 'bg-cyan-500 scale-110 animate-pulse' : 'bg-purple-600 hover:bg-purple-500 hover:scale-105'}
                              disabled:bg-slate-500 disabled:scale-100 disabled:cursor-not-allowed`}>
                          <MicrophoneIcon className="w-12 h-12 sm:w-16 sm:h-16 text-white" />
                      </button>
                      <div className="h-8 text-xl mt-2">{getStatusMessage()}</div>
                  </div>
              </div>
              <button onClick={onComplete} className="absolute bottom-6 right-6 text-slate-400 hover:text-white font-semibold transition-colors bg-black/30 px-4 py-2 rounded-lg">
                  Finish Practice
              </button>
          </div>
      }
      case 'COMPLETE': return <div className="text-center flex flex-col items-center justify-center h-full p-4">
            <p className="text-3xl sm:text-4xl text-green-400 mb-2">Perfect!</p>
            <p className="text-lg sm:text-xl text-slate-300 mb-8">You said everything correctly. Great job!</p>
            <button onClick={() => onComplete()} className="px-6 py-3 sm:px-8 sm:py-4 bg-green-600 text-white font-bold rounded-lg text-xl sm:text-2xl w-full max-w-xs hover:bg-green-700 transition-transform transform hover:scale-105">Finish</button>
        </div>;
      default: return null;
    }
  };

  return (
    <div className="w-full h-full text-white relative flex flex-col justify-center animate-fade-in">
        <div className="absolute top-4 sm:top-6 left-1/2 -translate-x-1/2 w-full px-4 text-center z-20">
             <h1 className="text-3xl sm:text-5xl font-bold text-cyan-400" style={{textShadow: '2px 2px 8px rgba(0,0,0,0.7)'}}>{renderTitle()}</h1>
        </div>
        {step !== 'GAME' && <div className="absolute top-4 right-4 sm:top-6 sm:right-6 bg-slate-900/70 px-4 py-2 rounded-lg text-base sm:text-lg font-bold text-cyan-300 z-20 backdrop-blur-sm">
          Level: {currentLevel}
        </div>}
        {(step === 'SCENE' || step === 'CHARACTER') && (
            <button
                onClick={step === 'SCENE' ? onBackToGrades : handleBackToScenes}
                className="absolute top-4 left-4 sm:top-6 sm:left-6 text-slate-300 hover:text-white transition-colors z-20 font-bold flex items-center gap-2 text-sm sm:text-base"
            >
                <span className="text-xl sm:text-2xl">&larr;</span> {step === 'SCENE' ? 'Back to Grades' : 'Back'}
            </button>
        )}
        <div className="flex-grow flex flex-col justify-center overflow-hidden pt-20 sm:pt-24">
            {renderContent()}
        </div>
    </div>
  );
};