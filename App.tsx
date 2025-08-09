import React from 'react';
import { TalkersCaveGame } from './components/TalkersCaveGame';
import { GradeSelectionScreen } from './components/GradeSelectionScreen';

export const App: React.FC = () => {
  const [userGrade, setUserGrade] = React.useState<number | null>(null);
  const [currentLevel, setCurrentLevel] = React.useState(0);
  const [gameKey, setGameKey] = React.useState(0);
  const [isFinished, setIsFinished] = React.useState(false);

  const handleSetGrade = (grade: number) => {
    setUserGrade(grade);
    setCurrentLevel(0);
    setGameKey(0);
    setIsFinished(false);
  };

  const handleBackToGrades = () => {
    setUserGrade(null);
  };

  const handleFinish = () => {
    setCurrentLevel(prevLevel => prevLevel + 1);
    setIsFinished(true);
  };

  const handlePlayAgain = () => {
    setIsFinished(false);
    setGameKey(prevKey => prevKey + 1);
  };

  return (
    <main className="h-screen w-screen text-white font-sans flex flex-col select-none relative">
      <div className="flex-1 flex flex-col overflow-hidden">
        {!userGrade ? (
          <GradeSelectionScreen onGradeSelect={handleSetGrade} />
        ) : (
          <div className="relative flex-1 overflow-hidden">
            {isFinished ? (
              <div className="absolute inset-0 backdrop-blur-sm flex flex-col justify-center items-center z-10 animate-fade-in text-center p-4">
                <h2 className="text-4xl sm:text-5xl font-bold text-cyan-400 mb-2" style={{textShadow: '2px 2px 8px rgba(0,0,0,0.7)'}}>
                  Level {currentLevel - 1} Complete!
                </h2>
                <p className="text-xl sm:text-2xl text-white mb-8" style={{textShadow: '2px 2px 8px rgba(0,0,0,0.7)'}}>Next Level: {currentLevel}</p>
                <button
                  onClick={handlePlayAgain}
                  className="px-8 py-4 bg-cyan-500 text-white font-bold rounded-lg text-xl sm:text-2xl hover:bg-cyan-600 transition-transform transform hover:scale-105"
                >
                  Start Level {currentLevel}
                </button>
              </div>
            ) : (
              <TalkersCaveGame
                key={gameKey}
                onComplete={handleFinish}
                userGrade={userGrade}
                currentLevel={currentLevel}
                onBackToGrades={handleBackToGrades}
              />
            )}
          </div>
        )}
      </div>
    </main>
  );
};