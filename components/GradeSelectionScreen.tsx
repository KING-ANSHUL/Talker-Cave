import React from 'react';

interface GradeSelectionScreenProps {
  onGradeSelect: (grade: number) => void;
}

export const GradeSelectionScreen: React.FC<GradeSelectionScreenProps> = ({ onGradeSelect }) => {
  const grades = Array.from({ length: 10 }, (_, i) => i + 1);

  return (
    <div className="w-full h-full text-white flex flex-col items-center justify-center p-8 animate-fade-in text-center">
      <h2 className="text-2xl sm:text-3xl font-semibold text-slate-200 mb-12">
        Please select your grade to begin.
      </h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4 sm:gap-6">
        {grades.map((grade) => (
          <button
            key={grade}
            onClick={() => onGradeSelect(grade)}
            className="w-24 h-24 sm:w-28 sm:h-28 bg-slate-800/70 rounded-2xl flex flex-col items-center justify-center text-slate-200 hover:bg-cyan-600 hover:text-white hover:scale-110 transition-all duration-300 shadow-lg backdrop-blur-sm"
          >
            <span className="text-sm font-semibold">Grade</span>
            <span className="text-4xl font-bold">{grade}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
