"use client"

import { useState } from "react"
import LevelSelector from "@/components/content/LevelSelector"
import VideoPlayer from "@/components/content/VideoPlayer"
import ContentText from "@/components/content/ContentText"
import Quiz from "@/components/content/Quiz"
import ScrollReveal from "@/components/ScrollReveal"
import AIAssistantPopup from "@/components/AiAssistantPopup" // Import the new component

// Sample content data - in a real app, this might come from an API or CMS
const contentLevels = [
  {
    id: 1,
    title: "Introduction to Climate Change",
    videoUrl: "https://www.youtube.com/embed/G4H1N_yXBiA",
    content:
      "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil, and gas, which produces heat-trapping gases.",
    quiz: [
      {
        question: "What is the main driver of climate change since the 1800s?",
        options: ["Natural weather patterns", "Human activities", "Solar flares", "Ocean currents"],
        correctAnswer: 1,
      },
      {
        question: "Which of the following is a major contributor to human-caused climate change?",
        options: ["Planting trees", "Burning fossil fuels", "Using renewable energy", "Recycling"],
        correctAnswer: 1,
      },
      {
        question: "What do greenhouse gases do in our atmosphere?",
        options: ["Cool the planet", "Have no effect", "Trap heat", "Filter out radiation"],
        correctAnswer: 2,
      },
    ],
  },
  {
    id: 2,
    title: "Renewable Energy Solutions",
    videoUrl: "https://www.youtube.com/embed/1kUE0BZtTRc",
    content:
      "Renewable energy comes from sources that are naturally replenishing but flow-limited. They are virtually inexhaustible in duration but limited in the amount of energy that is available per unit of time. Renewable energy sources include biomass, hydropower, geothermal, wind, and solar.",
    quiz: [
      {
        question: "Which of the following is NOT a renewable energy source?",
        options: ["Solar", "Wind", "Natural Gas", "Hydropower"],
        correctAnswer: 2,
      },
      {
        question: "What is a key advantage of renewable energy?",
        options: ["It's more expensive", "It's inexhaustible", "It requires more land", "It produces more pollution"],
        correctAnswer: 1,
      },
      {
        question: "Which renewable energy source harnesses the power of moving water?",
        options: ["Geothermal", "Biomass", "Solar", "Hydropower"],
        correctAnswer: 3,
      },
    ],
  },
  {
    id: 3,
    title: "Sustainable Living Practices",
    videoUrl: "https://www.youtube.com/embed/kZIrIQDf1nQ",
    content:
      "Sustainable living is a lifestyle that attempts to reduce an individual's or society's use of the Earth's natural resources. Practitioners of sustainable living often attempt to reduce their carbon footprint by altering methods of transportation, energy consumption, and diet.",
    quiz: [
      {
        question: "What is a key principle of sustainable living?",
        options: ["Using more resources", "Reducing waste", "Increasing consumption", "Ignoring environmental impact"],
        correctAnswer: 1,
      },
      {
        question: "Which of the following is a sustainable practice?",
        options: ["Single-use plastics", "Fast fashion", "Composting", "Long commutes by car"],
        correctAnswer: 2,
      },
      {
        question: "How can diet choices impact sustainability?",
        options: [
          "They have no impact",
          "Plant-based diets typically have lower environmental impacts",
          "More meat consumption is always better",
          "Food choices don't affect carbon footprint",
        ],
        correctAnswer: 1,
      },
    ],
  },
]

// Sample user data - in a real app, this would come from your user context
const sampleUserData = {
  impact: {
    co2Saved: 45.8,
    wasteSaved: 12.3,
    energySaved: 30.5,
    waterSaved: 150.2
  },
  logs: [
    { category: "transport", impact: { co2Saved: 12.5, wasteSaved: 0 } },
    { category: "energy", impact: { co2Saved: 8.3, wasteSaved: 0 } },
    { category: "waste", impact: { co2Saved: 5.2, wasteSaved: 3.4 } },
    { category: "transport", impact: { co2Saved: 19.8, wasteSaved: 0 } }
  ],
  points: 450
};

export default function ContentPage() {
  const [currentLevel, setCurrentLevel] = useState(0)
  const [quizStarted, setQuizStarted] = useState(false)
  const [quizCompleted, setQuizCompleted] = useState(false)
  const [score, setScore] = useState(0)
  const [animateContent, setAnimateContent] = useState(true)
  const [userData, setUserData] = useState(sampleUserData) // Add user data state

  const handleLevelSelect = (index: number) => {
    setAnimateContent(false)
    setTimeout(() => {
      setCurrentLevel(index)
      setQuizStarted(false)
      setQuizCompleted(false)
      setScore(0)
      setAnimateContent(true)
    }, 300)
  }

  const handleStartQuiz = () => {
    setAnimateContent(false)
    setTimeout(() => {
      setQuizStarted(true)
      setAnimateContent(true)
    }, 300)
  }

  const handleQuizComplete = (finalScore: number) => {
    setScore(finalScore)
    setQuizCompleted(true)
    setQuizStarted(false)
  }

  return (
    <div className="container mx-auto px-4 py-24">
      {/* Add the AI Assistant Popup component */}
      <AIAssistantPopup contentLevels={contentLevels} />

      <ScrollReveal>
        <h1 className="text-4xl md:text-5xl font-bold gradient-text mb-12 text-center">Educational Content</h1>
      </ScrollReveal>

      <ScrollReveal delay={200}>
        <LevelSelector
          levels={contentLevels.map((level) => level.title)}
          currentLevel={currentLevel}
          onSelectLevel={handleLevelSelect}
        />
      </ScrollReveal>

      <div
        className={`bg-white rounded-xl shadow-xl p-8 mt-12 transition-all duration-300 ${animateContent ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}`}
      >
        <h2 className="text-2xl md:text-3xl font-semibold gradient-text mb-6">{contentLevels[currentLevel].title}</h2>

        {!quizStarted && !quizCompleted && (
          <>
            <VideoPlayer videoUrl={contentLevels[currentLevel].videoUrl} />
            <ContentText content={contentLevels[currentLevel].content} />
            <div className="mt-8 text-center">
              <button
                onClick={handleStartQuiz}
                className="gradient-btn text-white font-bold py-3 px-8 rounded-full transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50"
              >
                Take Quiz
              </button>
            </div>
          </>
        )}

        {quizStarted && <Quiz questions={contentLevels[currentLevel].quiz} onComplete={handleQuizComplete} />}

        {quizCompleted && (
          <div className="text-center py-8 animate-fadeIn">
            <div className="mb-8 relative">
              <div className="w-32 h-32 mx-auto bg-green-50 rounded-full flex items-center justify-center">
                <div className="text-4xl font-bold gradient-text">
                  {score}/{contentLevels[currentLevel].quiz.length}
                </div>
              </div>
              <svg
                className="absolute top-0 left-1/2 transform -translate-x-1/2"
                width="140"
                height="140"
                viewBox="0 0 120 120"
              >
                <circle cx="60" cy="60" r="54" fill="none" stroke="#e6e6e6" strokeWidth="12" />
                <circle
                  cx="60"
                  cy="60"
                  r="54"
                  fill="none"
                  stroke="#3a9d6e"
                  strokeWidth="12"
                  strokeDasharray="339.3"
                  strokeDashoffset={339.3 - (339.3 * score) / contentLevels[currentLevel].quiz.length}
                  className="transform -rotate-90 origin-center transition-all duration-1000 ease-out"
                />
              </svg>
            </div>

            <h3 className="text-2xl font-semibold mb-4">Quiz Completed!</h3>
            <p className="text-lg mb-6 text-gray-700">
              Your score: {score} out of {contentLevels[currentLevel].quiz.length}
            </p>
            <button
              onClick={() => {
                setAnimateContent(false)
                setTimeout(() => {
                  setQuizCompleted(false)
                  if (currentLevel < contentLevels.length - 1) {
                    setCurrentLevel(currentLevel + 1)
                  }
                  setAnimateContent(true)
                }, 300)
              }}
              className="gradient-btn text-white font-bold py-3 px-8 rounded-full transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50"
            >
              {currentLevel < contentLevels.length - 1 ? "Next Level" : "Review Content"}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}