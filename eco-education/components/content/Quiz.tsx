"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { CheckCircle, XCircle } from "lucide-react"

type Question = {
  question: string
  options: string[]
  correctAnswer: number
}

type QuizProps = {
  questions: Question[]
  onComplete: (score: number) => void
}

export default function Quiz({ questions, onComplete }: QuizProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [selectedOption, setSelectedOption] = useState<number | null>(null)
  const [score, setScore] = useState(0)
  const [showFeedback, setShowFeedback] = useState(false)
  const [isCorrect, setIsCorrect] = useState(false)
  const [timeLeft, setTimeLeft] = useState(30)
  const [isTimerActive, setIsTimerActive] = useState(true)

  // Timer effect
  useEffect(() => {
    if (!isTimerActive) return

    if (timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000)
      return () => clearTimeout(timer)
    } else {
      // Time's up, move to next question
      handleNext()
    }
  }, [timeLeft, isTimerActive])

  const handleOptionSelect = (index: number) => {
    setSelectedOption(index)
    setIsTimerActive(false)
  }

  const handleNext = () => {
    // Check if answer is correct
    const correct = selectedOption === questions[currentQuestion].correctAnswer
    if (correct) {
      setScore(score + 1)
    }

    setIsCorrect(correct)
    setShowFeedback(true)
    setIsTimerActive(false)

    // Move to next question after a delay
    setTimeout(() => {
      setShowFeedback(false)
      setSelectedOption(null)

      if (currentQuestion < questions.length - 1) {
        setCurrentQuestion(currentQuestion + 1)
        setTimeLeft(30)
        setIsTimerActive(true)
      } else {
        onComplete(correct ? score + 1 : score)
      }
    }, 1500)
  }

  // Calculate progress percentage
  const progressPercentage = ((currentQuestion + 1) / questions.length) * 100

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 animate-fadeIn">
      {/* Progress bar */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-600 mb-2">
          <span>
            Question {currentQuestion + 1} of {questions.length}
          </span>
          <span className={`font-medium ${timeLeft < 10 ? "text-red-500" : "text-gray-600"}`}>
            Time left: {timeLeft}s
          </span>
        </div>
        <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-green-500 to-blue-500"
            initial={{ width: `${(currentQuestion / questions.length) * 100}%` }}
            animate={{ width: `${progressPercentage}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={currentQuestion}
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          transition={{ duration: 0.3 }}
        >
          <h3 className="text-xl font-semibold text-gray-800 mb-6">{questions[currentQuestion].question}</h3>

          <div className="space-y-3 mb-6">
            {questions[currentQuestion].options.map((option, index) => (
              <motion.button
                key={index}
                onClick={() => handleOptionSelect(index)}
                disabled={showFeedback}
                className={`w-full text-left p-4 rounded-lg transition-all duration-200 relative ${
                  selectedOption === index
                    ? "border-2 border-green-500 bg-green-50"
                    : "border border-gray-200 bg-white hover:border-green-300 hover:bg-green-50"
                } ${
                  showFeedback && index === questions[currentQuestion].correctAnswer
                    ? "border-2 border-green-500 bg-green-50"
                    : showFeedback && index === selectedOption && index !== questions[currentQuestion].correctAnswer
                      ? "border-2 border-red-400 bg-red-50"
                      : ""
                }`}
                whileHover={{ scale: showFeedback ? 1 : 1.02 }}
                whileTap={{ scale: showFeedback ? 1 : 0.98 }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="flex items-center">
                  <div
                    className={`w-6 h-6 rounded-full mr-3 flex items-center justify-center ${
                      selectedOption === index ? "bg-green-500 text-white" : "bg-gray-200 text-gray-700"
                    }`}
                  >
                    {String.fromCharCode(65 + index)}
                  </div>
                  <span>{option}</span>
                </div>

                {showFeedback && index === questions[currentQuestion].correctAnswer && (
                  <CheckCircle className="absolute right-4 top-1/2 transform -translate-y-1/2 text-green-500 h-6 w-6" />
                )}

                {showFeedback && index === selectedOption && index !== questions[currentQuestion].correctAnswer && (
                  <XCircle className="absolute right-4 top-1/2 transform -translate-y-1/2 text-red-500 h-6 w-6" />
                )}
              </motion.button>
            ))}
          </div>
        </motion.div>
      </AnimatePresence>

      {showFeedback && (
        <motion.div
          className={`p-4 rounded-lg mb-6 ${isCorrect ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"}`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center">
            {isCorrect ? <CheckCircle className="h-5 w-5 mr-2" /> : <XCircle className="h-5 w-5 mr-2" />}
            {isCorrect ? "Correct!" : "Incorrect. The correct answer is highlighted."}
          </div>
        </motion.div>
      )}

      <div className="flex justify-center">
        <motion.button
          onClick={handleNext}
          disabled={selectedOption === null || showFeedback}
          className={`gradient-btn text-white font-bold py-3 px-8 rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 ${
            selectedOption === null ? "opacity-50 cursor-not-allowed" : ""
          } ${showFeedback ? "hidden" : ""}`}
          whileHover={{ scale: selectedOption === null ? 1 : 1.05 }}
          whileTap={{ scale: selectedOption === null ? 1 : 0.95 }}
        >
          {currentQuestion < questions.length - 1 ? "Next Question" : "Finish Quiz"}
        </motion.button>
      </div>
    </div>
  )
}

