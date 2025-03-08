"use client"

import { motion } from "framer-motion"

type LevelSelectorProps = {
  levels: string[]
  currentLevel: number
  onSelectLevel: (index: number) => void
}

export default function LevelSelector({ levels, currentLevel, onSelectLevel }: LevelSelectorProps) {
  return (
    <div className="flex flex-wrap gap-3 justify-center">
      {levels.map((level, index) => (
        <motion.button
          key={index}
          onClick={() => onSelectLevel(index)}
          className={`px-5 py-3 rounded-full text-sm font-medium transition-all duration-300 relative overflow-hidden ${
            currentLevel === index ? "text-white" : "bg-white text-green-800 hover:bg-green-50 shadow-md"
          }`}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          {currentLevel === index && (
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-green-500 to-blue-500 -z-10"
              layoutId="activeLevel"
              transition={{ type: "spring", stiffness: 300, damping: 30 }}
            />
          )}
          Level {index + 1}: {level}
        </motion.button>
      ))}
    </div>
  )
}

