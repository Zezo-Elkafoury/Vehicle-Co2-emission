"use client"

import { motion } from "framer-motion"

type ContentTextProps = {
  content: string
}

export default function ContentText({ content }: ContentTextProps) {
  // Split content into paragraphs
  const paragraphs = content.split(". ").map((p) => p.trim() + ".")

  return (
    <div className="prose prose-green max-w-none">
      {paragraphs.map((paragraph, index) => (
        <motion.p
          key={index}
          className="text-gray-700 leading-relaxed mb-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.2 }}
        >
          {paragraph}
        </motion.p>
      ))}
    </div>
  )
}

