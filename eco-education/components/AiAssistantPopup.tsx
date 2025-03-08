"use client"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Brain, Send, Loader2, X } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Avatar } from "@/components/ui/avatar"
import { AvatarFallback } from "@/components/ui/avatar"
import { motion, AnimatePresence } from "framer-motion"
import { GoogleGenerativeAI } from "@google/generative-ai"

// Initialize Gemini API client
const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY || "AIzaSyDfXK-s7Fvhe4ZpYfTm_Nb2ExaFAHn-1MY");

export default function AIAssistantPopup() {
  const [isOpen, setIsOpen] = useState(false)
  const [aiPrompt, setAiPrompt] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [chatHistory, setChatHistory] = useState([])

  // Function to get AI response
  const getAiResponse = async () => {
    if (!aiPrompt.trim()) return

    setIsLoading(true)
    setChatHistory(prev => [...prev, { type: 'user', message: aiPrompt }])

    try {
      const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" })
      const enhancedPrompt = `
      System Instruction: You are an AI assistant created for the EcoLearn Platform. 
      You are proficient in all topics related to environment, climate, and sustainability.
      Your responses should be informative, concise, and focused on environmental education.
      Do not use markdown marks or any symbols

      User Query: "${aiPrompt}"
    `;
      const result = await model.generateContent(enhancedPrompt)
      const responseText = result.response.text()

      setChatHistory(prev => [...prev, { type: 'assistant', message: responseText }])
    } catch (error) {
      console.error('Error:', error)
      setChatHistory(prev => [...prev, { type: 'assistant', message: "Sorry, I couldn't process your request. Please try again." }])
    } finally {
      setIsLoading(false)
      setAiPrompt("")
    }
  }

  return (
    <>
      {/* Floating action button */}
      <motion.div 
        className="fixed bottom-6 right-6 z-50"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Button
          onClick={() => setIsOpen(true)}
          className={`h-14 w-14 rounded-full bg-green-600 shadow-lg transition-all duration-300 hover:shadow-xl ${isOpen ? 'scale-0' : 'scale-100'}`}
        >
          <Brain className="h-6 w-6 text-white" />
        </Button>
      </motion.div>

      {/* Popup modal */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed bottom-24 right-6 w-80 md:w-96 bg-white rounded-xl shadow-2xl z-50 overflow-hidden"
            initial={{ y: 20, opacity: 0, height: 0 }}
            animate={{ y: 0, opacity: 1, height: 'auto' }}
            exit={{ y: 20, opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            {/* Header */}
            <div className="bg-green-600 p-4 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Avatar className="h-8 w-8 bg-white/20">
                  <AvatarFallback>
                    <Brain className="h-4 w-4 text-white" />
                  </AvatarFallback>
                </Avatar>
                <h3 className="font-medium text-white">AI Assistant</h3>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)} className="text-white hover:bg-white/20">
                <X className="h-5 w-5" />
              </Button>
            </div>

            {/* Chat content */}
            <div className="p-4 h-72 overflow-y-auto flex flex-col gap-3">
              {chatHistory.length === 0 ? (
                <div className="text-center text-gray-500">Ask me anything about sustainability or enviroment!</div>
              ) : (
                chatHistory.map((chat, index) => (
                  <div key={index} className={`flex ${chat.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`rounded-lg p-3 max-w-[85%] ${chat.type === 'user' ? 'bg-green-100' : 'bg-gray-100'}`}>
                      {chat.message}
                    </div>
                  </div>
                ))
              )}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="rounded-lg p-3 bg-gray-100">
                    <Loader2 className="h-5 w-5 animate-spin" />
                  </div>
                </div>
              )}
            </div>

            {/* Input area */}
            <div className="p-3 border-t flex gap-2">
              <Input
                placeholder="Ask something..."
                value={aiPrompt}
                onChange={(e) => setAiPrompt(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && getAiResponse()}
                className="flex-1"
              />
              <Button onClick={getAiResponse} disabled={isLoading || !aiPrompt.trim()} className="bg-green-600 hover:bg-green-700">
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Background overlay */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="fixed inset-0 bg-black/10 z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsOpen(false)}
          />
        )}
      </AnimatePresence>
    </>
  )
}
