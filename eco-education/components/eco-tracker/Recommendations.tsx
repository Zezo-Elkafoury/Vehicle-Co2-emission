"use client"
import { useState, useEffect, useMemo } from "react"
import { useUser } from "./UserContext"
import { 
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import {
  Brain,
  Leaf,
  ThumbsUp,
  TrendingUpIcon as Trending,
  Sparkles,
  AlertTriangle,
  Send,
  Loader2
} from "lucide-react"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { GoogleGenerativeAI } from "@google/generative-ai"

import { categoryIcons, getCategoryColor } from "@/lib/eco-tracker/utils"
import { EcoCard } from "./EcoCard"
import { motion } from "framer-motion"

// Initialize Gemini API client
const genAI = new GoogleGenerativeAI(process.env.NEXT_PUBLIC_GEMINI_API_KEY || "AIzaSyDfXK-s7Fvhe4ZpYfTm_Nb2ExaFAHn-1MY");

export default function Recommendations() {
  const { userData, onAddAction } = useUser()
  const [update, setUpdate] = useState(false)
  const [aiPrompt, setAiPrompt] = useState("")
  const [aiResponse, setAiResponse] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  // Trigger re-render when userData changes
  useEffect(() => {
    setUpdate((prev) => !prev)
  }, [userData.logs]) // Specifically track logs.length to detect changes

  // Function to get AI recommendations directly from Gemini
  const getAiRecommendation = async () => {
    if (!aiPrompt.trim()) return

    setIsLoading(true)
    try {
      // Create a context-rich prompt that includes user data
      const enhancedPrompt = `
       Here's my current data:
        - Total CO2 saved: ${userData.impact.co2Saved.toFixed(2)} kg
        - Total waste saved: ${userData.impact.wasteSaved.toFixed(2)} kg
        - Most active in: ${categoryAnalysis.mostFrequentCategory}
        - Total eco-actions logged: ${userData.logs.length}
        
        My question: ${aiPrompt}
        You are an Ai specialized in Enviroment saving ,greenery and climate change and related things
        -**Dont use markdown language or ** or any signs**
        -You might output insights and tips related to enviroment and saving it
        
        
       
      `

      // Call Gemini API directly
      const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
      const result = await model.generateContent(enhancedPrompt);
      const responseText = result.response.text();

      setAiResponse(responseText);
    } catch (error) {
      console.error('Error getting AI recommendation:', error)
      setAiResponse("Sorry, I couldn't get a recommendation at this time. Please try again later.")
    } finally {
      setIsLoading(false)
    }
  }

  // Rest of your component code remains the same
  // Calculate impact by category using useMemo
  const categoryAnalysis = useMemo(() => {
    // Calculate impact by category
    const categoryImpacts: Record<
      string,
      {
        count: number
        co2Saved: number
        wasteSaved: number
        energySaved: number
        waterSaved: number
      }
    > = {}

    userData.logs.forEach((log) => {
      if (!categoryImpacts[log.category]) {
        categoryImpacts[log.category] = {
          count: 0,
          co2Saved: 0,
          wasteSaved: 0,
          energySaved: 0,
          waterSaved: 0,
        }
      }
      categoryImpacts[log.category].count++
      categoryImpacts[log.category].co2Saved += log.impact.co2Saved
      categoryImpacts[log.category].wasteSaved += log.impact.wasteSaved
      categoryImpacts[log.category].energySaved += log.impact.energySaved
      categoryImpacts[log.category].waterSaved += log.impact.waterSaved
    })

    // Sort categories by total impact (using CO2 as the primary metric)
    const sortedCategories = Object.entries(categoryImpacts)
      .sort((a, b) => b[1].co2Saved - a[1].co2Saved)
      .map((entry) => ({
        category: entry[0],
        ...entry[1],
      }))

    // Get top categories
    const topCategories = sortedCategories.slice(0, 2)

    // Calculate most frequent category
    const categoryCounts: Record<string, number> = {}
    userData.logs.forEach((log) => {
      categoryCounts[log.category] = (categoryCounts[log.category] || 0) + 1
    })

    const sortedByFrequency = Object.entries(categoryCounts)
      .sort((a, b) => b[1] - a[1])
      .map((entry) => entry[0])

    const mostFrequentCategory = sortedByFrequency[0] || "recycling"

    // Find a category with low activity to suggest
    const allCategories = ["transport", "energy", "waste", "water", "food"]
    const lowActivityCategories = allCategories.filter(
      (cat) => !categoryCounts[cat] || categoryCounts[cat] < 2
    )

    const suggestedCategory =
      lowActivityCategories[0] ||
      (sortedByFrequency.length > 1 ? sortedByFrequency[sortedByFrequency.length - 1] : "energy conservation")

    return {
      topCategories,
      mostFrequentCategory,
      suggestedCategory,
    }
  }, [userData.logs])

  // Generate personalized recommendations based on user data
  const recommendations = useMemo(() => {
    // Find most common categories in user logs
    const categoryCounts: Record<string, number> = {}
    userData.logs.forEach((log) => {
      if (categoryCounts[log.category]) {
        categoryCounts[log.category]++
      } else {
        categoryCounts[log.category] = 1
      }
    })

    // Sort categories by count
    const sortedCategories = Object.entries(categoryCounts)
    .sort((a, b) => b[1] - a[1])
    .map((entry) => entry[0])

    // Find actions in top categories that user hasn't logged recently
    const topCategory = sortedCategories[0] || "transport"
    const recentActionIds = new Set(
      userData.logs.slice(0, 20).map((log) => log.actionId)
    )

    const recommendedActions = userData.availableActions.filter(
      (action) => action.category === topCategory && !recentActionIds.has(action.id)
    )
      .filter((action) => action.category === topCategory && !recentActionIds.has(action.id))
      .slice(0, 3)

    return recommendedActions
  }, [userData.logs, userData.availableActions])

  // Generate "because you..." messages
  const getBecauseYouMessage = (category: string) => {
    switch (category) {
      case "transport":
        return "Because you logged eco-friendly transport actions"
      case "energy":
        return "Because you care about energy conservation"
      case "waste":
        return "Because you track waste reduction"
      case "water":
        return "Because you log water conservation actions"
      case "food":
        return "Because you track sustainable food choices"
      default:
        return "Based on your activity"
    }
  }

  // Handle key press in the input field
  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      getAiRecommendation()
    }
  }

  // Add a key prop to force re-render when userData changes
  return (
    <div className="space-y-6" key={`recommendations-${userData.logs.length}`}>
      <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold gradient-text mb-2">
            Smart Insights & Recommendations
          </h2>
          <p className="text-gray-600 dark:text-gray-300">
            Personalized suggestions based on your sustainability journey
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
            <EcoCard>
              <div className="flex items-start gap-3 mb-6">
                <Avatar className="h-10 w-10">
                 
                  <AvatarFallback>
                    <Sparkles className="h-5 w-5" />
                  </AvatarFallback>
                </Avatar>
                <div>
                  <h3 className="font-semibold text-lg">Your Personal Eco-Assistant</h3>
                  <p className="text-gray-600 text-sm">
                    AI-powered insights tailored to your sustainability journey
                  </p>
                </div>
              </div>
              
              {/* Gemini AI Input Section */}
              <div className="mb-6">
                <div className="flex gap-2">
                  <Input
                    placeholder="Ask for eco-friendly tips, insights, or actions..."
                    value={aiPrompt}
                    onChange={(e) => setAiPrompt(e.target.value)}
                    onKeyDown={handleKeyDown}
                    className="flex-1"
                  />
                  <Button 
                    onClick={getAiRecommendation} 
                    disabled={isLoading || !aiPrompt.trim()}
                    size="icon"
                  >
                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                  </Button>
                </div>
                
                {aiResponse && (
                  <div className="mt-4 p-4 bg-green-50 rounded-lg border border-green-100">
                    <div className="flex items-start gap-2">
                      <Brain className="h-5 w-5 text-green-600 mt-0.5 flex-shrink-0" />
                      <div>
                        <h4 className="font-medium mb-1">Eco AI Assistant</h4>
                        <p className="text-gray-700 text-sm">{aiResponse}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="p-4 bg-green-50 rounded-lg border border-green-100 mb-6">
                <h4 className="font-medium flex items-center mb-2">
                  <Brain className="h-4 w-4 text-green-600 mr-2" />
                  Your Eco-Impact Analysis{" "}
                </h4>
                <p className="text-gray-700">
                  {categoryAnalysis.topCategories.length === 0 ? (
                    "Start logging your eco-friendly actions to see your impact analysis!"
                  ) : categoryAnalysis.topCategories.length === 1 ? (
                    <>
                      Based on your logged actions, you're making the most
                      impact in{" "}
                      <span className="font-medium text-green-700">
                        {categoryAnalysis.topCategories[0].category.replace(
                          /^\w/,
                          (c) => c.toUpperCase()
                        )}
                      </span>
                      . Your CO₂ savings are{" "}
                      <span className="font-medium text-green-700">
                        {userData.impact.co2Saved > 50
                          ? "impressive"
                          : "growing steadily"}
                      </span>
                      !
                    </>
                  ) : (
                    <>
                      Based on your logged actions, you're making the most
                      impact in{" "}
                      <span className="font-medium text-green-700">
                        {categoryAnalysis.topCategories[0].category.replace(
                          /^\w/,
                          (c) => c.toUpperCase()
                        )}
                      </span>{" "}
                      and{" "}
                      <span className="font-medium text-green-700">
                        {categoryAnalysis.topCategories[1].category.replace(
                          /^\w/,
                          (c) => c.toUpperCase()
                        )}
                      </span>
                      . Your CO₂ savings are{" "}
                      <span className="font-medium text-green-700">
                        {userData.impact.co2Saved > 50
                          ? "impressive"
                          : "growing steadily"}
                      </span>
                      !
                    </>
                  )}
                </p>
                <Separator className="my-3" />
                <h4 className="font-medium flex items-center mb-2">
                  <Trending className="h-4 w-4 text-green-600 mr-2" />
                  Your Trends
                </h4>
                <p className="text-gray-700"> 
                  You've been most consistent with{" "}
                  <span className="font-medium text-green-700">
                    {categoryAnalysis.mostFrequentCategory.replace(/^\w/, (c) => c.toUpperCase())}
                  </span>{" "}
                  activities. Consider exploring more actions in{" "}
                  <span className="font-medium text-green-700">
                    {categoryAnalysis.suggestedCategory.replace(/^\w/, (c) => c.toUpperCase())}
                  </span>{" "}
                  to boost your impact.
                </p>
              </div>
            </EcoCard>
          </motion.div>
        </div>

        <div>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.3 }}
          >
            <EcoCard>
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-lg">Your Weekly Report</h3>
                <Sparkles className="h-5 w-5 text-amber-500" />
              </div>

              <div className="space-y-3">
                <div className="bg-green-50 p-4 rounded-lg border border-green-100">
                  <div className="font-medium text-green-800 mb-1">Great Progress!</div>
                  <p className="text-sm text-gray-700">
                    You've logged {userData.logs.length} eco-actions this week, that's{" "}
                    {userData.logs.length > 5 ? "above" : "slightly below"} your
                    weekly average.
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">
                      CO₂ saved this week
                    </span>
                    <span className="font-medium">{(userData.impact.co2Saved / 4).toFixed(1)} kg</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Most frequent category</span>
                    <span className="font-medium capitalize">{categoryAnalysis.mostFrequentCategory}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Points earned</span>
                    <span className="font-medium">+{(userData.points / 10).toFixed(0)}</span>
                  </div>
                </div>

                <Button className="w-full mt-2">View Full Report</Button>
              </div>
            </EcoCard>
          </motion.div>
        </div>
      </div>
    </div>
  )
}