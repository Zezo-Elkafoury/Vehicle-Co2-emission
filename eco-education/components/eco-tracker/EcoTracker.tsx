"use client"

import { useState, useCallback, useMemo } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import ActionLogging from "./ActionLogging"
import ImpactDashboard from "./ImpactDashboard"
import ActionTimeline from "./ActionTimeline"
import Challenges from "./Challenges"
import Recommendations from "./Recommendations"
import { UserProvider } from "./UserContext"
import { generateMockUserData } from "@/lib/eco-tracker/mockData"
import type { EcoAction, UserLog, Challenge } from "@/lib/eco-tracker/types"
import { EcoCard } from "./EcoCard"
import { motion } from "framer-motion"
import { Leaf, LineChart, Clock, Trophy, Lightbulb } from "lucide-react"

export default function EcoTracker() {
  const [userData, setUserData] = useState(() => generateMockUserData())
  // Use a version counter instead of a boolean toggle for more predictable updates
  const [version, setVersion] = useState(0)

  // Memoize handlers to prevent unnecessary re-renders
  const handleAddAction = useCallback((action: EcoAction, date: Date) => {
    const newLog: UserLog = {
      id: `log-${Date.now()}`,
      actionId: action.id,
      actionName: action.name,
      category: action.category,
      impact: action.impact,
      date: date,
      notes: "",
    }

    setUserData((prev) => {
      // Create a completely new object to ensure React detects the change
      return {
        ...prev,
        logs: [newLog, ...prev.logs],
        impact: {
          co2Saved: prev.impact.co2Saved + action.impact.co2Saved,
          wasteSaved: prev.impact.wasteSaved + action.impact.wasteSaved,
          energySaved: prev.impact.energySaved + action.impact.energySaved,
          waterSaved: prev.impact.waterSaved + action.impact.waterSaved,
        },
      }
    })
    
    // Increment version to signal updates to components that need to re-render
    setVersion(v => v + 1)
  }, [])

  const handleRemoveAction = useCallback((actionId: string) => {
    setUserData((prev) => ({
      ...prev,
      availableActions: prev.availableActions.filter((action) => action.id !== actionId),
    }))
    setVersion(v => v + 1)
  }, [])

  const handleAddCustomAction = useCallback((action: Omit<EcoAction, "id">) => {
    const newAction: EcoAction = {
      ...action,
      id: `custom-${Date.now()}`,
    }

    setUserData((prev) => ({
      ...prev,
      availableActions: [...prev.availableActions, newAction],
    }))
    setVersion(v => v + 1)
  }, [])

  const handleSetGoal = useCallback((goal: { type: string; target: number }) => {
    setUserData((prev) => ({
      ...prev,
      goals: [...prev.goals, { ...goal, id: `goal-${Date.now()}`, progress: 0 }],
    }))
    setVersion(v => v + 1)
  }, [])

  const handleJoinChallenge = useCallback((challenge: Challenge) => {
    if (!userData.activeChallenges.some((c) => c.id === challenge.id)) {
      setUserData((prev) => ({
        ...prev,
        activeChallenges: [...prev.activeChallenges, challenge],
      }))
      setVersion(v => v + 1)
    }
  }, [userData.activeChallenges])

  // Memoize the context value to prevent unnecessary re-renders
  const contextValue = useMemo(() => ({
    userData,
    setUserData,
    onAddAction: handleAddAction,
    onRemoveAction: handleRemoveAction,
    onSetGoal: handleSetGoal,
    onJoinChallenge: handleJoinChallenge,
    onAddCustomAction: handleAddCustomAction,
    version, // Include version in context to allow components to react to changes
  }), [
    userData, 
    handleAddAction, 
    handleRemoveAction, 
    handleSetGoal, 
    handleJoinChallenge, 
    handleAddCustomAction,
    version
  ])

  return (
    <UserProvider value={contextValue}>
      <div className="container mx-auto px-4 py-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
          <h1 className="text-3xl md:text-4xl font-bold gradient-text text-center mt-12 mb-6">
            Eco-Friendly Actions Tracker
          </h1>
          <p className="text-center text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto">
            Track your sustainable actions, monitor your environmental impact, and join challenges to make a difference
            for our planet.
          </p>
        </motion.div>

        <EcoCard className="mb-8">
          <Tabs defaultValue="log" className="w-full">
            <TabsList className="grid grid-cols-5 mb-8">
              <TabsTrigger value="log" className="flex items-center gap-2">
                <Leaf className="h-4 w-4" />
                <span className="hidden sm:inline">Log Actions</span>
              </TabsTrigger>
              <TabsTrigger value="impact" className="flex items-center gap-2">
                <LineChart className="h-4 w-4" />
                <span className="hidden sm:inline">Impact</span>
              </TabsTrigger>
              <TabsTrigger value="timeline" className="flex items-center gap-2">
                <Clock className="h-4 w-4" />
                <span className="hidden sm:inline">Timeline</span>
              </TabsTrigger>
              <TabsTrigger value="challenges" className="flex items-center gap-2">
                <Trophy className="h-4 w-4" />
                <span className="hidden sm:inline">Challenges</span>
              </TabsTrigger>
              <TabsTrigger value="recommendations" className="flex items-center gap-2">
                <Lightbulb className="h-4 w-4" />
                <span className="hidden sm:inline">Insights</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="log" className="space-y-4">
              <ActionLogging />
            </TabsContent>

            <TabsContent value="impact">
              <ImpactDashboard />
            </TabsContent>

            <TabsContent value="timeline">
              <ActionTimeline />
            </TabsContent>

            <TabsContent value="challenges">
              <Challenges />
            </TabsContent>

            <TabsContent value="recommendations">
              <Recommendations />
            </TabsContent>
          </Tabs>
        </EcoCard>
      </div>
    </UserProvider>
  )
}