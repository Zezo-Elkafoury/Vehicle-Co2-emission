import type { UserData, EcoAction, Challenge } from "./types"

// Generate mock data for the Eco-Tracker feature
export const generateMockUserData = (): UserData => {
  // Create one action per category
  const availableActions: EcoAction[] = [
    {
      id: "action-transport-1",
      name: "Used public transport",
      category: "transport",
      impact: { co2Saved: 2.5, wasteSaved: 0, energySaved: 1.2, waterSaved: 0 },
      icon: "🚌",
    },
    {
      id: "action-waste-1",
      name: "Used a reusable water bottle",
      category: "waste",
      impact: { co2Saved: 0.5, wasteSaved: 0.2, energySaved: 0.1, waterSaved: 0.5 },
      icon: "🧴",
    },
    {
      id: "action-energy-1",
      name: "Turned off lights when not needed",
      category: "energy",
      impact: { co2Saved: 0.7, wasteSaved: 0, energySaved: 0.5, waterSaved: 0 },
      icon: "💡",
    },
    {
      id: "action-water-1",
      name: "Took a shorter shower",
      category: "water",
      impact: { co2Saved: 0.4, wasteSaved: 0, energySaved: 0.2, waterSaved: 30 },
      icon: "🚿",
    },
    {
      id: "action-food-1",
      name: "Ate a plant-based meal",
      category: "food",
      impact: { co2Saved: 1.5, wasteSaved: 0.1, energySaved: 0.3, waterSaved: 1.2 },
      icon: "🥗",
    },
  ]

  // Generate a few random logs
  const today = new Date()
  const logs = Array.from({ length: 5 }, (_, i) => {
    const action = availableActions[Math.floor(Math.random() * availableActions.length)]
    const date = new Date()
    date.setDate(today.getDate() - Math.floor(Math.random() * 7)) // Last week only
    date.setHours(Math.floor(Math.random() * 24))

    return {
      id: `log-${i}`,
      actionId: action.id,
      actionName: action.name,
      category: action.category,
      impact: action.impact,
      date: date,
      notes: "",
    }
  }).sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())

  // Calculate total impact
  const impact = logs.reduce(
    (total, log) => {
      return {
        co2Saved: total.co2Saved + log.impact.co2Saved,
        wasteSaved: total.wasteSaved + log.impact.wasteSaved,
        energySaved: total.energySaved + log.impact.energySaved,
        waterSaved: total.waterSaved + log.impact.waterSaved,
      }
    },
    {
      co2Saved: 0,
      wasteSaved: 0,
      energySaved: 0,
      waterSaved: 0,
    },
  )

  // Generate challenges
  const availableChallenges: Challenge[] = [
    {
      id: "challenge-1",
      title: "Zero Waste Week",
      description: "Reduce your waste production to near-zero for 7 days",
      category: "waste",
      difficulty: "hard",
      duration: "7 days",
      points: 200,
      participants: 1865,
      progress: 0,
    },
    {
      id: "challenge-2",
      title: "Bike Commuter",
      description: "Use your bike for all commutes for 5 days straight",
      category: "transport",
      difficulty: "medium",
      duration: "5 days",
      points: 150,
      participants: 2317,
      progress: 0,
    },
  ]

  // Create active challenges
  const activeChallenges: Challenge[] = [
    {
      ...availableChallenges[1],
      progress: 60,
    },
  ]

  return {
    username: "EcoFriend",
    avatarUrl: "/placeholder.svg?height=32&width=32",
    points: 1890,
    streak: 7,
    impact,
    logs,
    availableActions,
    activeChallenges,
    availableChallenges,
    goals: [
      {
        id: "goal-1",
        type: "co2",
        target: 50,
        progress: 31.5,
      },
    ],
  }
}

