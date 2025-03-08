// Define types for the Eco-Tracker feature

export interface EcoAction {
  id: string
  name: string
  category: string
  impact: {
    co2Saved: number
    wasteSaved: number
    energySaved: number
    waterSaved: number
  }
  icon: string
}

export interface UserLog {
  id: string
  actionId: string
  actionName: string
  category: string
  impact: {
    co2Saved: number
    wasteSaved: number
    energySaved: number
    waterSaved: number
  }
  date: Date
  notes: string
}

export interface Challenge {
  id: string
  title: string
  description: string
  category: string
  difficulty: string
  duration: string
  points: number
  participants: number
  progress: number
  completed?: boolean
}

export interface Goal {
  id: string
  type: string
  target: number
  progress: number
}

export interface UserData {
  username: string
  avatarUrl: string
  points: number
  streak: number
  impact: {
    co2Saved: number
    wasteSaved: number
    energySaved: number
    waterSaved: number
  }
  logs: UserLog[]
  availableActions: EcoAction[]
  activeChallenges: Challenge[]
  availableChallenges: Challenge[]
  goals: Goal[]
}

