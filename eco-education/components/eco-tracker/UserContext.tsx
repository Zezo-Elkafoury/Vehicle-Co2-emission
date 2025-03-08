"use client"

import type React from "react"

import { createContext, useContext } from "react"
import type { UserData, EcoAction, Challenge } from "@/lib/eco-tracker/types"

// Update the UserContextType interface to ensure date is required
interface UserContextType {
  userData: UserData
  setUserData: React.Dispatch<React.SetStateAction<UserData>>
  onAddAction: (action: EcoAction, date: Date) => void
  onRemoveAction: (actionId: string) => void
  onSetGoal: (goal: { type: string; target: number }) => void
  onJoinChallenge: (challenge: Challenge) => void
  onAddCustomAction: (action: Omit<EcoAction, "id">) => void
}

const UserContext = createContext<UserContextType | null>(null)

export const UserProvider = UserContext.Provider

export function useUser(): UserContextType {
  const context = useContext(UserContext)
  if (!context) {
    throw new Error("useUser must be used within a UserProvider")
  }
  return context
}

