// Utility functions for the Eco-Tracker feature

// Category icons mapping
export const categoryIcons: Record<string, string> = {
  transport: "🚗",
  energy: "⚡",
  waste: "♻️",
  water: "💧",
  food: "🍽️",
  other: "🌱",
}

// Get CSS classes for category colors
export const getCategoryColor = (category: string): string => {
  switch (category) {
    case "transport":
      return "bg-blue-100 text-blue-800"
    case "energy":
      return "bg-yellow-100 text-yellow-800"
    case "waste":
      return "bg-amber-100 text-amber-800"
    case "water":
      return "bg-indigo-100 text-indigo-800"
    case "food":
      return "bg-green-100 text-green-800"
    default:
      return "bg-purple-100 text-purple-800"
  }
}

// Format number with unit
export const formatNumberWithUnit = (value: number, unit: string): string => {
  return `${value.toFixed(1)} ${unit}`
}

// Calculate percentage
export const calculatePercentage = (value: number, total: number): number => {
  return Math.min(100, Math.round((value / total) * 100))
}

