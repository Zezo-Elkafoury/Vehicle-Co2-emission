"use client"

import { useState, useEffect, useMemo } from "react"
import { useUser } from "./UserContext"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  AlertCircle,
  Award,
  BarChart3,
  Droplets,
  Leaf,
  CloudLightningIcon as Lightning,
  Target,
  Trash2,
} from "lucide-react"
import { EcoCard } from "./EcoCard"
import { motion } from "framer-motion"

export default function ImpactDashboard() {
  const { userData, onSetGoal } = useUser()
  const [timeframe, setTimeframe] = useState("month")
  const [newGoalType, setNewGoalType] = useState("co2")
  const [newGoalTarget, setNewGoalTarget] = useState(100)
  const [update, setUpdate] = useState(false)

  useEffect(() => {
    setUpdate((prev) => !prev)
  }, [userData.impact, userData.goals])

  // Replace the getImpactData function with useMemo
  const impactData = useMemo(() => {
    // For this demo, we'll just use the total impact data
    return userData.impact
  }, [userData.impact])

  // Calculate average user data for comparison (mock data)
  const averageUserData = {
    co2Saved: impactData.co2Saved * 0.8,
    wasteSaved: impactData.wasteSaved * 0.7,
    energySaved: impactData.energySaved * 0.9,
    waterSaved: impactData.waterSaved * 0.75,
  }

  // Format number with appropriate units
  const formatWithUnit = (value: number, type: string) => {
    switch (type) {
      case "co2":
        return `${value.toFixed(1)} kg CO₂`
      case "waste":
        return `${value.toFixed(1)} kg`
      case "energy":
        return `${value.toFixed(1)} kWh`
      case "water":
        return `${value.toFixed(1)} L`
      default:
        return `${value.toFixed(1)}`
    }
  }

  const handleSubmitNewGoal = () => {
    onSetGoal({
      type: newGoalType,
      target: newGoalTarget,
    })
  }

  // Calculate impact metrics with useMemo
  const equivalents = useMemo(
    () => ({
      trees: Math.round(impactData.co2Saved / 20), // Each tree absorbs approx 20kg CO2 per year
      carKm: Math.round(impactData.co2Saved * 4), // Approx 250g CO2 per km in average car
      phoneCharges: Math.round(impactData.energySaved * 200), // Approx 0.005 kWh per phone charge
      bottlesOfWater: Math.round(impactData.waterSaved / 0.5), // 0.5L per standard bottle
    }),
    [impactData],
  )

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold gradient-text mb-2">Your Environmental Impact</h2>
          <p className="text-gray-600 dark:text-gray-300">See the difference your actions are making</p>
        </div>

        <Tabs value={timeframe} onValueChange={setTimeframe} className="sm:w-auto">
          <TabsList>
            <TabsTrigger value="week">Week</TabsTrigger>
            <TabsTrigger value="month">Month</TabsTrigger>
            <TabsTrigger value="year">Year</TabsTrigger>
            <TabsTrigger value="all">All Time</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
        >
          <EcoCard>
            <div className="flex justify-between items-start mb-3">
              <h3 className="font-semibold text-gray-700">CO₂ Reduction</h3>
              <Leaf className="h-5 w-5 text-green-600" />
            </div>
            <p className="text-3xl font-bold gradient-text mb-2">{impactData.co2Saved.toFixed(1)} kg</p>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm text-gray-500">You</span>
              <span className="text-sm text-gray-500">Avg. User</span>
            </div>
            <Progress value={(impactData.co2Saved / (averageUserData.co2Saved * 1.5)) * 100} className="h-2 mb-3" />
            <p className="text-xs text-gray-500">
              You've saved {Math.round((impactData.co2Saved / averageUserData.co2Saved - 1) * 100)}% more CO₂ than
              average users
            </p>
          </EcoCard>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <EcoCard>
            <div className="flex justify-between items-start mb-3">
              <h3 className="font-semibold text-gray-700">Waste Reduction</h3>
              <Trash2 className="h-5 w-5 text-amber-600" />
            </div>
            <p className="text-3xl font-bold gradient-text mb-2">{impactData.wasteSaved.toFixed(1)} kg</p>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm text-gray-500">You</span>
              <span className="text-sm text-gray-500">Avg. User</span>
            </div>
            <Progress value={(impactData.wasteSaved / (averageUserData.wasteSaved * 1.5)) * 100} className="h-2 mb-3" />
            <p className="text-xs text-gray-500">
              You've reduced {Math.round((impactData.wasteSaved / averageUserData.wasteSaved - 1) * 100)}% more waste
              than average users
            </p>
          </EcoCard>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.3 }}
        >
          <EcoCard>
            <div className="flex justify-between items-start mb-3">
              <h3 className="font-semibold text-gray-700">Energy Saved</h3>
              <Lightning className="h-5 w-5 text-yellow-500" />
            </div>
            <p className="text-3xl font-bold gradient-text mb-2">{impactData.energySaved.toFixed(1)} kWh</p>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm text-gray-500">You</span>
              <span className="text-sm text-gray-500">Avg. User</span>
            </div>
            <Progress
              value={(impactData.energySaved / (averageUserData.energySaved * 1.5)) * 100}
              className="h-2 mb-3"
            />
            <p className="text-xs text-gray-500">
              You've saved {Math.round((impactData.energySaved / averageUserData.energySaved - 1) * 100)}% more energy
              than average users
            </p>
          </EcoCard>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.4 }}
        >
          <EcoCard>
            <div className="flex justify-between items-start mb-3">
              <h3 className="font-semibold text-gray-700">Water Saved</h3>
              <Droplets className="h-5 w-5 text-blue-500" />
            </div>
            <p className="text-3xl font-bold gradient-text mb-2">{impactData.waterSaved.toFixed(1)} L</p>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm text-gray-500">You</span>
              <span className="text-sm text-gray-500">Avg. User</span>
            </div>
            <Progress value={(impactData.waterSaved / (averageUserData.waterSaved * 1.5)) * 100} className="h-2 mb-3" />
            <p className="text-xs text-gray-500">
              You've saved {Math.round((impactData.waterSaved / averageUserData.waterSaved - 1) * 100)}% more water than
              average users
            </p>
          </EcoCard>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
        <EcoCard className="lg:col-span-2">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-semibold gradient-text">Your Impact in Real Terms</h3>
            <BarChart3 className="h-5 w-5 text-green-600" />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="bg-green-50 rounded-lg p-4 flex items-center gap-4">
              <div className="h-12 w-12 rounded-full bg-green-100 flex items-center justify-center">
                <Leaf className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Equivalent to</p>
                <p className="text-lg font-semibold">{equivalents.trees} trees</p>
                <p className="text-xs text-gray-500">absorbing CO₂ for a year</p>
              </div>
            </div>

            <div className="bg-blue-50 rounded-lg p-4 flex items-center gap-4">
              <div className="h-12 w-12 rounded-full bg-blue-100 flex items-center justify-center">
                <Lightning className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Equivalent to</p>
                <p className="text-lg font-semibold">{equivalents.phoneCharges} phone charges</p>
                <p className="text-xs text-gray-500">worth of energy saved</p>
              </div>
            </div>

            <div className="bg-amber-50 rounded-lg p-4 flex items-center gap-4">
              <div className="h-12 w-12 rounded-full bg-amber-100 flex items-center justify-center">
                <AlertCircle className="h-6 w-6 text-amber-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Prevented</p>
                <p className="text-lg font-semibold">{equivalents.carKm} km</p>
                <p className="text-xs text-gray-500">of car travel emissions</p>
              </div>
            </div>

            <div className="bg-indigo-50 rounded-lg p-4 flex items-center gap-4">
              <div className="h-12 w-12 rounded-full bg-indigo-100 flex items-center justify-center">
                <Droplets className="h-6 w-6 text-indigo-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Saved</p>
                <p className="text-lg font-semibold">{equivalents.bottlesOfWater} bottles</p>
                <p className="text-xs text-gray-500">of water (500ml each)</p>
              </div>
            </div>
          </div>
        </EcoCard>

        <EcoCard>
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-semibold gradient-text">Your Goals</h3>
            <Target className="h-5 w-5 text-green-600" />
          </div>

          <div className="space-y-4">
            {userData.goals.length > 0 ? (
              userData.goals.map((goal, index) => (
                <div key={index} className="border border-gray-200 rounded-lg p-4">
                  <div className="flex justify-between items-center mb-2">
                    <h4 className="font-medium">{formatWithUnit(goal.target, goal.type)}</h4>
                    <span className="text-xs text-gray-500">
                      {Math.min(100, Math.round((goal.progress / goal.target) * 100))}%
                    </span>
                  </div>
                  <Progress value={Math.min(100, (goal.progress / goal.target) * 100)} className="h-2 mb-2" />
                  <p className="text-xs text-gray-500">
                    {goal.progress < goal.target
                      ? `${formatWithUnit(goal.target - goal.progress, goal.type)} to go`
                      : "Goal completed! 🎉"}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-gray-500 text-center py-4">No goals set yet</p>
            )}

            <Dialog>
              <DialogTrigger asChild>
                <Button className="w-full">
                  <Target className="h-4 w-4 mr-2" />
                  Set New Goal
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Set a New Eco-Goal</DialogTitle>
                </DialogHeader>

                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="goal-type">Goal Type</Label>
                    <Select value={newGoalType} onValueChange={setNewGoalType}>
                      <SelectTrigger id="goal-type">
                        <SelectValue placeholder="Select goal type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="co2">CO₂ Reduction</SelectItem>
                        <SelectItem value="waste">Waste Reduction</SelectItem>
                        <SelectItem value="energy">Energy Saving</SelectItem>
                        <SelectItem value="water">Water Saving</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="goal-target">Target Amount</Label>
                    <div className="flex gap-2">
                      <Input
                        id="goal-target"
                        type="number"
                        min="1"
                        value={newGoalTarget}
                        onChange={(e) => setNewGoalTarget(Number.parseInt(e.target.value, 10) || 1)}
                      />
                      <div className="flex items-center bg-gray-100 px-3 rounded-md">
                        <span className="text-gray-600">
                          {newGoalType === "co2"
                            ? "kg CO₂"
                            : newGoalType === "waste"
                              ? "kg"
                              : newGoalType === "energy"
                                ? "kWh"
                                : "L"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="flex justify-end">
                  <Button onClick={handleSubmitNewGoal}>
                    <Award className="h-4 w-4 mr-2" />
                    Create Goal
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </EcoCard>
      </div>
    </div>
  )
}

