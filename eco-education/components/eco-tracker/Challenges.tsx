"use client"

import { useState, useEffect, useMemo } from "react"
import { useUser } from "./UserContext"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Award, Clock, Trophy, Users, CheckCircle, XCircle, Share2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { EcoCard } from "./EcoCard"
import { Badge } from "@/components/ui/badge"
import { motion } from "framer-motion"

export default function Challenges() {
  const { userData, onJoinChallenge } = useUser()
  const [update, setUpdate] = useState(false)

  // Add useEffect to trigger re-render when userData changes
  useEffect(() => {
    setUpdate((prev) => !prev)
  }, [userData])

  // Replace the activeChallenges and availableChallenges calculations with useMemo
  const activeChallenges = useMemo(() => {
    return userData.activeChallenges
  }, [userData.activeChallenges])

  const availableChallenges = useMemo(() => {
    return userData.availableChallenges.filter(
      (challenge) => !userData.activeChallenges.some((ac) => ac.id === challenge.id),
    )
  }, [userData.availableChallenges, userData.activeChallenges])

  // Replace the leaderboard and userRank calculations with useMemo
  const { leaderboard, userRank } = useMemo(() => {
    const leaderboardData = [
      { username: "EcoWarrior", avatarUrl: "", points: 2450 },
      { username: "GreenThumb", avatarUrl: "", points: 2340 },
      { username: "ZeroWaster", avatarUrl: "", points: 2180 },
      { username: "SustainableSam", avatarUrl: "", points: 1950 },
      { userData: userData.username, avatarUrl: userData.avatarUrl, points: userData.points },
      { username: "EcoChampion", avatarUrl: "", points: 1840 },
      { username: "EarthSaver", avatarUrl: "", points: 1720 },
      { username: "RecycleKing", avatarUrl: "", points: 1680 },
      { username: "PlanetProtector", avatarUrl: "", points: 1550 },
      { username: "GreenLiving", avatarUrl: "", points: 1490 },
    ].sort((a, b) => b.points - a.points)

    const rank = leaderboardData.findIndex((user) => "userData" in user) + 1

    return { leaderboard: leaderboardData, userRank: rank }
  }, [userData.username, userData.avatarUrl, userData.points])

  // Simulate completed challenges
  const completedChallenges = [
    {
      id: "challenge-completed-1",
      title: "Plastic-Free Week",
      description: "Avoid single-use plastics for an entire week",
      category: "waste",
      difficulty: "medium",
      duration: "7 days",
      points: 150,
      participants: 1243,
      progress: 100,
      completed: true,
    },
    {
      id: "challenge-completed-2",
      title: "Walk This Way",
      description: "Replace 10 car trips with walking or cycling",
      category: "transport",
      difficulty: "easy",
      duration: "30 days",
      points: 100,
      participants: 2789,
      progress: 100,
      completed: true,
    },
  ]

  // Get difficulty color
  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case "easy":
        return "bg-green-100 text-green-800"
      case "medium":
        return "bg-amber-100 text-amber-800"
      case "hard":
        return "bg-red-100 text-red-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold gradient-text mb-2">Sustainability Challenges</h2>
          <p className="text-gray-600 dark:text-gray-300">Join challenges, compete with others, and earn rewards</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Tabs defaultValue="active">
            <TabsList className="mb-6">
              <TabsTrigger value="active" className="flex items-center gap-2">
                <Trophy className="h-4 w-4" />
                Active Challenges
              </TabsTrigger>
              <TabsTrigger value="available" className="flex items-center gap-2">
                <Award className="h-4 w-4" />
                Available
              </TabsTrigger>
              <TabsTrigger value="completed" className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4" />
                Completed
              </TabsTrigger>
            </TabsList>

            <TabsContent value="active">
              {activeChallenges.length > 0 ? (
                <div className="space-y-4">
                  {activeChallenges.map((challenge, index) => (
                    <motion.div
                      key={challenge.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                    >
                      <EcoCard>
                        <div className="flex flex-col md:flex-row md:items-center gap-4">
                          <div className="flex-grow">
                            <div className="flex justify-between items-start">
                              <div>
                                <h3 className="font-semibold text-lg">{challenge.title}</h3>
                                <p className="text-gray-600 text-sm mt-1">{challenge.description}</p>
                              </div>
                              <Badge className={getDifficultyColor(challenge.difficulty)}>{challenge.difficulty}</Badge>
                            </div>

                            <div className="mt-4">
                              <div className="flex justify-between items-center text-sm text-gray-600 mb-1">
                                <span>Progress</span>
                                <span>{challenge.progress}%</span>
                              </div>
                              <Progress value={challenge.progress} className="h-2" />
                            </div>

                            <div className="flex flex-wrap gap-2 mt-4">
                              <div className="flex items-center text-xs text-gray-600">
                                <Clock className="h-3 w-3 mr-1" />
                                {challenge.duration}
                              </div>
                              <div className="flex items-center text-xs text-gray-600">
                                <Trophy className="h-3 w-3 mr-1" />
                                {challenge.points} points
                              </div>
                              <div className="flex items-center text-xs text-gray-600">
                                <Users className="h-3 w-3 mr-1" />
                                {challenge.participants.toLocaleString()} participants
                              </div>
                            </div>
                          </div>

                          <div className="flex gap-2 md:flex-col">
                            <Button variant="outline" size="sm" className="flex-1 md:w-[120px]">
                              <Share2 className="h-4 w-4 mr-2" />
                              Share
                            </Button>
                            <Button variant="destructive" size="sm" className="flex-1 md:w-[120px]">
                              <XCircle className="h-4 w-4 mr-2" />
                              Abandon
                            </Button>
                          </div>
                        </div>
                      </EcoCard>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12 bg-gray-50 rounded-lg border border-dashed border-gray-200">
                  <Trophy className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-600 mb-2">No Active Challenges</h3>
                  <p className="text-gray-500 mb-6">Join a challenge to start earning points and making an impact!</p>
                  <Button>Explore Challenges</Button>
                </div>
              )}
            </TabsContent>

            <TabsContent value="available">
              <div className="space-y-4">
                {availableChallenges.map((challenge, index) => (
                  <motion.div
                    key={challenge.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <EcoCard>
                      <div className="flex flex-col md:flex-row md:items-center gap-4">
                        <div className="flex-grow">
                          <div className="flex justify-between items-start">
                            <div>
                              <h3 className="font-semibold text-lg">{challenge.title}</h3>
                              <p className="text-gray-600 text-sm mt-1">{challenge.description}</p>
                            </div>
                            <Badge className={getDifficultyColor(challenge.difficulty)}>{challenge.difficulty}</Badge>
                          </div>

                          <div className="flex flex-wrap gap-2 mt-4">
                            <div className="flex items-center text-xs text-gray-600">
                              <Clock className="h-3 w-3 mr-1" />
                              {challenge.duration}
                            </div>
                            <div className="flex items-center text-xs text-gray-600">
                              <Trophy className="h-3 w-3 mr-1" />
                              {challenge.points} points
                            </div>
                            <div className="flex items-center text-xs text-gray-600">
                              <Users className="h-3 w-3 mr-1" />
                              {challenge.participants.toLocaleString()} participants
                            </div>
                          </div>
                        </div>

                        <Button className="md:w-[120px]" onClick={() => onJoinChallenge(challenge)}>
                          Join
                        </Button>
                      </div>
                    </EcoCard>
                  </motion.div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="completed">
              <div className="space-y-4">
                {completedChallenges.map((challenge, index) => (
                  <motion.div
                    key={challenge.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <EcoCard>
                      <div className="flex flex-col md:flex-row md:items-center gap-4">
                        <div className="flex-grow">
                          <div className="flex justify-between items-start">
                            <div className="flex items-start gap-2">
                              <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                              <div>
                                <h3 className="font-semibold text-lg">{challenge.title}</h3>
                                <p className="text-gray-600 text-sm mt-1">{challenge.description}</p>
                              </div>
                            </div>
                            <Badge variant="outline" className="text-green-600 bg-green-50 border-green-200">
                              Completed
                            </Badge>
                          </div>

                          <div className="mt-4">
                            <div className="flex justify-between items-center text-sm text-gray-600 mb-1">
                              <span>Completed</span>
                              <span>{challenge.progress}%</span>
                            </div>
                            <Progress value={challenge.progress} className="h-2 bg-green-100" />
                          </div>

                          <div className="flex flex-wrap gap-2 mt-4">
                            <div className="flex items-center text-xs text-gray-600">
                              <Trophy className="h-3 w-3 mr-1" />
                              Earned {challenge.points} points
                            </div>
                          </div>
                        </div>

                        <div className="flex gap-2">
                          <Button variant="outline" size="sm" className="flex-1 md:w-[120px]">
                            <Share2 className="h-4 w-4 mr-2" />
                            Share
                          </Button>
                        </div>
                      </div>
                    </EcoCard>
                  </motion.div>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </div>

        <div>
          <EcoCard>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-lg">Leaderboard</h3>
              <Trophy className="h-5 w-5 text-amber-500" />
            </div>

            <div className="space-y-3">
              {leaderboard.map((user, index) => {
                const isCurrentUser = "userData" in user

                return (
                  <div
                    key={index}
                    className={cn(
                      "flex items-center gap-3 p-2 rounded-lg",
                      isCurrentUser ? "bg-green-50 border border-green-100" : "",
                    )}
                  >
                    <div
                      className={cn(
                        "w-7 h-7 rounded-full flex items-center justify-center text-sm font-medium",
                        index === 0
                          ? "bg-amber-100 text-amber-800"
                          : index === 1
                            ? "bg-gray-200 text-gray-800"
                            : index === 2
                              ? "bg-amber-900/20 text-amber-800"
                              : "bg-gray-100 text-gray-600",
                      )}
                    >
                      {index + 1}
                    </div>

                    <Avatar className="h-8 w-8">
                      <AvatarImage src={isCurrentUser ? userData.avatarUrl : user.avatarUrl} />
                      <AvatarFallback>
                        {isCurrentUser ? userData.username.charAt(0) : user.username.charAt(0)}
                      </AvatarFallback>
                    </Avatar>

                    <div className="flex-grow truncate">
                      <span className={isCurrentUser ? "font-medium" : ""}>
                        {isCurrentUser ? userData.username : user.username}
                      </span>
                    </div>

                    <div className="font-medium text-green-700">{user.points}</div>
                  </div>
                )
              })}
            </div>

            <div className="mt-6 text-center text-sm text-gray-500">
              <p>
                You are ranked <span className="font-medium text-green-700">#{userRank}</span> out of{" "}
                {leaderboard.length + 45} users.
              </p>
            </div>
          </EcoCard>

          <EcoCard className="mt-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-lg">Your Stats</h3>
              <Award className="h-5 w-5 text-green-600" />
            </div>

            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Points earned</span>
                <span className="font-medium">{userData.points}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Challenges completed</span>
                <span className="font-medium">{completedChallenges.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Challenges active</span>
                <span className="font-medium">{activeChallenges.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Current streak</span>
                <span className="font-medium">{userData.streak} days</span>
              </div>
            </div>
          </EcoCard>
        </div>
      </div>
    </div>
  )
}

