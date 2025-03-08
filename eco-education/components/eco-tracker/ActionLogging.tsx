"use client"

import { useState, useEffect, useMemo } from "react"
import { useUser } from "./UserContext"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Calendar } from "@/components/ui/calendar"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Label } from "@/components/ui/label"
import { Mic, Search, CalendarIcon, Check, Plus, X, Trash2 } from "lucide-react"
import { format } from "date-fns"
import type { EcoAction } from "@/lib/eco-tracker/types"
import { Badge } from "@/components/ui/badge"
import { motion, AnimatePresence } from "framer-motion"
import { categoryIcons, getCategoryColor } from "@/lib/eco-tracker/utils"
import { EcoCard } from "./EcoCard"
import { cn } from "@/lib/utils"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"

// Mock speech recognition for demo purposes
class MockSpeechRecognition {
  onresult: ((event: any) => void) | null = null
  onend: (() => void) | null = null

  start() {
    setTimeout(() => {
      if (this.onresult) {
        const mockResult = {
          results: [
            [
              {
                transcript: "Used a reusable water bottle today",
                confidence: 0.9,
              },
            ],
          ],
        }
        this.onresult(mockResult)
      }

      if (this.onend) {
        this.onend()
      }
    }, 2000)
  }

  stop() {}
}

export default function ActionLogging() {
  const { userData, onAddAction, onRemoveAction, onAddCustomAction } = useUser()
  const [update, setUpdate] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedDate, setSelectedDate] = useState<Date>(new Date())
  const [isListening, setIsListening] = useState(false)
  const [voiceText, setVoiceText] = useState("")
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [customAction, setCustomAction] = useState("")
  const [showAddCustom, setShowAddCustom] = useState(false)
  const [selectedCustomCategory, setSelectedCustomCategory] = useState("transport")
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [actionToDelete, setActionToDelete] = useState<EcoAction | null>(null)
  const [customImpact, setCustomImpact] = useState({
    co2Saved: 1.0,
    wasteSaved: 0.2,
    energySaved: 0.5,
    waterSaved: 0.3,
  })
  const [recentlyUsed, setRecentlyUsed] = useState<EcoAction[]>([])

  useEffect(() => {
    setUpdate((prev) => !prev)
  }, [userData.logs.length])

  // Simulate fetching recently used actions from user history
  const recentlyUsedMemo = useMemo(() => {
    const recent = userData.logs.slice(0, 5).map((log) => {
      return (
        userData.availableActions.find((action) => action.id === log.actionId) || {
          id: log.actionId,
          name: log.actionName,
          category: log.category,
          impact: log.impact,
          icon: "✓",
        }
      )
    })
    return recent
  }, [userData.logs, userData.availableActions])

  const handleStartListening = () => {
    setIsListening(true)
    setVoiceText("")

    // Mock voice recognition
    const recognition = new MockSpeechRecognition()

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      setVoiceText(transcript)
    }

    recognition.onend = () => {
      setIsListening(false)
    }

    recognition.start()
  }

  const handleLogAction = (action: EcoAction) => {
    onAddAction(action, new Date())

    // Animation or feedback could be added here
    const newRecentlyUsed = [action, ...recentlyUsed.filter((a) => a.id !== action.id)].slice(0, 5)
    setRecentlyUsed(newRecentlyUsed)
  }

  const handleCreateCustomAction = () => {
    if (!customAction.trim()) return

    const newAction: Omit<EcoAction, "id"> = {
      name: customAction,
      category: selectedCustomCategory,
      impact: customImpact,
      icon: categoryIcons[selectedCustomCategory] || "✓",
    }

    onAddCustomAction(newAction)
    setCustomAction("")
    setShowAddCustom(false)

    // Reset impact values to defaults
    setCustomImpact({
      co2Saved: 1.0,
      wasteSaved: 0.2,
      energySaved: 0.5,
      waterSaved: 0.3,
    })
  }

  const handleDeleteAction = () => {
    if (actionToDelete) {
      onRemoveAction(actionToDelete.id)
      setActionToDelete(null)
      setShowDeleteDialog(false)
    }
  }

  const confirmDelete = (action: EcoAction) => {
    setActionToDelete(action)
    setShowDeleteDialog(true)
  }

  // Filter actions based on search and category
  const filteredActions = userData.availableActions.filter((action) => {
    const matchesSearch = searchQuery === "" || action.name.toLowerCase().includes(searchQuery.toLowerCase())

    const matchesCategory = selectedCategory === null || action.category === selectedCategory

    return matchesSearch && matchesCategory
  })

  const categories = Array.from(new Set(userData.availableActions.map((action) => action.category)))

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold gradient-text mb-2">Log Your Eco-Actions</h2>
          <p className="text-gray-600 dark:text-gray-300">Track your sustainable activities and see your impact grow</p>
        </div>

        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" className="flex items-center gap-2">
              <CalendarIcon className="h-4 w-4" />
              {format(selectedDate, "PPP")}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0" align="end">
            <Calendar
              mode="single"
              selected={selectedDate}
              onSelect={(date) => date && setSelectedDate(date)}
              initialFocus
              disabled={(date) => date > new Date()}
            />
          </PopoverContent>
        </Popover>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2">
          <Tabs defaultValue="all">
            <div className="flex justify-between items-start mb-4">
              <TabsList>
                <TabsTrigger value="all">All Actions</TabsTrigger>
                <TabsTrigger value="recent">Recently Used</TabsTrigger>
                <TabsTrigger value="custom">Custom</TabsTrigger>
              </TabsList>

              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search actions..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 w-full md:w-auto"
                />
              </div>
            </div>

            <TabsContent value="all" className="space-y-4">
              <div className="flex flex-wrap gap-2 mb-4">
                <Badge
                  variant={selectedCategory === null ? "default" : "outline"}
                  onClick={() => setSelectedCategory(null)}
                  className="cursor-pointer"
                >
                  All Categories
                </Badge>
                {categories.map((category) => (
                  <Badge
                    key={category}
                    variant={selectedCategory === category ? "default" : "outline"}
                    onClick={() => setSelectedCategory(category)}
                    className={cn("cursor-pointer", selectedCategory === category ? getCategoryColor(category) : "")}
                  >
                    {categoryIcons[category]} {category.charAt(0).toUpperCase() + category.slice(1)}
                  </Badge>
                ))}
              </div>

              <ScrollArea className="h-[400px] pr-4">
                <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {filteredActions.map((action) => (
                    <motion.li
                      key={action.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={`
                        flex items-center justify-between p-4 rounded-lg border border-gray-200 
                        hover:border-green-500 hover:shadow-md transition-all duration-200
                      `}
                    >
                      <div
                        className="flex items-center gap-3 flex-grow cursor-pointer"
                        onClick={() => handleLogAction(action)}
                      >
                        <Badge className={getCategoryColor(action.category)}>
                          {categoryIcons[action.category] || "♻️"}
                        </Badge>
                        <span>{action.name}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="text-xs text-gray-500">-{action.impact.co2Saved} kg CO₂</div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-gray-500 hover:text-red-500"
                          onClick={() => confirmDelete(action)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </motion.li>
                  ))}
                </ul>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="recent">
              <ScrollArea className="h-[400px] pr-4">
                <ul className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {recentlyUsed.map((action) => (
                    <motion.li
                      key={action.id}
                      className={`
                        flex items-center justify-between p-4 rounded-lg border border-gray-200 
                        hover:border-green-500 hover:shadow-md transition-all duration-200
                        cursor-pointer
                      `}
                      onClick={() => handleLogAction(action)}
                    >
                      <div className="flex items-center gap-3">
                        <Badge className={getCategoryColor(action.category)}>
                          {categoryIcons[action.category] || "♻️"}
                        </Badge>
                        <span>{action.name}</span>
                      </div>
                      <div className="text-xs text-gray-500">-{action.impact.co2Saved} kg CO₂</div>
                    </motion.li>
                  ))}
                </ul>
              </ScrollArea>
            </TabsContent>

            <TabsContent value="custom">
              <AnimatePresence>
                {showAddCustom ? (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="border border-dashed border-gray-300 rounded-lg p-4 mb-4"
                  >
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="custom-action">Action Name</Label>
                        <Input
                          id="custom-action"
                          placeholder="e.g., Used a bamboo toothbrush"
                          value={customAction}
                          onChange={(e) => setCustomAction(e.target.value)}
                        />
                      </div>

                      <div>
                        <Label>Category</Label>
                        <div className="flex flex-wrap gap-2 mt-2">
                          {categories.map((category) => (
                            <Badge
                              key={category}
                              variant={selectedCustomCategory === category ? "default" : "outline"}
                              onClick={() => setSelectedCustomCategory(category)}
                              className={cn(
                                "cursor-pointer",
                                selectedCustomCategory === category ? getCategoryColor(category) : "",
                              )}
                            >
                              {categoryIcons[category]} {category.charAt(0).toUpperCase() + category.slice(1)}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="co2-impact">CO₂ Saved (kg)</Label>
                          <Input
                            id="co2-impact"
                            type="number"
                            min="0"
                            step="0.1"
                            value={customImpact.co2Saved}
                            onChange={(e) =>
                              setCustomImpact({ ...customImpact, co2Saved: Number.parseFloat(e.target.value) || 0 })
                            }
                          />
                        </div>
                        <div>
                          <Label htmlFor="waste-impact">Waste Saved (kg)</Label>
                          <Input
                            id="waste-impact"
                            type="number"
                            min="0"
                            step="0.1"
                            value={customImpact.wasteSaved}
                            onChange={(e) =>
                              setCustomImpact({ ...customImpact, wasteSaved: Number.parseFloat(e.target.value) || 0 })
                            }
                          />
                        </div>
                        <div>
                          <Label htmlFor="energy-impact">Energy Saved (kWh)</Label>
                          <Input
                            id="energy-impact"
                            type="number"
                            min="0"
                            step="0.1"
                            value={customImpact.energySaved}
                            onChange={(e) =>
                              setCustomImpact({ ...customImpact, energySaved: Number.parseFloat(e.target.value) || 0 })
                            }
                          />
                        </div>
                        <div>
                          <Label htmlFor="water-impact">Water Saved (L)</Label>
                          <Input
                            id="water-impact"
                            type="number"
                            min="0"
                            step="0.1"
                            value={customImpact.waterSaved}
                            onChange={(e) =>
                              setCustomImpact({ ...customImpact, waterSaved: Number.parseFloat(e.target.value) || 0 })
                            }
                          />
                        </div>
                      </div>

                      <div className="flex justify-end gap-2">
                        <Button variant="outline" size="sm" onClick={() => setShowAddCustom(false)}>
                          <X className="h-4 w-4 mr-1" /> Cancel
                        </Button>
                        <Button size="sm" onClick={handleCreateCustomAction} disabled={!customAction.trim()}>
                          <Check className="h-4 w-4 mr-1" /> Save Action
                        </Button>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-6">
                    <Button variant="outline" onClick={() => setShowAddCustom(true)} className="gap-2">
                      <Plus className="h-4 w-4" /> Add Custom Action
                    </Button>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="text-sm text-gray-500 mt-4">
                <p>Custom actions allow you to track unique sustainable behaviors not in our presets.</p>
              </div>
            </TabsContent>
          </Tabs>
        </div>

        <div className="md:col-span-1">
          <EcoCard>
            <h3 className="text-lg font-semibold mb-4">Voice Logging</h3>
            <div className="space-y-4">
              <div className="flex justify-center">
                <Button
                  variant="outline"
                  size="lg"
                  className={`rounded-full p-6 ${isListening ? "bg-green-100 text-green-700 animate-pulse" : ""}`}
                  onClick={handleStartListening}
                  disabled={isListening}
                >
                  <Mic className={`h-6 w-6 ${isListening ? "text-green-700" : ""}`} />
                </Button>
              </div>

              <div className="text-center">
                {isListening ? (
                  <p className="text-green-600 animate-pulse">Listening...</p>
                ) : (
                  <p className="text-gray-500">Tap to record your eco-action</p>
                )}
              </div>

              <AnimatePresence>
                {voiceText && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    className="bg-green-50 p-4 rounded-lg border border-green-200"
                  >
                    <p className="text-green-800 font-medium">Recognized action:</p>
                    <p className="text-gray-700">{voiceText}</p>
                    <div className="flex justify-end mt-3">
                      <Button
                        variant="default"
                        size="sm"
                        onClick={() => {
                          const customAction: EcoAction = {
                            id: `voice-${Date.now()}`,
                            name: voiceText,
                            category: "other",
                            impact: { co2Saved: 1.0, wasteSaved: 0.2, energySaved: 0.5, waterSaved: 0.3 },
                            icon: "🗣️",
                          }
                          handleLogAction(customAction)
                          setVoiceText("")
                        }}
                      >
                        Log This Action
                      </Button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="text-xs text-gray-500 mt-4">
                <p>You can also log eco-actions with your voice. Just tap the microphone and speak clearly.</p>
              </div>
            </div>
          </EcoCard>

          <EcoCard className="mt-4">
            <h3 className="text-lg font-semibold mb-3">Today's Stats</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Actions logged</span>
                <span className="font-medium">
                  {
                    userData.logs.filter((log) => {
                      const logDate = new Date(log.date)
                      const today = new Date()
                      return (
                        logDate.getDate() === today.getDate() &&
                        logDate.getMonth() === today.getMonth() &&
                        logDate.getFullYear() === today.getFullYear()
                      )
                    }).length
                  }
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">CO₂ saved today</span>
                <span className="font-medium">
                  {userData.logs
                    .filter((log) => {
                      const logDate = new Date(log.date)
                      const today = new Date()
                      return (
                        logDate.getDate() === today.getDate() &&
                        logDate.getMonth() === today.getMonth() &&
                        logDate.getFullYear() === today.getFullYear()
                      )
                    })
                    .reduce((total, log) => total + log.impact.co2Saved, 0)
                    .toFixed(1)}{" "}
                  kg
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Current streak</span>
                <span className="font-medium">{userData.streak} days</span>
              </div>
            </div>
          </EcoCard>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Action</DialogTitle>
          </DialogHeader>
          <p>Are you sure you want to delete "{actionToDelete?.name}"?</p>
          <p className="text-sm text-gray-500">This will remove the action from your available actions list.</p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowDeleteDialog(false)}>
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleDeleteAction}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

