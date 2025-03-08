"use client"

import { useState, useEffect, useMemo } from "react"
import { useUser } from "./UserContext"
import { CalendarIcon, Filter, Search } from "lucide-react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger } from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Calendar } from "@/components/ui/calendar"
import { Badge } from "@/components/ui/badge"
import { format, compareDesc, isSameDay } from "date-fns"
import { categoryIcons, getCategoryColor } from "@/lib/eco-tracker/utils"
import { motion, AnimatePresence } from "framer-motion"
import { EcoCard } from "./EcoCard"

export default function ActionTimeline() {
  const { userData } = useUser()
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [selectedDate, setSelectedDate] = useState<Date | null>(null)
  const [update, setUpdate] = useState(false)

  useEffect(() => {
    setUpdate((prev) => !prev)
  }, [userData.logs.length])

  const categories = useMemo(() => {
    return Array.from(new Set(userData.logs.map((log) => log.category)))
  }, [userData.logs])

  const filteredLogs = useMemo(() => {
    return userData.logs.filter((log) => {
      const matchesSearch = searchQuery === "" || log.actionName.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesCategory = selectedCategory === null || log.category === selectedCategory
      const matchesDate = selectedDate === null || isSameDay(new Date(log.date), selectedDate)
      return matchesSearch && matchesCategory && matchesDate
    })
  }, [userData.logs, searchQuery, selectedCategory, selectedDate])

  const groupedLogs = useMemo(() => {
    const grouped: Record<string, typeof userData.logs> = {}

    filteredLogs
      .sort((a, b) => compareDesc(new Date(a.date), new Date(b.date)))
      .forEach((log) => {
        const dateKey = format(new Date(log.date), "yyyy-MM-dd")
        if (!grouped[dateKey]) {
          grouped[dateKey] = []
        }
        grouped[dateKey].push(log)
      })

    return grouped
  }, [filteredLogs])

  return (
    <div className="space-y-6">
      <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold gradient-text mb-2">Your Actions Timeline</h2>
          <p className="text-gray-600 dark:text-gray-300">Track your sustainability journey over time</p>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        <div className="relative flex-grow">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search actions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>

        <Select
          value={selectedCategory || "All Categories"}
          onValueChange={(val) => setSelectedCategory(val === "All Categories" ? null : val)}
        >
          <SelectTrigger className="w-full sm:w-[180px]">
            <div className="flex items-center gap-2">
              <Filter className="h-4 w-4" />
              <span>
                {selectedCategory
                  ? `${selectedCategory.charAt(0).toUpperCase()}${selectedCategory.slice(1)}`
                  : "All Categories"}
              </span>
            </div>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="All Categories">All Categories</SelectItem>
            {categories.map((category) => (
              <SelectItem key={category} value={category}>
                {categoryIcons[category]} {category.charAt(0).toUpperCase() + category.slice(1)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Popover>
          <PopoverTrigger asChild>
            <Button variant="outline" className="w-full sm:w-auto">
              <CalendarIcon className="h-4 w-4 mr-2" />
              {selectedDate ? format(selectedDate, "PP") : "Filter by date"}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0" align="end">
            <Calendar mode="single" selected={selectedDate || undefined} onSelect={setSelectedDate} initialFocus />
            {selectedDate && (
              <div className="p-3 border-t border-gray-100">
                <Button variant="ghost" className="w-full justify-start text-xs" onClick={() => setSelectedDate(null)}>
                  Clear date filter
                </Button>
              </div>
            )}
          </PopoverContent>
        </Popover>
      </div>

      <ScrollArea className="h-[500px] pr-4">
        <AnimatePresence>
          {Object.keys(groupedLogs).length > 0 ? (
            Object.entries(groupedLogs).map(([dateKey, logs], groupIndex) => (
              <motion.div
                key={dateKey}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.3, delay: groupIndex * 0.05 }}
                className="mb-8"
              >
                <div className="flex items-center gap-4 mb-4">
                  <div className="bg-green-100 text-green-800 rounded-full h-10 w-10 flex items-center justify-center font-semibold">
                    {format(new Date(dateKey), "d")}
                  </div>
                  <div>
                    <h3 className="font-semibold">{format(new Date(dateKey), "EEEE")}</h3>
                    <p className="text-sm text-gray-500">{format(new Date(dateKey), "MMMM yyyy")}</p>
                  </div>
                </div>

                <EcoCard>
                  <ul className="space-y-4">
                    {logs.map((log, index) => (
                      <motion.li
                        key={log.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3, delay: 0.1 + index * 0.05 }}
                        className="flex items-start gap-4 pb-4 border-b border-gray-100 last:border-0 last:pb-0"
                      >
                        <div className="flex-shrink-0 mt-1">
                          <Badge className={getCategoryColor(log.category)}>{categoryIcons[log.category] || "♻️"}</Badge>
                        </div>

                        <div className="flex-grow">
                          <div className="flex justify-between">
                            <h4 className="font-medium">{log.actionName}</h4>
                            <span className="text-sm text-gray-500">{format(new Date(log.date), "h:mm a")}</span>
                          </div>

                          <div className="mt-1 flex flex-wrap gap-2">
                            {log.impact.co2Saved > 0 && (
                              <span className="inline-block text-xs bg-green-50 text-green-700 px-2 py-1 rounded-full">
                                -{log.impact.co2Saved.toFixed(1)} kg CO₂
                              </span>
                            )}
                            {log.impact.wasteSaved > 0 && (
                              <span className="inline-block text-xs bg-amber-50 text-amber-700 px-2 py-1 rounded-full">
                                -{log.impact.wasteSaved.toFixed(1)} kg waste
                              </span>
                            )}
                            {log.impact.energySaved > 0 && (
                              <span className="inline-block text-xs bg-yellow-50 text-yellow-700 px-2 py-1 rounded-full">
                                -{log.impact.energySaved.toFixed(1)} kWh energy
                              </span>
                            )}
                            {log.impact.waterSaved > 0 && (
                              <span className="inline-block text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded-full">
                                -{log.impact.waterSaved.toFixed(1)} L water
                              </span>
                            )}
                          </div>

                          {log.notes && <p className="text-sm text-gray-600 mt-1">{log.notes}</p>}
                        </div>
                      </motion.li>
                    ))}
                  </ul>
                </EcoCard>
              </motion.div>
            ))
          ) : (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-12">
              <p className="text-gray-500">No actions found matching your filters.</p>
              <Button
                variant="outline"
                className="mt-4"
                onClick={() => {
                  setSearchQuery("")
                  setSelectedCategory(null)
                  setSelectedDate(null)
                }}
              >
                Clear All Filters
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </ScrollArea>
    </div>
  )
}

