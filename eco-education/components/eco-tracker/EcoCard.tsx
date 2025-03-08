"use client"

import type React from "react"

import { cn } from "@/lib/utils"
import { motion } from "framer-motion"

interface EcoCardProps {
  children: React.ReactNode
  className?: string
}

export function EcoCard({ children, className }: EcoCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={cn(
        "glass rounded-2xl shadow-xl overflow-hidden transition-all duration-300 hover:shadow-2xl",
        className,
      )}
    >
      <div className="p-6">{children}</div>
    </motion.div>
  )
}

