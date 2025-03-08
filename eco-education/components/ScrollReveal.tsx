"use client"

import { useEffect, useRef, type ReactNode } from "react"

type ScrollRevealProps = {
  children: ReactNode
  className?: string
  delay?: number
  direction?: "up" | "down" | "left" | "right" | "none"
}

export default function ScrollReveal({ children, className = "", delay = 0, direction = "up" }: ScrollRevealProps) {
  const elementRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setTimeout(() => {
              entry.target.classList.add("active")
            }, delay)
            observer.unobserve(entry.target)
          }
        })
      },
      {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px",
      },
    )

    if (elementRef.current) {
      observer.observe(elementRef.current)
    }

    return () => {
      if (elementRef.current) {
        observer.unobserve(elementRef.current)
      }
    }
  }, [delay])

  const getDirectionClass = () => {
    switch (direction) {
      case "up":
        return "translate-y-16"
      case "down":
        return "-translate-y-16"
      case "left":
        return "translate-x-16"
      case "right":
        return "-translate-x-16"
      default:
        return "opacity-0"
    }
  }

  return (
    <div
      ref={elementRef}
      className={`reveal ${getDirectionClass()} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  )
}

