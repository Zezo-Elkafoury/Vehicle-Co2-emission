"use client"

import { useEffect, useRef } from "react"
import Link from "next/link"
import Image from "next/image"
import { ArrowRight } from "lucide-react"

export default function Hero() {
  const heroRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!heroRef.current) return

      const { clientX, clientY } = e
      const { width, height, left, top } = heroRef.current.getBoundingClientRect()

      const x = (clientX - left) / width
      const y = (clientY - top) / height

      const moveX = (x - 0.5) * 20
      const moveY = (y - 0.5) * 20

      heroRef.current.style.setProperty("--move-x", `${moveX}px`)
      heroRef.current.style.setProperty("--move-y", `${moveY}px`)
    }

    document.addEventListener("mousemove", handleMouseMove)

    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
    }
  }, [])

  return (
    <div ref={heroRef} className="relative overflow-hidden">
      {/* Animated background shapes */}
      <div className="animated-bg">
        <div className="animated-bg-shape"></div>
        <div className="animated-bg-shape"></div>
        <div className="animated-bg-shape"></div>
      </div>

      <div className="container mx-auto px-4 py-24 md:py-32 relative z-10">
        <div className="max-w-3xl mx-auto text-center">
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight animate-fadeIn gradient-text">
            Learn, Understand, and Protect Our Environment
          </h1>
          <p
            className="text-xl md:text-2xl mb-8 text-gray-700 dark:text-gray-200 animate-fadeIn opacity-0"
            style={{ animationDelay: "0.3s" }}
          >
            Join our community of environmental enthusiasts and learn how your actions can make a positive impact on our
            planet.
          </p>
          <div
            className="flex flex-col sm:flex-row justify-center space-y-4 sm:space-y-0 sm:space-x-4 animate-fadeIn opacity-0"
            style={{ animationDelay: "0.6s" }}
          >
            <Link
              href="/content"
              className="gradient-btn text-white font-bold py-4 px-8 rounded-full transition-all duration-300 flex items-center justify-center group"
            >
              Start Learning
              <ArrowRight className="ml-2 h-5 w-5 transform transition-transform group-hover:translate-x-1" />
            </Link>
            <Link
              href="/calc"
              className="bg-transparent hover:bg-green-600 border-2 border-green-500 text-green-600 hover:text-white font-bold py-4 px-8 rounded-full transition-all duration-300 flex items-center justify-center"
            >
              CO2 Calculator
            </Link>
          </div>
        </div>

        {/* Floating images */}
        <div className="hidden md:block">
          <div className="absolute top-1/4 -left-16 w-40 h-40 animate-float" style={{ animationDelay: "0s" }}>
            <Image
              src="/placeholder.svg?height=160&width=160"
              alt="Leaf"
              width={160}
              height={160}
              className="opacity-20 dark:opacity-10 transform rotate-12"
            />
          </div>
          <div className="absolute bottom-1/4 -right-16 w-48 h-48 animate-float" style={{ animationDelay: "1s" }}>
            <Image
              src="/placeholder.svg?height=180&width=180"
              alt="Tree"
              width={180}
              height={180}
              className="opacity-20 dark:opacity-10 transform -rotate-12"
            />
          </div>
        </div>
      </div>

      {/* Wave divider */}
      <div className="absolute bottom-0 left-0 w-full overflow-hidden">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320" className="w-full h-auto">
          <path
            fill="#ffffff"
            fillOpacity="1"
            d="M0,96L48,112C96,128,192,160,288,160C384,160,480,128,576,122.7C672,117,768,139,864,149.3C960,160,1056,160,1152,138.7C1248,117,1344,75,1392,53.3L1440,32L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"
          ></path>
        </svg>
      </div>
    </div>
  )
}

