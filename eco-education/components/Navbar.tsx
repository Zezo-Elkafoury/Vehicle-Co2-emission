"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { LeafyGreen, Menu, X } from "lucide-react"

export default function Navbar() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setScrolled(true)
      } else {
        setScrolled(false)
      }
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  return (
    <nav
      className={`fixed top-0 left-0 w-full z-50 transition-all duration-300 ${
        scrolled ? "bg-white/90 dark:bg-gray-900/90 backdrop-blur-md shadow-md py-2" : "bg-transparent py-4"
      }`}
    >
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center">
          <Link href="/" className="flex items-center space-x-2 group">
            <div className="bg-green-500 rounded-full p-2 transform transition-transform group-hover:rotate-12">
              <LeafyGreen className="h-6 w-6 text-white" />
            </div>
            <span
              className={`font-bold text-xl ${scrolled ? "text-green-700 dark:text-green-400" : "text-green-700 dark:text-white"}`}
            >
              EcoLearn
            </span>
          </Link>

          {/* Desktop menu */}
          <div className="hidden md:flex items-center space-x-8">
            <Link
              href="/"
              className={`${scrolled ? "text-gray-700 dark:text-gray-300" : "text-gray-700 dark:text-white"} hover:text-green-500 dark:hover:text-green-400 transition-colors relative group`}
            >
              Home
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-green-500 transition-all duration-300 group-hover:w-full"></span>
            </Link>
            <Link
              href="/content"
              className={`${scrolled ? "text-gray-700 dark:text-gray-300" : "text-gray-700 dark:text-white"} hover:text-green-500 dark:hover:text-green-400 transition-colors relative group`}
            >
              Educational Content
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-green-500 transition-all duration-300 group-hover:w-full"></span>
            </Link>
            <Link
              href="/calc"
              className={`${scrolled ? "text-gray-700 dark:text-gray-300" : "text-gray-700 dark:text-white"} hover:text-green-500 dark:hover:text-green-400 transition-colors relative group`}
            >
              Vehicle CO2 Calculator
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-green-500 transition-all duration-300 group-hover:w-full"></span>
            </Link>
            <Link
              href="/eco-tracker"
              className={`${scrolled ? "text-gray-700 dark:text-gray-300" : "text-gray-700 dark:text-white"} hover:text-green-500 dark:hover:text-green-400 transition-colors relative group`}
            >
              Keep It green Tracker
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-green-500 transition-all duration-300 group-hover:w-full"></span>
            </Link>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className={`${scrolled ? "text-gray-700 dark:text-gray-300" : "text-gray-700 dark:text-white"} focus:outline-none`}
            >
              {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      {isMenuOpen && (
        <div className="md:hidden bg-white dark:bg-gray-800 shadow-lg py-4 animate-slideInRight">
          <div className="container mx-auto px-4 flex flex-col space-y-4">
            <Link
              href="/"
              className="block py-2 px-4 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-lg transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              Home
            </Link>
            <Link
              href="/content"
              className="block py-2 px-4 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-lg transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              Educational Content
            </Link>
            <Link
              href="/calc"
              className="block py-2 px-4 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-lg transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              CO2 Calculator
            </Link>
            <Link
              href="/eco-tracker"
              className="block py-2 px-4 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-lg transition-colors"
              onClick={() => setIsMenuOpen(false)}
            >
              Eco Tracker
            </Link>
          </div>
        </div>
      )}
    </nav>
  )
}

