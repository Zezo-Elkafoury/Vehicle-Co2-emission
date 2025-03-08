"use client"

import Link from "next/link"
import { ArrowRight } from "lucide-react"
import ScrollReveal from "./ScrollReveal"

export default function CallToAction() {
  return (
    <section className="py-20 relative overflow-hidden">
      {/* Animated background */}
      <div className="animated-bg">
        <div className="animated-bg-shape"></div>
        <div className="animated-bg-shape"></div>
      </div>

      <div className="container mx-auto px-4 text-center relative z-10">
        <div className="max-w-3xl mx-auto glass rounded-2xl p-10 shadow-xl">
          <ScrollReveal>
            <h2 className="text-3xl md:text-4xl font-bold gradient-text mb-6">Ready to Make a Difference?</h2>
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <p className="text-xl text-gray-700 dark:text-gray-200 mb-8">
              Start your journey towards environmental awareness and sustainability today. Every small action counts
              towards a greener future.
            </p>
          </ScrollReveal>

          <ScrollReveal delay={400}>
            <Link
              href="/content"
              className="gradient-btn inline-flex items-center text-white font-bold py-4 px-8 rounded-full transition-all duration-300 group"
            >
              Begin Your Learning Journey
              <ArrowRight className="ml-2 h-5 w-5 transform transition-transform group-hover:translate-x-1" />
            </Link>
          </ScrollReveal>
        </div>
      </div>
    </section>
  )
}

