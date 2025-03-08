"use client"

import { BookOpen, BarChart2, Award, Users, Leaf, CircleCheckBig } from "lucide-react"
import ScrollReveal from "./ScrollReveal"

const features = [
  {
    icon: <BookOpen className="h-10 w-10 text-green-600" />,
    title: "Educational Content",
    description:
      "Access comprehensive educational materials about environmental conservation, climate change, and sustainable practices.",
  },
  {
    icon: <BarChart2 className="h-10 w-10 text-green-600" />,
    title: "CO2 Emissions Predictor",
    description:
      "Calculate the carbon footprint of your vehicle and receive personalized suggestions to reduce your environmental impact.",
  },
  {
    icon: <Award className="h-10 w-10 text-green-600" />,
    title: "Interactive Quizzes",
    description:
      "Test your knowledge with our interactive quizzes and track your progress as you learn about environmental topics.",
  },
  {
    icon: <CircleCheckBig className="h-10 w-10 text-green-600" />,
    title: "Keep It Green Tracker",
    description:
      "Track your sustainable actions, gain personalized insights, and get expert guidance from the AI Eco-Assistant to make eco-friendly living effortless and impactful.",
  },
  {
    icon: <Leaf className="h-10 w-10 text-green-600" />,
    title: "Eco-Tips and Resources",
    description:
      "Discover practical eco-tips, to help you integrate sustainable practices into your daily life.",
  },
]

export default function FeatureSection() {
  return (
    <section className="py-20 bg-white relative">
      <div className="container mx-auto px-4">
        <ScrollReveal>
          <h2 className="text-3xl md:text-4xl font-bold text-center gradient-text mb-16">Our Features</h2>
        </ScrollReveal>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <ScrollReveal key={index} delay={index * 200} direction={index % 2 === 0 ? "left" : "right"}>
              <div className="bg-white rounded-xl p-6 shadow-lg hover:shadow-xl transition-all duration-300 card-3d border border-gray-100 h-full">
                <div className="mb-6 bg-green-50 w-16 h-16 rounded-full flex items-center justify-center transform transition-transform hover:scale-110">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold text-green-700 mb-3">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            </ScrollReveal>
          ))}
        </div>
      </div>
    </section>
  )
}

