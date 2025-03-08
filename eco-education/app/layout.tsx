import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import Navbar from "@/components/Navbar"
import Footer from "@/components/Footer"
import AnimatedBackground from "@/components/AnimatedBackground"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "EcoLearn - Environmental Education Platform",
  description: "Learn about environmental conservation and sustainability",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-green-50 overflow-x-hidden`}>
        <AnimatedBackground />
        <Navbar />
        <main className="min-h-screen relative z-10">{children}</main>
        <Footer />
      </body>
    </html>
  )
}



import './globals.css'