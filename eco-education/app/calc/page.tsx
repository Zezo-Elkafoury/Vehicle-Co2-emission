"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { AlertTriangle, Check, Info, Car, Zap, Leaf, BarChart2, Droplet, Sparkles } from "lucide-react"
import confetti from "canvas-confetti"
import ScrollReveal from "@/components/ScrollReveal"

// Define types for form data and API response
type VehicleData = {
  Make?: string
  Vehicle_Class?: string
  Engine_Size?: string
  Cylinders?: string
  Transmission?: string
  Fuel_Type?: string
  Fuel_Consumption_City?: string
  Fuel_Consumption_Hwy?: string
  Fuel_Consumption_Comb?: string
}

type EmissionResult = {
  CO2_Emissions_Prediction: number
  Reduction_Tips: string[]
}

export default function CalcPage() {
  const [formData, setFormData] = useState<VehicleData>({})
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<EmissionResult | null>(null)
  const [filledFieldsCount, setFilledFieldsCount] = useState(0)
  const [showAnimation, setShowAnimation] = useState(false)

  // Count filled fields whenever formData changes
  useEffect(() => {
    const count = Object.values(formData).filter((value) => value && value.toString().trim() !== "").length
    setFilledFieldsCount(count)
  }, [formData])

  // Trigger confetti when results are shown
  useEffect(() => {
    if (result) {
      confetti({
        particleCount: 150,
        spread: 80,
        origin: { y: 0.6 },
        colors: ["#2ecc71", "#3498db", "#9b59b6"],
      })
    }
  }, [result])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    // Validate that at least 2 fields are filled
    if (filledFieldsCount < 2) {
      setError("Please fill at least 2 fields to get a prediction.")
      return
    }

    setIsLoading(true)
    setError(null)
    setShowAnimation(true)

    try {
      const response = await fetch("https://underlying-tiffany-ziad-2007-547ae087.koyeb.app/co2_predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        throw new Error("Failed to get prediction. Please try again.")
      }

      // Add a slight delay to show the loading animation
      await new Promise((resolve) => setTimeout(resolve, 1500))

      const result = await response.json()
      setResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsLoading(false)
      setShowAnimation(false)
    }
  }

  // Determine emission level category
  const getEmissionCategory = (emissions: number) => {
    if (emissions < 120) return { label: "Low", color: "green" }
    if (emissions < 160) return { label: "Moderate", color: "yellow" }
    if (emissions < 200) return { label: "High", color: "orange" }
    return { label: "Very High", color: "red" }
  }

  // Vehicle class options
  const vehicleClassOptions = [
    "COMPACT",
    "SUV - SMALL",
    "MID-SIZE",
    "TWO-SEATER",
    "MINICOMPACT",
    "SUBCOMPACT",
    "FULL-SIZE",
    "STATION WAGON - SMALL",
    "SUV - STANDARD",
    "VAN - CARGO",
    "VAN - PASSENGER",
    "PICKUP TRUCK - STANDARD",
    "MINIVAN",
    "SPECIAL PURPOSE VEHICLE",
    "STATION WAGON - MID-SIZE",
    "PICKUP TRUCK - SMALL",
  ]

  // Transmission options
  const transmissionOptions = [
    "AS5",
    "M6",
    "AV7",
    "AS6",
    "AM6",
    "A6",
    "AM7",
    "AV8",
    "AS8",
    "A7",
    "A8",
    "M7",
    "A4",
    "M5",
    "AV",
    "A5",
    "AS7",
    "A9",
    "AS9",
    "AV6",
    "AS4",
    "AM5",
    "AM8",
    "AM9",
    "AS10",
    "A10",
    "AV10",
  ]

  // Fuel type options with descriptions
  const fuelTypeOptions = [
    { value: "Z", label: "Premium Gasoline" },
    { value: "D", label: "Diesel" },
    { value: "X", label: "Regular Gasoline" },
    { value: "E", label: "Ethanol" },
    { value: "N", label: "Natural Gas" },
  ]

  // Pre-fill some example values for demonstration
  const fillExampleValues = () => {
    setFormData({
      Make: "Toyota",
      Vehicle_Class: "COMPACT",
      Engine_Size: "2.0",
      Fuel_Type: "X",
    })
  }

  return (
    <div className="relative min-h-screen pt-20">
      {/* Animated background */}
      <div className="animated-bg">
        <div className="animated-bg-shape"></div>
        <div className="animated-bg-shape"></div>
        <div className="animated-bg-shape"></div>
      </div>

      <div className="container mx-auto px-4 py-8 relative z-10">
        <ScrollReveal>
          <div className="text-center mb-12">
            <h1 className="text-4xl md:text-5xl font-bold gradient-text mb-4">Vehicle CO2 Emissions Calculator</h1>
            <p className="text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
              Discover your vehicle's environmental impact and learn how to reduce your carbon footprint, and know the effect of your future vehicle on the enviroment before you buy it
            </p>
          </div>
        </ScrollReveal>

        <div className="max-w-4xl mx-auto">
          <ScrollReveal delay={200}>
            <div className="glass rounded-2xl shadow-xl overflow-hidden transition-all duration-300 hover:shadow-2xl mb-8">
              <div className="bg-gradient-to-r from-green-500 to-blue-500 p-4 flex items-center">
                <Car className="h-8 w-8 text-white mr-3" />
                <h2 className="text-2xl font-bold text-white">Vehicle Information</h2>
              </div>

              <div className="p-6">
                <p className="mb-6 text-gray-600 dark:text-gray-300">
                  Enter your vehicle details below to get an estimate of CO2 emissions and personalized suggestions for
                  reducing your carbon footprint. <span className="font-semibold">Please fill at least 2 fields</span>{" "}
                  to get a prediction.
                </p>

                <div className="bg-blue-50 dark:bg-blue-900/30 border-l-4 border-blue-400 dark:border-blue-500 p-4 mb-6 rounded-r-lg transform transition-transform hover:scale-102 hover:shadow-md">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <Info className="h-5 w-5 text-blue-500 dark:text-blue-400" />
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        Fields filled:
                        <span className="ml-1 font-medium">{filledFieldsCount}/9</span>
                        <span className="ml-2 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 text-xs px-2 py-0.5 rounded-full">
                          {filledFieldsCount < 2 ? "Need " + (2 - filledFieldsCount) + " more" : "Ready to calculate!"}
                        </span>
                        {filledFieldsCount < 2 && (
                          <button
                            onClick={fillExampleValues}
                            className="ml-2 text-blue-600 dark:text-blue-400 underline hover:text-blue-800 dark:hover:text-blue-300 transition-colors"
                          >
                            Fill example values
                          </button>
                        )}
                      </p>
                    </div>
                  </div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="group">
                      <label
                        htmlFor="Make"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors"
                      >
                        <Car className="h-4 w-4 inline mr-1" /> Manufacturer
                      </label>
                      <input
                        type="text"
                        id="Make"
                        name="Make"
                        value={formData.Make || ""}
                        onChange={handleChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 dark:bg-gray-700 dark:text-white transition-all duration-200 hover:border-green-400"
                        placeholder="e.g., Toyota, Honda"
                      />
                    </div>

                    <div className="group">
                      <label
                        htmlFor="Vehicle_Class"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors"
                      >
                        <Car className="h-4 w-4 inline mr-1" /> Vehicle Class
                      </label>
                      <select
                        id="Vehicle_Class"
                        name="Vehicle_Class"
                        value={formData.Vehicle_Class || ""}
                        onChange={handleChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 dark:bg-gray-700 dark:text-white transition-all duration-200 hover:border-green-400"
                      >
                        <option value="">Select Vehicle Class</option>
                        {vehicleClassOptions.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="group">
                      <label
                        htmlFor="Engine_Size"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors"
                      >
                        <Zap className="h-4 w-4 inline mr-1" /> Engine Size (L)
                      </label>
                      <input
                        type="number"
                        id="Engine_Size"
                        name="Engine_Size"
                        value={formData.Engine_Size || ""}
                        onChange={handleChange}
                        step="0.1"
                        min="0.5"
                        max="10"
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 dark:bg-gray-700 dark:text-white transition-all duration-200 hover:border-green-400"
                        placeholder="e.g., 2.0"
                      />
                    </div>

                    <div className="group">
                      <label
                        htmlFor="Cylinders"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors"
                      >
                        <Zap className="h-4 w-4 inline mr-1" /> Cylinders
                      </label>
                      <input
                        type="number"
                        id="Cylinders"
                        name="Cylinders"
                        value={formData.Cylinders || ""}
                        onChange={handleChange}
                        min="3"
                        max="12"
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 dark:bg-gray-700 dark:text-white transition-all duration-200 hover:border-green-400"
                        placeholder="e.g., 4"
                      />
                    </div>

                    <div className="group">
                      <label
                        htmlFor="Transmission"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors"
                      >
                        <Zap className="h-4 w-4 inline mr-1" /> Transmission
                      </label>
                      <select
                        id="Transmission"
                        name="Transmission"
                        value={formData.Transmission || ""}
                        onChange={handleChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 dark:bg-gray-700 dark:text-white transition-all duration-200 hover:border-green-400"
                      >
                        <option value="">Select Transmission</option>
                        {transmissionOptions.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="group">
                      <label
                        htmlFor="Fuel_Type"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors"
                      >
                        <Droplet className="h-4 w-4 inline mr-1" /> Fuel Type
                      </label>
                      <select
                        id="Fuel_Type"
                        name="Fuel_Type"
                        value={formData.Fuel_Type || ""}
                        onChange={handleChange}
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 dark:bg-gray-700 dark:text-white transition-all duration-200 hover:border-green-400"
                      >
                        <option value="">Select Fuel Type</option>
                        {fuelTypeOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.value} - {option.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="group">
                      <label
                        htmlFor="Fuel_Consumption_City"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors"
                      >
                        <Droplet className="h-4 w-4 inline mr-1" /> Fuel Consumption City (L/100km)
                      </label>
                      <input
                        type="number"
                        id="Fuel_Consumption_City"
                        name="Fuel_Consumption_City"
                        value={formData.Fuel_Consumption_City || ""}
                        onChange={handleChange}
                        step="0.1"
                        min="1"
                        max="30"
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 dark:bg-gray-700 dark:text-white transition-all duration-200 hover:border-green-400"
                        placeholder="e.g., 9.5"
                      />
                    </div>

                    <div className="group">
                      <label
                        htmlFor="Fuel_Consumption_Hwy"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors"
                      >
                        <Droplet className="h-4 w-4 inline mr-1" /> Fuel Consumption Highway (L/100km)
                      </label>
                      <input
                        type="number"
                        id="Fuel_Consumption_Hwy"
                        name="Fuel_Consumption_Hwy"
                        value={formData.Fuel_Consumption_Hwy || ""}
                        onChange={handleChange}
                        step="0.1"
                        min="1"
                        max="30"
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 dark:bg-gray-700 dark:text-white transition-all duration-200 hover:border-green-400"
                        placeholder="e.g., 7.2"
                      />
                    </div>

                    <div className="group">
                      <label
                        htmlFor="Fuel_Consumption_Comb"
                        className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 group-hover:text-green-600 dark:group-hover:text-green-400 transition-colors"
                      >
                        <Droplet className="h-4 w-4 inline mr-1" /> Fuel Consumption Combined (L/100km)
                      </label>
                      <input
                        type="number"
                        id="Fuel_Consumption_Comb"
                        name="Fuel_Consumption_Comb"
                        value={formData.Fuel_Consumption_Comb || ""}
                        onChange={handleChange}
                        step="0.1"
                        min="1"
                        max="30"
                        className="w-full px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500 dark:bg-gray-700 dark:text-white transition-all duration-200 hover:border-green-400"
                        placeholder="e.g., 8.3"
                      />
                    </div>
                  </div>

                  <div className="flex justify-center mt-8">
                    <button
                      type="submit"
                      disabled={isLoading || filledFieldsCount < 2}
                      className={`relative overflow-hidden gradient-btn text-white font-bold py-3 px-8 rounded-full transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 shadow-lg ${
                        isLoading || filledFieldsCount < 2 ? "opacity-70 cursor-not-allowed" : ""
                      }`}
                    >
                      <span className="relative z-10 flex items-center justify-center">
                        {isLoading ? (
                          <>
                            <svg
                              className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                              xmlns="http://www.w3.org/2000/svg"
                              fill="none"
                              viewBox="0 0 24 24"
                            >
                              <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                              ></circle>
                              <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                              ></path>
                            </svg>
                            Calculating...
                          </>
                        ) : (
                          <>
                            <BarChart2 className="mr-2 h-5 w-5" />
                            Calculate CO2 Emissions
                          </>
                        )}
                      </span>
                      {!isLoading && (
                        <span className="absolute top-0 left-0 w-full h-full bg-white opacity-20 transform -translate-x-full skew-x-12 transition-transform duration-700 ease-out group-hover:translate-x-0"></span>
                      )}
                    </button>
                  </div>
                </form>
              </div>
            </div>
          </ScrollReveal>

          {error && (
            <div
              className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 text-red-700 dark:text-red-300 px-4 py-3 rounded-lg relative mb-6 animate-shake"
              role="alert"
            >
              <span className="block sm:inline">{error}</span>
            </div>
          )}

          {showAnimation && !result && (
            <div className="flex flex-col items-center justify-center p-8 animate-pulse">
              <div className="w-24 h-24 rounded-full bg-gradient-to-r from-green-400 to-blue-500 mb-4 animate-spin"></div>
              <p className="text-gray-600 dark:text-gray-300">Calculating your environmental impact...</p>
            </div>
          )}

          {result && (
            <ScrollReveal>
              <div className="glass rounded-2xl shadow-xl overflow-hidden transition-all duration-500 animate-slideUp">
                <div className="bg-gradient-to-r from-green-500 to-blue-500 p-4 flex items-center justify-center">
                  <Sparkles className="h-6 w-6 text-white mr-2" />
                  <h2 className="text-2xl font-bold text-white text-center">CO2 Emissions Result</h2>
                </div>

                <div className="p-6">
                  <div className="flex flex-col items-center mb-8">
                    {result.CO2_Emissions_Prediction && (
                      <>
                        {(() => {
                          const emissions = result.CO2_Emissions_Prediction
                          const category = getEmissionCategory(emissions)

                          return (
                            <>
                              <div
                                className={`w-48 h-48 rounded-full flex items-center justify-center mb-6 shadow-lg transform transition-transform hover:scale-105 animate-fadeIn ${
                                  category.color === "green"
                                    ? "bg-gradient-to-br from-green-100 to-green-300 text-green-700 dark:from-green-900 dark:to-green-700 dark:text-green-100"
                                    : category.color === "yellow"
                                      ? "bg-gradient-to-br from-yellow-100 to-yellow-300 text-yellow-700 dark:from-yellow-900 dark:to-yellow-700 dark:text-yellow-100"
                                      : category.color === "orange"
                                        ? "bg-gradient-to-br from-orange-100 to-orange-300 text-orange-700 dark:from-orange-900 dark:to-orange-700 dark:text-orange-100"
                                        : "bg-gradient-to-br from-red-100 to-red-300 text-red-700 dark:from-red-900 dark:to-red-700 dark:text-red-100"
                                }`}
                              >
                                <div className="text-center">
                                  <div className="text-5xl font-bold animate-numberCount">{Math.round(emissions)}</div>
                                  <div className="text-lg mt-1">g/km</div>
                                </div>
                              </div>

                              <div
                                className={`px-6 py-2 rounded-full text-md font-medium shadow-md transform transition-transform hover:scale-105 ${
                                  category.color === "green"
                                    ? "bg-green-100 text-green-700 dark:bg-green-800 dark:text-green-100"
                                    : category.color === "yellow"
                                      ? "bg-yellow-100 text-yellow-700 dark:bg-yellow-800 dark:text-yellow-100"
                                      : category.color === "orange"
                                        ? "bg-orange-100 text-orange-700 dark:bg-orange-800 dark:text-orange-100"
                                        : "bg-red-100 text-red-700 dark:bg-red-800 dark:text-red-100"
                                }`}
                              >
                                <span className="flex items-center">
                                  {category.color === "green" ? (
                                    <Leaf className="h-5 w-5 mr-2" />
                                  ) : (
                                    <AlertTriangle className="h-5 w-5 mr-2" />
                                  )}
                                  {category.label} Emissions
                                </span>
                              </div>

                              <div className="bg-green-50 dark:bg-green-900/30 rounded-lg p-6 mb-6 mt-8 w-full shadow-md transform transition-all hover:shadow-lg animate-fadeIn">
                                <h3 className="font-semibold text-green-800 dark:text-green-300 mb-3 flex items-center text-lg">
                                  <AlertTriangle className="h-5 w-5 mr-2" />
                                  Environmental Impact
                                </h3>
                                <p className="text-gray-700 dark:text-gray-300">
                                  A vehicle emitting <span className="font-bold">{Math.round(emissions)} g/km</span> of
                                  CO2 will produce approximately{" "}
                                  <span className="font-bold">{((emissions * 15000) / 1000000).toFixed(2)} tonnes</span>{" "}
                                  of CO2 per year (based on 15,000 km driven).
                                </p>

                                <div className="mt-4 bg-white dark:bg-gray-800 p-3 rounded-lg">
                                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                                    Emissions Comparison
                                  </h4>
                                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4">
                                    <div
                                      className={`h-4 rounded-full ${
                                        category.color === "green"
                                          ? "bg-green-500"
                                          : category.color === "yellow"
                                            ? "bg-yellow-500"
                                            : category.color === "orange"
                                              ? "bg-orange-500"
                                              : "bg-red-500"
                                      }`}
                                      style={{ width: `${Math.min(100, (emissions / 300) * 100)}%` }}
                                    ></div>
                                  </div>
                                  <div className="flex justify-between text-xs mt-1">
                                    <span className="text-green-600 dark:text-green-400">Low (0 g/km)</span>
                                    <span className="text-yellow-600 dark:text-yellow-400">Moderate (150 g/km)</span>
                                    <span className="text-red-600 dark:text-red-400">High (300+ g/km)</span>
                                  </div>
                                </div>
                              </div>
                            </>
                          )
                        })()}
                      </>
                    )}
                  </div>

                  <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-md transform transition-all hover:shadow-lg animate-fadeIn">
                    <h3 className="font-semibold gradient-text mb-4 flex items-center text-lg">
                      <Check className="h-5 w-5 mr-2" />
                      Suggestions to Reduce Emissions
                    </h3>

                    <ul className="space-y-4">
                      {result.Reduction_Tips &&
                        result.Reduction_Tips.map((suggestion, index) => (
                          <li
                            key={index}
                            className="flex items-start bg-green-50 dark:bg-green-900/20 p-4 rounded-lg transform transition-transform hover:scale-102 hover:shadow-md animated-border"
                          >
                            <span className="flex-shrink-0 inline-flex items-center justify-center h-8 w-8 bg-gradient-to-r from-green-400 to-blue-500 text-white rounded-full mr-3 shadow-md">
                              {index + 1}
                            </span>
                            <span className="text-gray-700 dark:text-gray-300">{suggestion}</span>
                          </li>
                        ))}
                    </ul>

                    <div className="mt-8 text-center">
                      <button
                        onClick={() => {
                          setResult(null)
                          window.scrollTo({ top: 0, behavior: "smooth" })
                        }}
                        className="gradient-btn text-white font-bold py-2 px-6 rounded-full transition-all duration-300 transform hover:scale-105 shadow-md"
                      >
                        Calculate Again
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </ScrollReveal>
          )}
        </div>
      </div>
    </div>
  )
}

