"use client"

import { useState } from "react"
import CO2Form from "@/components/co2/CO2Form"
import ResultsDisplay from "@/components/co2/ResultsDisplay"

// Update the FormData type to match the API's expected format
export type FormData = {
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

// Update the PredictionResult type to match the API's response format
export type PredictionResult = {
  CO2_Emissions_Prediction: number
  Reduction_Tips: string[]
}

export default function PredictorPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)

  const handleSubmit = async (data: FormData) => {
    // Add a delay of 1 second
    await new Promise((resolve) => setTimeout(resolve, 1000));
    console.log("Data being sent to API:", data);

    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch("https://underlying-tiffany-ziad-2007-547ae087.koyeb.app/co2_predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        throw new Error("Failed to get prediction. Please try again.")
      }

      const result = await response.json()
      console.log("Response from API:", result);
      setResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold text-green-800 mb-8 text-center">CO2 Emissions Predictor</h1>

      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-semibold text-green-700 mb-4">Vehicle Information</h2>
          <p className="mb-6 text-gray-600">
            Enter your vehicle details below to get an estimate of CO2 emissions and personalized suggestions for
            reducing your carbon footprint. All fields are optional, but providing more information will result in a
            more accurate prediction.
          </p>

          <CO2Form onSubmit={handleSubmit} isLoading={isLoading} />
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded relative mb-6" role="alert">
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        {result && <ResultsDisplay result={result} />}
      </div>
    </div>
  )
}

