import type { PredictionResult } from "@/app/predictor/page"
import { AlertTriangle, Check } from "lucide-react"

type ResultsDisplayProps = {
  result: PredictionResult
}

export default function ResultsDisplay({ result }: ResultsDisplayProps) {
  // Determine emission level category
  const getEmissionCategory = (emissions: number) => {
    if (emissions < 120) return { label: "Low", color: "green" }
    if (emissions < 160) return { label: "Moderate", color: "yellow" }
    if (emissions < 200) return { label: "High", color: "orange" }
    return { label: "Very High", color: "red" }
  }

  const emissions = result.CO2_Emissions_Prediction
  const category = getEmissionCategory(emissions)

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 animate-fadeIn">
      <h2 className="text-2xl font-semibold text-green-700 mb-6 text-center">CO2 Emissions Result</h2>

      <div className="flex flex-col items-center mb-8">
        <div
          className={`w-40 h-40 rounded-full flex items-center justify-center mb-4 ${
            category.color === "green"
              ? "bg-green-100 text-green-700"
              : category.color === "yellow"
                ? "bg-yellow-100 text-yellow-700"
                : category.color === "orange"
                  ? "bg-orange-100 text-orange-700"
                  : "bg-red-100 text-red-700"
          }`}
        >
          <div className="text-center">
            <div className="text-3xl font-bold">{Math.round(emissions)}</div>
            <div className="text-sm">g/km</div>
          </div>
        </div>

        <div
          className={`px-4 py-1 rounded-full text-sm font-medium ${
            category.color === "green"
              ? "bg-green-100 text-green-700"
              : category.color === "yellow"
                ? "bg-yellow-100 text-yellow-700"
                : category.color === "orange"
                  ? "bg-orange-100 text-orange-700"
                  : "bg-red-100 text-red-700"
          }`}
        >
          {category.label} Emissions
        </div>
      </div>

      <div className="bg-green-50 rounded-lg p-4 mb-6">
        <h3 className="font-semibold text-green-800 mb-2 flex items-center">
          <AlertTriangle className="h-5 w-5 mr-2" />
          Environmental Impact
        </h3>
        <p className="text-gray-700">
          A vehicle emitting {Math.round(emissions)} g/km of CO2 will produce approximately{" "}
          {((emissions * 15000) / 1000000).toFixed(2)} tonnes of CO2 per year (based on 15,000 km driven).
        </p>
      </div>

      <div>
        <h3 className="font-semibold text-green-800 mb-3 flex items-center">
          <Check className="h-5 w-5 mr-2" />
          Suggestions to Reduce Emissions
        </h3>

        <ul className="space-y-2">
          {result.Reduction_Tips.map((suggestion, index) => (
            <li key={index} className="flex items-start">
              <span className="inline-block w-5 h-5 bg-green-100 text-green-700 rounded-full flex-shrink-0 flex items-center justify-center mr-2 mt-0.5">
                {index + 1}
              </span>
              <span className="text-gray-700">{suggestion}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

