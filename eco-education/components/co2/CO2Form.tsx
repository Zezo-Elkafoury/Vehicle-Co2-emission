import type React from "react"
import { useState } from "react"
import type { FormData } from "@/app/predictor/page"

type CO2FormProps = {
  onSubmit: (data: FormData) => void
  isLoading: boolean
}

export default function CO2Form({ onSubmit, isLoading }: CO2FormProps) {
  const [formData, setFormData] = useState<FormData>({
    Make: "",
    Vehicle_Class: "",
    Engine_Size: "",
    Cylinders: "",
    Transmission: "",
    Fuel_Type: "",
    Fuel_Consumption_City: "",
    Fuel_Consumption_Hwy: "",
    Fuel_Consumption_Comb: "",
  })

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit(formData)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label htmlFor="Make" className="block text-sm font-medium text-gray-700 mb-1">
            Make
          </label>
          <input
            type="text"
            id="Make"
            name="Make"
            value={formData.Make}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            placeholder="e.g., Toyota, Honda"
          />
        </div>

        <div>
          <label htmlFor="Vehicle_Class" className="block text-sm font-medium text-gray-700 mb-1">
            Vehicle Class
          </label>
          <select
            id="Vehicle_Class"
            name="Vehicle_Class"
            value={formData.Vehicle_Class}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            <option value="">Select Vehicle Class</option>
            <option value="SUV">SUV</option>
            <option value="COMPACT">Compact</option>
            <option value="MID-SIZE">Mid-Size</option>
            <option value="FULL-SIZE">Full-Size</option>
            <option value="PICKUP TRUCK">Pickup Truck</option>
            <option value="MINIVAN">Minivan</option>
          </select>
        </div>

        <div>
          <label htmlFor="Engine_Size" className="block text-sm font-medium text-gray-700 mb-1">
            Engine Size (L)
          </label>
          <input
            type="number"
            id="Engine_Size"
            name="Engine_Size"
            value={formData.Engine_Size}
            onChange={handleChange}
            step="0.1"
            min="0.5"
            max="10"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            placeholder="e.g., 2.0"
          />
        </div>

        <div>
          <label htmlFor="Cylinders" className="block text-sm font-medium text-gray-700 mb-1">
            Cylinders
          </label>
          <input
            type="number"
            id="Cylinders"
            name="Cylinders"
            value={formData.Cylinders}
            onChange={handleChange}
            min="3"
            max="12"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            placeholder="e.g., 4"
          />
        </div>

        <div>
          <label htmlFor="Transmission" className="block text-sm font-medium text-gray-700 mb-1">
            Transmission
          </label>
          <select
            id="Transmission"
            name="Transmission"
            value={formData.Transmission}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            <option value="">Select Transmission</option>
            <option value="A">Automatic</option>
            <option value="M">Manual</option>
            <option value="AM">Automated Manual</option>
            <option value="CVT">Continuously Variable</option>
          </select>
        </div>

        <div>
          <label htmlFor="Fuel_Type" className="block text-sm font-medium text-gray-700 mb-1">
            Fuel Type
          </label>
          <select
            id="Fuel_Type"
            name="Fuel_Type"
            value={formData.Fuel_Type}
            onChange={handleChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            <option value="">Select Fuel Type</option>
            <option value="X">Regular Gasoline</option>
            <option value="Z">Premium Gasoline</option>
            <option value="D">Diesel</option>
            <option value="E">Ethanol</option>
          </select>
        </div>

        <div>
          <label htmlFor="Fuel_Consumption_City" className="block text-sm font-medium text-gray-700 mb-1">
            Fuel Consumption City (L/100km)
          </label>
          <input
            type="number"
            id="Fuel_Consumption_City"
            name="Fuel_Consumption_City"
            value={formData.Fuel_Consumption_City}
            onChange={handleChange}
            step="0.1"
            min="1"
            max="30"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            placeholder="e.g., 9.5"
          />
        </div>

        <div>
          <label htmlFor="Fuel_Consumption_Hwy" className="block text-sm font-medium text-gray-700 mb-1">
            Fuel Consumption Highway (L/100km)
          </label>
          <input
            type="number"
            id="Fuel_Consumption_Hwy"
            name="Fuel_Consumption_Hwy"
            value={formData.Fuel_Consumption_Hwy}
            onChange={handleChange}
            step="0.1"
            min="1"
            max="30"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            placeholder="e.g., 7.2"
          />
        </div>

        <div>
          <label htmlFor="Fuel_Consumption_Comb" className="block text-sm font-medium text-gray-700 mb-1">
            Fuel Consumption Combined (L/100km)
          </label>
          <input
            type="number"
            id="Fuel_Consumption_Comb"
            name="Fuel_Consumption_Comb"
            value={formData.Fuel_Consumption_Comb}
            onChange={handleChange}
            step="0.1"
            min="1"
            max="30"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
            placeholder="e.g., 8.3"
          />
        </div>
      </div>

      <div className="flex justify-center mt-8">
        <button
          type="submit"
          disabled={isLoading}
          className={`bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-8 rounded-full transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-50 ${
            isLoading ? "opacity-70 cursor-not-allowed" : ""
          }`}
        >
          {isLoading ? "Calculating..." : "Calculate CO2 Emissions"}
        </button>
      </div>
    </form>
  )
}

