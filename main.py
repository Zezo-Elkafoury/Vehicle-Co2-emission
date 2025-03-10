from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd

# Load trained model and preprocessor
with open("co2_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, you can specify your frontend's origin instead
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the request model with optional fields
class Features(BaseModel):
    Make: Optional[str] = None
    Vehicle_Class: Optional[str] = None
    Engine_Size: Optional[float] = None
    Cylinders: Optional[int] = None
    Transmission: Optional[str] = None
    Fuel_Type: Optional[str] = None
    Fuel_Consumption_City: Optional[float] = None
    Fuel_Consumption_Hwy: Optional[float] = None
    Fuel_Consumption_Comb: Optional[float] = None

def get_reduction_tips(data: Features):
    tips = []

    # Engine Size & Cylinders
    if data.Engine_Size is not None and data.Engine_Size > 2.5:
        tips.append("Consider a smaller engine size or a hybrid vehicle to reduce CO₂ emissions.")
    if data.Cylinders is not None and data.Cylinders > 4:
        tips.append("Opt for a car with fewer cylinders, as higher-cylinder engines consume more fuel.")

    # Transmission
    if data.Transmission and data.Transmission.startswith("M"):
        tips.append("Manual transmissions can sometimes be more fuel-efficient, but modern automatic cars may offer better efficiency.")
    
    # Fuel Type
    if data.Fuel_Type in ["Z", "D"]:  # Diesel & certain gasoline cars
        tips.append("Consider switching to an electric or hybrid vehicle to drastically lower emissions.")

    # Fuel Efficiency
    if data.Fuel_Consumption_Comb is not None and data.Fuel_Consumption_Comb > 8.0:
        tips.append("Improve fuel efficiency by maintaining steady speeds and reducing unnecessary acceleration.")

    if not tips:
        tips.append("Your car already has relatively low CO₂ emissions. Keep maintaining it for optimal performance!")

    return tips

@app.post("/co2_predict")
def predict(data: Features):
    try:
        # Convert input to DataFrame with correct column names
        input_data = {
            "Make": data.Make or "Unknown",
            "Vehicle Class": data.Vehicle_Class or "Unknown",
            "Engine Size(L)": data.Engine_Size if data.Engine_Size is not None else 0.0,
            "Cylinders": data.Cylinders if data.Cylinders is not None else 0,
            "Transmission": data.Transmission or "Unknown",
            "Fuel Type": data.Fuel_Type or "Unknown",
            "Fuel Consumption City (L/100 km)": data.Fuel_Consumption_City if data.Fuel_Consumption_City is not None else 0.0,
            "Fuel Consumption Hwy (L/100 km)": data.Fuel_Consumption_Hwy if data.Fuel_Consumption_Hwy is not None else 0.0,
            "Fuel Consumption Comb (L/100 km)": data.Fuel_Consumption_Comb if data.Fuel_Consumption_Comb is not None else 0.0,
        }
        
        input_df = pd.DataFrame([input_data])

        # Apply preprocessing
        transformed_input = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(transformed_input)

        # Get reduction tips
        reduction_tips = get_reduction_tips(data)

        return {
            "CO2_Emissions_Prediction": float(prediction[0]),
            "Reduction_Tips": reduction_tips
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
