from fastapi import FastAPI, HTTPException
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

# Define the request model
class Features(BaseModel):
    Make: str
    Vehicle_Class: str
    Engine_Size: float
    Cylinders: int
    Transmission: str
    Fuel_Type: str
    Fuel_Consumption_City: float
    Fuel_Consumption_Hwy: float
    Fuel_Consumption_Comb: float

def get_reduction_tips(data: Features):
    tips = []

    # Engine Size & Cylinders
    if data.Engine_Size > 2.5:
        tips.append("Consider a smaller engine size or a hybrid vehicle to reduce CO₂ emissions.")
    if data.Cylinders > 4:
        tips.append("Opt for a car with fewer cylinders, as higher-cylinder engines consume more fuel.")

    # Transmission
    if data.Transmission.startswith("M"):
        tips.append("Manual transmissions can sometimes be more fuel-efficient, but modern automatic cars may offer better efficiency.")
    
    # Fuel Type
    if data.Fuel_Type in ["Z", "D"]:  # Diesel & certain gasoline cars
        tips.append("Consider switching to an electric or hybrid vehicle to drastically lower emissions.")

    # Fuel Efficiency
    if data.Fuel_Consumption_Comb > 8.0:
        tips.append("Improve fuel efficiency by maintaining steady speeds and reducing unnecessary acceleration.")

    if not tips:
        tips.append("Your car already has relatively low CO₂ emissions. Keep maintaining it for optimal performance!")

    return tips

@app.post("/co2_predict")
def predict(data: Features):
    try:
        # Convert input to DataFrame with correct column names
        input_df = pd.DataFrame([{
            "Make": data.Make,
            "Vehicle Class": data.Vehicle_Class,
            "Engine Size(L)": data.Engine_Size,
            "Cylinders": data.Cylinders,
            "Transmission": data.Transmission,
            "Fuel Type": data.Fuel_Type,
            "Fuel Consumption City (L/100 km)": data.Fuel_Consumption_City,
            "Fuel Consumption Hwy (L/100 km)": data.Fuel_Consumption_Hwy,
            "Fuel Consumption Comb (L/100 km)": data.Fuel_Consumption_Comb
        }])

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
