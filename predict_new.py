import joblib

# Model aur scaler load karo
model = joblib.load("models/house_price_model.pkl")
scaler = joblib.load("models/house_price_scaler.pkl")

# Naye ghar ka data (Example)
# Order: LotArea, YearBuilt, 1stFlrSF, 2ndFlrSF, FullBath, BedroomAbvGr, TotRmsAbvGrd, GarageCars, GarageArea
new_house = [[8000, 2005, 1200, 400, 2, 3, 7, 2, 500]]

# Scaling lagao
new_house_scaled = scaler.transform(new_house)

# Prediction
predicted_price = model.predict(new_house_scaled)
print(f"Naye ghar ka estimated price: Rs {int(predicted_price[0])}")
