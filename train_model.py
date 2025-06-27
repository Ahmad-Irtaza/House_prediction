# Step 1: Libraries import karo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Step 2: Dataset load karo
df = pd.read_csv("data/train.csv")

# Step 3: Important columns select karo
# Ye columns regression mein useful hotay hain
columns = [
    'LotArea',
    'YearBuilt',
    '1stFlrSF',
    '2ndFlrSF',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd',
    'GarageCars',
    'GarageArea',
    'SalePrice'
]

df = df[columns]

# Step 4: Check missing values
print("Missing values:\n", df.isnull().sum())

# Step 5: Jo missing hain unko fill karo (0 se ya median se)
df = df.fillna(0)

# Step 6: Features aur Target define karo
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Step 7: Features ko scale karo (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Split karo train aur test mein
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Step 9: Model banao aur train karo
model = LinearRegression()
model.fit(X_train, y_train)

# Step 10: Prediction lo
y_pred = model.predict(X_test)

# Step 11: Performance dekho
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Step 12: Model aur Scaler save karo
joblib.dump(model, "models/house_price_model.pkl")
joblib.dump(scaler, "models/house_price_scaler.pkl")
print("\nModel aur scaler save ho gaye models/ folder mein.")
