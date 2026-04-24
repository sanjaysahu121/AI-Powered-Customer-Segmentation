from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pickle, os, pandas as pd, uvicorn
from fastapi.staticfiles import StaticFiles
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load Model
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))
preprocessor = pickle.load(open(PREPROCESSOR_PATH, "rb"))

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/customer")
def customer_page(request: Request):
    return templates.TemplateResponse("customer.html", {"request": request})

@app.post("/predict")
def predict(request: Request, Age: float = Form(...), Education: float = Form(...), Marital_Status: float = Form(...),
    Parental_Status: float = Form(...), Children: float = Form(...), Income: float = Form(...),
    Total_Spending: float = Form(...), Days_as_Customer: float = Form(...), Recency: float = Form(...),
    Wines: float = Form(...), Fruits: float = Form(...), Meat: float = Form(...), Fish: float = Form(...),
    Sweets: float = Form(...), Gold: float = Form(...), Web: float = Form(...), Catalog: float = Form(...),
    Store: float = Form(...), Discount_Purchases: float = Form(...), Total_Promo: float = Form(...),
    NumWebVisitsMonth: float = Form(...)):
    
    data = pd.DataFrame([{
        "Age": Age, "Education": Education, "Marital Status": Marital_Status,
        "Parental Status": Parental_Status, "Children": Children, "Income": Income,
        "Total_Spending": Total_Spending, "Days_as_Customer": Days_as_Customer,
        "Recency": Recency, "Wines": Wines, "Fruits": Fruits, "Meat": Meat,
        "Fish": Fish, "Sweets": Sweets, "Gold": Gold, "Web": Web,
        "Catalog": Catalog, "Store": Store, "Discount Purchases": Discount_Purchases,
        "Total Promo": Total_Promo, "NumWebVisitsMonth": NumWebVisitsMonth
    }])
    
    cols = ['Age', 'Education', 'Marital Status', 'Parental Status', 'Children', 'Income', 'Total_Spending', 'Days_as_Customer', 'Recency', 'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold', 'Web', 'Catalog', 'Store', 'Discount Purchases', 'Total Promo', 'NumWebVisitsMonth']
    prediction = model.predict(preprocessor.transform(data[cols]))
    segment_map = {0: "Budget Conscious", 1: "Elite / High Spenders", 2: "Loyal Customers"}
    
    return templates.TemplateResponse("customer.html", {
        "request": request, "cluster_id": int(prediction[0]), "cluster_name": segment_map.get(int(prediction[0]), "Standard")
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)