import pickle

# Load the model and DictVectorizer from the binary file
with open('pipeline.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# The lead data for Question 3
lead = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41
}

# Transform the dictionary and calculate the probability
X = dv.transform([lead])
prob = model.predict_proba(X)[0, 1]

print(f"Conversion probability: {round(prob, 3)}")
