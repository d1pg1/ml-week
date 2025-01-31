from flask import Flask, request, render_template
import pickle
import joblib
import spacy
import shap
import pandas as pd
import numpy as np
from scipy import stats
import re
import matplotlib.pyplot as plt
import io
import base64
import warnings
warnings.simplefilter(action='ignore', category=Warning)

app = Flask(__name__)

def extract_year(text):
    match = re.search(r'\b(19|20)\d{2}\b', text)
    return match.group(0) if match else "Unknown"

def prepare_data(data):
    cols = ["country", "description", "designation", "price", "province", "region_1", "taster_name", "variety", "winery", "year"]
    data["year"] = extract_year(data["title"])
    for key, value in data.items():
        if value == '':
            data[key] = "Unknown"
    features_array = pd.DataFrame([data])
    features_array = features_array[cols]
    print(features_array['price'].values.astype(float))
    features_array["box_cox_price"] = stats.boxcox(features_array['price'].values.astype(float), -0.3103003305981917)
    features_array['description'] = features_array['description'].str.replace('[^\w\s]', '')
    features_array['description'] = features_array['description'].str.replace('[0-9]+', '')
    features_array['description'] = features_array['description'].apply(lambda x: " ".join([token.lemma_ for token in nlp(x) if not token.is_stop]))

    return features_array

def plot_shap(input_data):
    """ Generate SHAP plot and return as base64 image. """
    shap_values = shap_explainer(input_data)
    #shap_values.base_values = 1 / (1 + np.exp(-shap_values.base_values))
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0], show=False)
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Load the trained regression model
with open('wine_points_XGBoostRegressor_v2.joblib', 'rb') as f:
    model = joblib.load(f)

dataset = pd.read_csv("winemag-data-130k-v2.csv")

preprocessor = model.named_steps['preprocessor']
feature_names = []
for name, transformer, cols in preprocessor.transformers_:
    if hasattr(transformer, 'get_feature_names_out'):  # Якщо є get_feature_names_out
      feature_names.extend(['word:' + name for name in transformer.get_feature_names_out()])

    else:
          if 8 in cols:
              cols = ["price"]
          print(cols)
          feature_names.extend(cols)  # Для категоріальних/числових ознак

# Initialize SHAP Explainer
shap_explainer = shap.Explainer(model["regressor"], feature_names=feature_names)

nlp = spacy.load("en_core_web_sm")

@app.route('/')
def home():
    return render_template('index.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form fields
        data = request.form.to_dict()
        
        features = prepare_data(data.copy())
        print(features)
        # Make prediction
        prediction = model.predict(features)

        transformed_features = model["preprocessor"].transform(features)
        print(type(transformed_features))
        print(len(feature_names))
        #transformed_features = pd.DataFrame(transformed_features.toarray(), columns=feature_names)
        shap_img = plot_shap(transformed_features)

        return render_template(
            'index.html', 
            prediction=prediction[0], 
            error=None, 
            input_data=data,
            shap_img=shap_img, 
        )
    except Exception as e:
        print(str(e))
        return render_template(
            'index.html', 
            prediction=None, 
            error=str(e), 
            input_data=request.form,
            shap_img=None, 
        )
    
@app.route('/get_random_data')
def get_random_data():
    # Select a random entry from the dataset
    random_entry = dataset.sample()
    random_entry = random_entry.to_dict(orient='records')[0]
    print(random_entry)

    return render_template(
            'index.html', 
            prediction=None, 
            error=None, 
            random_data=random_entry,
            shap_img=None, 
        )


if __name__ == '__main__':
    app.run(debug=True)
