import requests

def post_request():

    api_url = "https://api.mage.ai/v1/predict"
    payload = {
    "api_key": "656xaAU87E94JsteVV3zO0gTFpL5r4iqA22wNTMn",
    "features": [
        {
        "id": "Gomesa longipes"
        }
    ],
    "model": "custom_prediction_regression_1645324643229",
    "version": "1"
    }

    response = requests.post(api_url, json=payload)
    return response.json()[0]['prediction']