from flask import Flask, make_response, request, jsonify
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ImagePrediction import markPointInImage

app = Flask(__name__)

#load the model
file = os.getcwd()+"\API\Models\\"
linearModel = pickle.load(open(file+"linear.pkl", "rb"))
ridgeModel = pickle.load(open(file+"ridge.pkl", "rb"))
elasticNetModel = pickle.load(open(file+"elastic.pkl", "rb"))
decisionModel = pickle.load(open(file+"decision.pkl", "rb"))
ensembleModel = pickle.load(open(file+"ensemble.pkl", "rb"))
scaler = pickle.load(open(file+"scaler.pkl", "rb"))
pca = pickle.load(open(file+"pca.pkl", "rb"))

#method to convert feature to pca vectors
def convertToPcaVectors(features):
    scaledFeatures = scaler.transform([features])
    pcaVectors = pca.transform(scaledFeatures)
    return pcaVectors

#method to predict other models that use pca vectors
def predictingPcaModels(features):
    pca = convertToPcaVectors(features)
    linear = linearModel.predict(pca)
    ridge = ridgeModel.predict(pca)
    elastic = elasticNetModel.predict(pca)
    decision = decisionModel.predict(pca)
    ensemble = ensembleModel.predict([[linear[0], ridge[0], elastic[0], decision[0]]])
    return ensemble[0]

@app.route('/sliceLocalizationPrediction', methods = ['POST'])
def getFeatures():
    if request.method == "POST":
        data = request.get_json()
        features = data['features']
        if len(features) == 384:
            prediction = predictingPcaModels(features)
            predictionImage = markPointInImage(round(prediction,2))
            response = jsonify({'image':predictionImage.tolist()})
            return response
        else:
            return f"Input doesn't meet prerequisite, Input length is {len(features)}"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 105)