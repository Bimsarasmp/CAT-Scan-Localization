from flask import Flask, make_response, request, jsonify
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ImagePrediction import markPointInImage


app = Flask(__name__)


#load the model
file = os.getcwd()+"\API\Models\\"
linearModel = pickle.load(open(file+"linearModel.pkl", "rb"))
ridgeModel = pickle.load(open(file+"ridgeModel.pkl", "rb"))
elasticNetModel = pickle.load(open(file+"elasticNetModel.pkl", "rb"))
decisionModel = pickle.load(open(file+"decisionTreeModel.pkl", "rb"))
ensembleModel = pickle.load(open(file+"ensembleModel.pkl", "rb"))

#method to convert feature to pca vectors
def convertToPcaVectors(features):
    scaler = StandardScaler()
    scaledFeatures = scaler.fit_transform([features])
    pca = PCA(0.75)
    pcaVectors = pca.fit_transform(scaledFeatures)
    return pcaVectors


#method to predict linear regression values
def predictNonPcaModels(features):
    linear = linearModel.predict([features])
    ridge = ridgeModel.predict([features])
    elastic = elasticNetModel.predict([features])
    decision = decisionModel.predict([features])
    ensemble = ensembleModel.predict([[linear[0],ridge[0],elastic[0],decision[0]]])
    return ensemble[0]

#method to predict other models that use pca vectors
def predictingPcaModels(pca):
    pass


@app.route('/sliceLocalizationPrediction', methods = ['POST'])
def getFeatures():
    if request.method == "POST":
        data = request.get_json()
        features = data['features']
        if len(features) == 384:
            prediction = predictNonPcaModels(features)
            predictionImage = markPointInImage(round(prediction,2))
            response = jsonify({'image':predictionImage.tolist()})
            return response
        else:
            return f"Input doesn't meet prerequisite, Input length is {len(features)}"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 105)