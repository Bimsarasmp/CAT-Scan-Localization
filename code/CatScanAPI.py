from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


#load the model
file = r"D:/Sem-2/3014 - Introduction to Artificial Intelligence/Project/"
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
    return f"Linear Model : {linear[0]}\nRidge Model : {ridge[0]}\nElastice Net Model : {elastic[0]}\nDecision Tree Model : {decision[0]}\nEnsemble Model : {ensemble[0]}"

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
            pcaVectors = convertToPcaVectors(features)
            #otherPredictions = predictingPcaModels(pcaVectors)
            #finalOuptut = prediction+otherPredictions
            #return finalOuptut
            return prediction
        else:
            return f"Input is {len(features)}"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 105)