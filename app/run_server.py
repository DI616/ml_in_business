# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
# 	python simple_request.py


# import the necessary packages
import dill
import pandas as pd
import os
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime


dill._dill._reverse_typemap['ClassType'] = type


# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
    # load the pre-trained model
    global model
    with open(model_path, 'rb') as f:
        model = dill.load(f)
    print(model)


modelpath = "/home/valentine/Documents/GU_AI_396/07_ML_in_business/course_project/app/model/random_forest_pipeline.dill"
load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
    return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    dt = strftime("[%Y-%b-%d %H:%M:%S]")
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        feats = {'MinTemp': [], 'MaxTemp': [], 'Rainfall': [], 'WindGustDir': [], 'WindGustSpeed': [], 'WindDir9am': [],
                 'WindDir3pm': [], 'WindSpeed9am': [], 'WindSpeed3pm': [], 'Humidity9am': [], 'Humidity3pm': [],
                 'Pressure9am': [], 'Pressure3pm': [], 'Temp9am': [], 'Temp3pm': [], 'RainToday': []}

        request_json = flask.request.get_json()

        for feat in feats:
            if request_json[feat]:
                feats[feat].append(request_json[feat])
            else:
                feats[feat] = [0]

        # description, company_profile, benefits = "", "", ""

        # if request_json["description"]:
        #     description = request_json['description']
        #
        # if request_json["company_profile"]:
        #     company_profile = request_json['company_profile']
        #
        # if request_json["benefits"]:
        #     benefits = request_json['benefits']

        # logger.info(f'{dt} Data: description={description}, company_profile={company_profile}, benefits={benefits}')
        try:
            print(feats)
            # print(pd.DataFrame(feats))
            preds = model.predict_proba(pd.DataFrame(feats))
        except AttributeError as e:
            logger.warning(f'{dt} Exception: {str(e)}')
            data['predictions'] = str(e)
            data['success'] = False
            return flask.jsonify(data)

        data["predictions"] = preds[:, 1][0]
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading the model and Flask starting server..."
           "please wait until server has fully started"))
    port = int(os.environ.get('PORT', 8180))
    app.run(host='0.0.0.0', debug=True, port=port)
