from flask import Flask, render_template, jsonify, request, send_file 
# Flask: the main object that represents your web application.
# render_template: Renders an HTML page from the templates folder
# jsonify: Returns JSON data (commonly used for APIs)
# request: Accesses incoming request data such as forms, JSON, query parameters, files, and headers
# send_file: Sends a file (PDF, image, CSV, etc.) from the server to the client
from src.exception import CustomException 
from src.logger import logging 
import os, sys 

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline 
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train", methods=['GET', 'POST'])
def train_route():
    try:
        if request.method == 'POST' or request.method == 'GET':

            train_pipeline = TrainingPipeline()
            train_file_detail = train_pipeline.run_pipeline()

            logging.info("training completed. Downloading the model file.")
            return send_file(train_file_detail,
                            download_name="model.pkl",
                            as_attachment=True)

    # return "Training Completed."
    except Exception as e:
        raise CustomException(e, sys)


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    try:
        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipline()

            logging.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                             download_name=prediction_file_detail.prediction_file_name,
                                as_attachment=True)

        else:
            return render_template('prediction.html')

    except Exception as e:
        raise CustomException(e, sys)

    
if __name__=="__main__":
    host = "127.0.0.1"
    # host = "0.0.0.0"
    port = 5010
    # print(f"🚀 App is running on: http://{host}:{port}")
    print(f"🚀 App is running on: http://localhost:{port}")
    app.run(host=host, port=port, debug=True) 

# if __name__ == "__main__":
#     host = "0.0.0.0"
#     port = 5010

#     print(f"🚀 App is running on: http://localhost:{port}")

#     app.run(
#         host=host,
#         port=port,
#         debug=False
#     )
