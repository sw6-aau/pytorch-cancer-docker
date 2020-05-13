from flask import Flask, request
from subprocess import Popen, PIPE
from google.cloud import storage

import os

app = Flask(__name__)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"x
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

@app.route('/train', methods=['GET'])
def train():
    build_id = request.args.get("build_id")
    build_data = build_id + '.txt'
    build_model = build_id + '.pt'
    download_blob("astep-storage", build_id, build_data)
    
    horizon = request.args.get("horizon")
    dropout = request.args.get("dropout")
    epoch = request.args.get("epoch")
    hid_cnn = request.args.get("hid_cnn")
    hid_rnn = request.args.get("hid_rnn")
    window_rnn = request.args.get("window_rnn")
    window_hw = request.args.get("windows_hw")
    
    os.system('python main.py ' + '--data ' + build_data + ' --save ' + build_model + ' --horizon ' + horizon +
              ' --dropout ' + dropout  + ' --epochs ' + epoch + ' --hidRNN ' + hid_rnn + ' --window ' + window_rnn +
              ' --highway_window ' + window_hw + ' --L1Loss False --output_fun None ' + ' --hidCNN ' + hid_cnn)

    upload_blob("astep-storage", build_model, build_model)
    return build_id


@app.route('/predict', methods=['GET'])
def predict():
    build_id = request.args.get("build_id")
    datafile_id = request.args.get("datafile_id")
    build_model = build_id + '.pt'
    build_predict = build_id + '.predict'
    download_blob("astep-storage", build_model, build_model)
    download_blob("astep-storage", datafile_id, datafile_id)

    os.system('python predict.py' + ' --data ' + datafile_id + ' --save ' + build_model + ' --hidCNN 50 --hidRNN 50 --L1Loss False --output_fun None --epochs 1000')

    upload_blob("astep-storage", "output.csv", build_predict)
    return build_id

@app.route('/test', methods=['GET'])
def test():
    return "works"

@app.route('/default', methods=['GET'])
def default():
    build_id = "test123"
    os.system('python main.py --data exchange_rate.txt --save exchange_rate.pt --hidCNN 50 --hidRNN 50 --epochs 10 --L1Loss False --output_fun None')
    upload_blob("astep-storage", "exchange_rate.pt", build_id)
    return "done"

port = int(os.environ.get('PORT', 8080))
if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=port)