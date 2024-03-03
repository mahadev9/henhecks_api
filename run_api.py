from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from bson import ObjectId
from flask_cors import CORS
from io import BytesIO
import base64
from bson import Binary

app = Flask(__name__)
CORS(app)


mongo_uri = 'mongodb+srv://sm4825:VDSIdNRsYU9v9EVr@cluster0.gvo92qb.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'


db_name = 'test-db'
collection_name= 'test-connection'
pdf_collection_name = 'fs.chunks'


client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]
pdf_collection = db[pdf_collection_name]

@app.route('/api/data', methods=['GET'])
def get_all_data():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight request handled'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

    data = list(collection.find({}))

    for item in data:
        item['_id'] = str(item['_id'])

    print(data)

    return jsonify(data)

@app.route('/api/data/<id>', methods=['GET'])
def get_data_by_id(id):
    try:
        object_id = ObjectId(id)
    except Exception as e:
        return jsonify({'error': 'Invalid ID'}), 400

    data = collection.find_one({'_id': object_id})

    if data:
        data['_id'] = str(data['_id'])
        return jsonify(data)
    else:
        return jsonify({'error': 'Data not found'}), 404

@app.route('/store_data', methods=['POST', 'OPTIONS'])
def store_data():

    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight request handled'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
  
    
    data = request.get_json()
   

    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    
    result= collection.insert_one(data)

    inserted_id = str(result.inserted_id)

   
    response_data = {'message': 'Data inserted successfully', 'inserted_id': inserted_id}
    
    return jsonify(response_data), 200



@app.route('/pdf/<pdf_id>', methods=['GET'])
def view_pdf(pdf_id):
 
    pdf_document = collection.find_one({"_id": ObjectId(pdf_id)})
    

    if pdf_document:
        pdf_data = pdf_document["pdf_data"]
       
        return send_file(BytesIO(pdf_data), mimetype='application/pdf')
    else:
        return "PDF not found", 404
    

    
@app.route('/upload/pdf', methods=['POST'])
def upload_pdf():
    pdf_base64 = request.json.get("pdf_data")
    
    pdf_binary = base64.b64decode(pdf_base64)

    file_id = collection.insert_one({"pdf_data": Binary(pdf_binary)}).inserted_id

    return jsonify({"file_id": str(file_id)})


@app.route('/login', methods=['POST'])
def login():
    
    login_data = request.json

    email = login_data.get("email")
    password = login_data.get("password")

    user = collection.find_one({"email": email, "password": password}, {"_id": 1})

    if user:
        user_id = str(user["_id"])
        return jsonify({"user_id": user_id}), 200
    else:
        return jsonify({"error": "Invalid email or password"}), 401

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)