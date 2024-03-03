from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from bson import ObjectId
from flask_cors import CORS
from io import BytesIO
import base64
from bson import Binary
import requests
import json
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import fitz

app = Flask(__name__)
CORS(app)

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

mongo_uri = ''

db_name = 'test-db'
collection_name= 'test-connection'
pdf_collection_name = 'fs.chunks'

client = MongoClient(mongo_uri)
db = client[db_name]
collection = db[collection_name]
pdf_collection = db[pdf_collection_name]

WORKDAY_JOBS = {
  "WALMART": [
    "https://walmart.wd5.myworkdayjobs.com/wday/cxs/walmart/WalmartExternal/jobs",
    "https://walmart.wd5.myworkdayjobs.com/wday/cxs/walmart/WalmartExternal/job",
  ],
  "NVIDIA": [
    "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/jobs",
    "https://nvidia.wd5.myworkdayjobs.com/wday/cxs/nvidia/NVIDIAExternalCareerSite/job",
  ],
  "Salesforce": [
     "https://salesforce.wd12.myworkdayjobs.com/wday/cxs/salesforce/External_Career_Site/jobs",
     "https://salesforce.wd12.myworkdayjobs.com/wday/cxs/salesforce/External_Career_Site/job",
  ],
  "Henryschein": [
     "https://henryschein.wd1.myworkdayjobs.com/wday/cxs/henryschein/External_Careers/jobs",
     "https://henryschein.wd1.myworkdayjobs.com/wday/cxs/henryschein/External_Careers/job",
  ],
  "Airbus": [
     "https://ag.wd3.myworkdayjobs.com/wday/cxs/ag/Airbus/jobs",
     "https://ag.wd3.myworkdayjobs.com/wday/cxs/ag/Airbus/job",
  ],
  "Exact Science": [
     "https://exactsciences.wd1.myworkdayjobs.com/wday/cxs/exactsciences/Exact_Sciences/jobs",
     "https://exactsciences.wd1.myworkdayjobs.com/wday/cxs/exactsciences/Exact_Sciences/job",
  ],
  "Zoetis": [
     "https://zoetis.wd5.myworkdayjobs.com/wday/cxs/zoetis/zoetis/jobs",
     "https://zoetis.wd5.myworkdayjobs.com/wday/cxs/zoetis/zoetis/job",
  ],
  "Aveva": [
     "https://aveva.wd3.myworkdayjobs.com/wday/cxs/aveva/AVEVA_careers/jobs",
     "https://aveva.wd3.myworkdayjobs.com/wday/cxs/aveva/AVEVA_careers/job",
  ],
  "Orion": [
     "https://orionadvisor.wd1.myworkdayjobs.com/wday/cxs/orionadvisor/Orion_Careers/jobs",
     "https://orionadvisor.wd1.myworkdayjobs.com/wday/cxs/orionadvisor/Orion_Careers/job",
  ],
  "Adobe": [
     "https://adobe.wd5.myworkdayjobs.com/wday/cxs/adobe/external_experienced/jobs",
     "https://adobe.wd5.myworkdayjobs.com/wday/cxs/adobe/external_experienced/job",
  ],
  "Enbridge": [
     "https://enbridge.wd3.myworkdayjobs.com/wday/cxs/enbridge/enbridge_careers/jobs",
     "https://enbridge.wd3.myworkdayjobs.com/wday/cxs/enbridge/enbridge_careers/job",
  ],
  "Zillow": [
     "https://zillow.wd5.myworkdayjobs.com/wday/cxs/zillow/Zillow_Group_External/jobs",
     "https://zillow.wd5.myworkdayjobs.com/wday/cxs/zillow/Zillow_Group_External/job",
  ],
}

BODY = {
  "appliedFacets": {}, 
  "limit": 20, 
  "offset": 0, 
  "searchText": "",
}

HEADERS = {
  "Content-Type": "application/json",
  "Accept": "*/*",
  "Accept-Language": "en-US",
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

def calculate_match_score(pdf_text, job_description):
    set1 = word_tokenize(pdf_text.lower())
    set2 = word_tokenize(job_description.lower())

    set1_str = ' '.join(set1)
    set2_str = ' '.join(set2)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([set1_str, set2_str])

    cosine_similarity_score = cosine_similarity(vectors[0], vectors[1])[0, 0]
    return cosine_similarity_score

def scale_scores(jobs, sc, new_min, new_max):
    old_min = min(sc)
    old_max = max(sc)
    for j in jobs:
        new_score = round(((j["match_score"] - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min,3)
        j["match_score"] = new_score
    return jobs

def remove_html_tags(input_string):
    soup = BeautifulSoup(input_string, 'html.parser')
    text_without_tags = soup.get_text()
    return text_without_tags

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

@app.route('/getjobs', methods=['POST'])
def get_jobs():
  links = []
  BODY["searchText"] = "Software Engineer"
  for company in WORKDAY_JOBS:
    response = requests.post(WORKDAY_JOBS[company][0], data=json.dumps(BODY), headers=HEADERS)
    data = response.json()
    for job in data["jobPostings"]:
      job_data = {}
      job_data["title"] = job["title"]
      job_data["location"] = job["locationsText"]
      job_data["url"] = WORKDAY_JOBS[company][1] + "/" + job["externalPath"].split("/")[-1]
      job_data["company"] = company
      job_data["postedOn"] = job["postedOn"]
      resp = requests.get(job_data["url"])
      resp_data = resp.json()
      job_data["description"] = remove_html_tags(resp_data["jobPostingInfo"]["jobDescription"])
      job_data["job_link"] = resp_data["jobPostingInfo"]["externalUrl"]
      links.append(job_data)
  
  resume_text = fitz.open("resume.pdf")
  pdf_text = ""
  for page_num in range(len(resume_text)):
      page = resume_text.load_page(page_num)
      pdf_text += page.get_text()
  resume_text.close()

  scores = []
  for link in links:
    match_score = calculate_match_score(pdf_text, link["description"])
    match_percentage = match_score * 100
    link["match_score"] = match_percentage
    scores.append(match_percentage)

  updated_links = scale_scores(links, scores, 30, 85)

  updated_links = [i for i in updated_links if i["match_score"] > 50]
  updated_links.sort(key=lambda x: x["match_score"], reverse=True)
  return updated_links

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
