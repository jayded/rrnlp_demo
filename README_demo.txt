# This is an unpolished demonstration application of the RobotReviewer.

# Get a Huggingface token and copy it to ~/.cache/huggingface/token
See [Huggingface Documentation](https://huggingface.co/docs/hub/en/security-tokens)

# Get a database of pubmed data:
Download it:
```bash
aws s3 cp s3://rrnlp-artifacts/pubmed_extractions.db ./
```
Or [create the database](scripts/process_pubmed_nxmls.sh); you will need to [download the pubmed baseline](https://pubmed.ncbi.nlm.nih.gov/download/) and potentially update the elements with these scripts. 

# Embed articles
Again, download:
```bash
aws s3 cp s3://rrnlp-artifacts/annoy_dot.bin ./
aws s3 cp s3://rrnlp-artifacts/annoy_dot.json ./
```
Or [create the embeddings](scripts/demo/build_pubmed_index_embeds.sh).

# User database
```bash
aws s3 cp s3://rrnlp-artifacts/user_data.db ./
```
or create one with this schema:
```sql
CREATE TABLE user_topics(topic_uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, uid INTEGER, topic_name, search_text, search_query, final INTEGER DEFAULT 0, generated_query default NULL, used_cochrane_filter default 0, used_robot_reviewer_rct_filter default 0);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE users(uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, username TEXT, salt TEXT, hash TEXT, creationdate TEXT, passworddate TEXT);
CREATE TABLE search_screening_results(topic_uid, pmid, human_decision TEXT DEFAULT "Unscreened", robot_ranking REAL DEFAULT NULL, source TEXT DEFAULT NULL, UNIQUE(topic_uid, pmid));
CREATE INDEX search_screening_results_pmid ON search_screening_results(pmid);
CREATE INDEX search_screening_results_topic_uid_pmid ON search_screening_results(topic_uid, pmid);
CREATE INDEX search_screening_results_topic_uid_pmid_robot_ranking ON search_screening_results(topic_uid, pmid, robot_ranking);
```

# Set up the app config
```bash
aws s3 cp s3://rrnlp-artifacts/demo_pw.yml ./rrnlp/app
```
and modify any fields, e.g. username, and paths (these will need to be updated)
```yaml
cookie:
  expiry_days: 90
  key: random_signature_key
  name: random_cookie_name
credentials:
  usernames:
    demo:
      email: TODO
      failed_login_attempts: 0
      logged_in: true
      name: demo
      password: demodemo
      uid: 0
# an empty database used so we can join the user_data and the pubmed_data (they get updated at different rates)
source_db_path: empty.db
user_data_db_path: user_data.db
pubmed_data_db_path: pubmed_extractions.db
# only necessary for the "summary" page
openai_api_key: 
preauthorized:
  emails:
  - YOUREMAIL
search_bot:
  tokenizer: mistralai/Mistral-7B-Instruct-v0.2
  model_path: TODO

fine_tuned_screener_home: TODO

default_pubmed_index: contriever
pubmed_indexes:
  contriever:
    type: annoy
    base_weights: facebook/contriever
    embeddings_path: annoy_index.ann
    pmids_path: pmids.json
```

Install any packages needed from [installed.txt](installed.txt) and launch the app with 
```
PYTHONPATH=./:$PYTHONPATH streamlit run rrnlp/app/streamlit_app.py
```
