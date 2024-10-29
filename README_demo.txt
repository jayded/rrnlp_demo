# augment the result of Somin's project with some additional tables:
sqlite> CREATE TABLE search_screening_results(topic_uid, pmid, human_decision TEXT DEFAULT "Unscreened", robot_ranking REAL DEFAULT NULL, source TEXT DEFAULT NULL, UNIQUE(topic_uid, pmid));
sqlite> CREATE TABLE user_topics(topic_uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, uid INTEGER, topic_name, search_text, search_query);

# Create a database of pubmed meta-data
This will have to be repeated with each update! Annoying but a reasonably compromise at the moment.
## Copy pubmed: archive-pubmed -path /media/data/jay/pubmed_archive
# see https://www.nlm.nih.gov/dataguide/edirect/archive.html
Then read the contents into csvs:
```python
import glob
import pandas as pd
import pubmed_parser as pp
from tqdm import tqdm

for f in tqdm(list(glob.glob('/media/data/jay/pubmed_archive/Pubmed/*xml.gz'))):
    # TODO check if the csv file exists and is newer than the one being parsed; skip it if so.
    dicts_out = pp.parse_medline_xml(f)
    df = pd.DataFrame.from_records(dicts_out)
    df.to_csv(f + '.csv')
    print(f + '.csv')
```
# once the above is complete, confirm that only one header exists
```bash
head -n 1 Pubmed/*csv  | grep -v == | sort | uniq -c
```
# now let's create one big, happy, spreadsheet:
```
head -n 1 Pubmed/*csv  | grep -v == | sort | uniq -c | tail -n 1 | awk {'print $2}' | tee combined.csv
for x in Pubmed/*csv ; do tail -n+2 $x >> combined.csv ; done
```

Then load this into sqlite:
```bash
sqlite3
.open pubmed_data.db
.mode csv
.import combined.csv pubmed_data
CREATE INDEX pmid_index ON pubmed_data("pmid");
```

Mission all the below: failed:
Step 1: Start grobid:
    - pull grobid docker image: docker pull grobid/grobid:0.8.0
    - install nvidia-container-toolkit https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt
    - run the container (in this case on one gpu): docker run --rm --gpus '"device=1"' --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.8.0
Step 2: Install and set up robotreviewer: https://github.com/ijmarshall/robotreviewer (I used the docker version in the instructions)
    - note: I had to move and add a command in the Dockerfile:
        ADD robotreviewer_env.yml /tmp/robotreviewer_env.yml # move to BEFORE the USER deploy action
        RUN chmod 1777 /tmp/robotreviewer_env.yml
    - note: I modified the grobid version in the docker file before running `docker compose up -d`
Step 3: Trialstreamer:
    - postgres:
        psql
        createdb trialstreamer;
        createuser trialstreamer;
        alter user trialstreamer with password 'trialstreamer';
    
app:
    - CREATE TABLE users(uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, username TEXT, salt TEXT, hash TEXT, creationdate TEXT, passworddate TEXT);
    - CREATE TABLE user_topics(topic_uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, uid INTEGER, topic_name, search_text, search_query);






################### from scratch:
- process pubmed and take the resulting database:
CREATE INDEX pubmed_extractions_index_pmid on pubmed_extractions(pmid);
# does it make sense to index on the title?
#CREATE INDEX pubmed_extractions_index_title on pubmed_extractions(title);
CREATE INDEX pubmed_extractions_ico_re_index_pmid on pubmed_extractions_ico_re(pmid);
# drop any unncessary indexes so we can drop the "Index" column (oops; pandas)
drop index ix_pubmed_extractions_index;
drop index ix_pubmed_extractions_ico_re_index;

CREATE TABLE IF NOT EXISTS "pubmed_extractions" (
"index" INTEGER,
  "pmid" TEXT,
  "title" TEXT,
  "abstract" TEXT,
  "mesh_terms" TEXT,
  "keywords" TEXT,
  "is_rct" INTEGER,
  "prob_rct" REAL,
  "is_rct_sensitive" INTEGER,
  "is_rct_balanced" INTEGER,
  "is_rct_precise" INTEGER,
  "prob_lob_rob" REAL,
  "num_randomized" TEXT,
  "study_design" TEXT,
  "prob_sr" REAL,
  "is_sr" REAL,
  "prob_cohort" REAL,
  "is_cohort" REAL,
  "prob_consensus" REAL,
  "is_consensus" REAL,
  "prob_ct" REAL,
  "is_ct" REAL,
  "prob_ct_protocol" REAL,
  "is_ct_protocol" REAL,
  "prob_guideline" REAL,
  "is_guideline" REAL,
  "prob_qual" REAL,
  "is_qual" REAL,
  "p" TEXT,
  "p_mesh" TEXT,
  "i" TEXT,
  "i_mesh" TEXT,
  "o" TEXT,
  "o_mesh" TEXT
);
CREATE TABLE IF NOT EXISTS "pubmed_extractions_ico_re" (
"Intervention" TEXT,
  "Outcome" TEXT,
  "Comparator" TEXT,
  "Evidence" TEXT,
  "Sentence" TEXT,
  "pmid" TEXT
);
CREATE INDEX pubmed_extractions_index_pmid on pubmed_extractions(pmid);
CREATE INDEX pubmed_extractions_ico_re_index_pmid on pubmed_extractions_ico_re(pmid);
CREATE TABLE users(uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, username TEXT, salt TEXT, hash TEXT, creationdate TEXT, passworddate TEXT);
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE user_topics(topic_uid INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE, uid INTEGER, topic_name, search_text, search_query, final INTEGER DEFAULT 0, generated_query default NULL, used_cochrane_filter default 0, used_robot_reviewer_rct_filter default 0);
CREATE TABLE search_screening_results(topic_uid, pmid, human_decision TEXT DEFAULT "Unscreened", robot_ranking REAL DEFAULT NULL, source TEXT DEFAULT NULL, UNIQUE(topic_uid, pmid));
CREATE INDEX search_screening_results_pmid ON search_screening_results(pmid);
CREATE INDEX search_screening_results_topic_uid_pmid ON search_screening_results(topic_uid, pmid);
CREATE INDEX search_screening_results_topic_uid_pmid_robot_ranking ON search_screening_results(topic_uid, pmid, robot_ranking);
CREATE TABLE IF NOT EXISTS "titles_abstracts"(idx INTEGER PRIMARY KEY AUTOINCREMENT, pmid, title, abstract);
CREATE INDEX pubmed_abstracts_pmid ON "titles_abstracts"(pmid);
CREATE TABLE IF NOT EXISTS "pubmed_data"(
"pmid" TEXT, "titles" TEXT, "abstracts" TEXT);
CREATE INDEX pmid_index ON pubmed_data("pmid");
CREATE TABLE pubmed_extractions_numerical(pmid, intervention, comparator, outcome, outcome_type, binary_result, continuous_result);
CREATE INDEX pubmed_extractions_numerical_pmid on pubmed_extractions_numerical("pmid");


######################## AWS
- create an ec2 instance with a 1tb extra disk
sudo mkfs -t xfs /dev/nvme1n1
sudo mount /dev/nvme1n1 /data/
sudo useradd -m -g users website
sudo groupadd website
sudo usermod -a -G website ubuntu
sudo mkdir /data/ei_demo
sudo chmod -R ug+rwX /data/
sudo chown -R website:website /data/
