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
