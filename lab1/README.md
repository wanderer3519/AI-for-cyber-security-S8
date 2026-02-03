## Directory structure

```
root/
- data/
    - processed_data.py (given data)
- src/
    - NaiveClassifier.py (Contains the naive classifier)
    - utils/
        - email_read_util.py (provided helper functions)
        - blacklist.py (Contains a list of blacklist words)
- main.py (Main script to run)
- requirements.txt (Dependencies of the project)
```

## How to run
Make sure the directory structure is like the above and run the following command

``` bash []
$ python3 -m venv .venv


(For windows)
$ .venv/Scripts/activate

(For linux/mac)
$ source .venv/bin/activate

$ pip install -r requirements.txt
$ python3 main.py

```
