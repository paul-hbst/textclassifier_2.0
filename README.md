# textclassifier_2.0
### Description
This is a basic example for a Classifier Service for the SLR Toolkit.
This includes basic auth that should be improved when hosted publicly.

### Setup:
Create a .env file with 
`BASIC_AUTH_PASSWORD=
BASIC_AUTH_USERNAME=`

Setup python venv
`python3 -m venv env`
Activate venv
`source env/bin/activate`
Install requirements
`pip install -r requirements.txt`

### Run:

`source env/bin/activate`

`fastapi dev main.py`
