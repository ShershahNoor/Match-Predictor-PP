
Personal Project Football AI Predictor Outline 

This project uses Python to take in information from SportsAPI, Football database.
The Python libaries used are:
- pytorch
- os
- json
- time
- random
- requests
- numpy
- streamlit

The pytorch framework is a Multi-Layer Perceptron that takes in various points of data,
and outputs who it predicts will win the match along with a confidence score.

This is available for the big 5 leagues in Europe, the Premier League, Ligue 1, Serie A,
Bundesliga, and La Liga. 

How to Use

Open your terminal after saving the desired file and downloading the needed libraries,
by running command "pip install -r requirements.txt", then run the command:

"streamlit run (filename).py" 

This will open a local hosted streamlit frontend on your browser where you can interact 
with the UI, as in selcting your team and running the predictor to guess which team wins.

Known Issues
- Some teams cause error's as there is not sufficent data from the API.

Possible Solutions
- Regress the year the data is being drawn from.
