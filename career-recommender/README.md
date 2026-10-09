# PathFinder - Career & Course Recommender
SDG 4 (Quality Education) + SDG 8 (Decent Work). Flask + scikit-learn (RandomForest).

Run locally: `pip install -r requirements.txt && python app.py` -> http://127.0.0.1:5000
Deploy on Render: push to GitHub -> New Web Service -> Build `pip install -r requirements.txt`, Start `gunicorn app:app --workers 1 --timeout 120`.
