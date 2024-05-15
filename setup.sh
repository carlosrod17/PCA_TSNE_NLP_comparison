pip install -r requirements.txt

python -m spacy download es_core_news_md
python -m spacy download es_core_news_sm
python -c "import nltk ; nltk.download('stopwords')"