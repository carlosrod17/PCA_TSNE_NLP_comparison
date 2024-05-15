pip install subprocess

import subprocess

libraries = {'numpy': '1.23.5',
             'pandas': '2.0.0',
             'scipy': '1.10.1',
             'numba': '0.50.1',
             'threadpoolctl': '3.1.0',
             'matplotlib': '3.7.1',
             'seaborn': '0.12.2',
             'scikit-learn': '1.2.2',
             'emoji': '2.2.0',
             'nltk': '3.8.1',
             'spacy': '3.5.1',
             'pattern': '3.6',
             'gensim': '4.3.1',
             'mip': '1.15.0',
             'langdetect': '1.0.9'}

for lib, version in libraries.items():
    
    result = subprocess.run(['pip', 'install', lib+'=='+version],
                            capture_output=True,
                            text=True)
    
    if result.returncode != 0:
        print(f"Error al instalar {lib} ({version}):")
        print(result.stdout)
        print(result.stderr)
        
    else:
        print(f"{lib} ({version}) instalado con éxito.")
        
        
!python -m spacy download es_core_news_md
!python -m spacy download es_core_news_sm

import nltk

nltk.download('stopwords')

