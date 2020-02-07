
#!/bin/sh

sudo apt install -y openjdk-8-jre screen htop git vim

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
rm -rf Miniconda3-latest-Linux-x86_64.sh

conda install -y pandas numpy networkx nltk spacy pyspark beautifulsoup4 scikit-learn
conda install pytorch cudatoolkit=9.0 -c pytorch
pip install -U newspaper3k textstat pandarallel simpletransformers
python -m nltk.downloader punkt vader_lexicon #-d /path/to/nltk_data
python -m spacy download en_core_web_lg 

git clone https://github.com/rwalk/gsdmm; cd gsdmm; python setup.py install; cd ..; rm -rf gsdmm

