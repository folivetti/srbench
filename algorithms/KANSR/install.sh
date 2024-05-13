#install PyKAN

pip install -r requirements.txt
pip install git+https://github.com/KindXiaoming/pykan.git
loc=$(pip show pykan | grep Location: | cut -d ' ' -f 2)
cp utils.py $loc/kan/

