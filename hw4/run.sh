git clone https://gitlab.com/antoncoding/gan-models.git
python3 generate_preprocess.py --text_file $1
python3 generate.py
