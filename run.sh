pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements
python main.py --wrf_path ../../data/WRF --aqs_path ../../data/AQS --out_path runoutput
