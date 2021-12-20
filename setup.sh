pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# load model weights
mkdir resources
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-h8A3AtB2Z_IQbmv2mROC3z4xew_Kehw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-h8A3AtB2Z_IQbmv2mROC3z4xew_Kehw" -O resources/hifi-gan.pth && rm -rf /tmp/cookies.txt