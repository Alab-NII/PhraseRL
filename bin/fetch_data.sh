# Multi-WoZ
mkdir -p data

echo "Downloading MultiWOZ..."
curl -sL -o data/multiwoz_2_0.zip https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.0.zip
curl -sL -o data/norm-multi-woz.zip https://github.com/snakeztc/NeuralDialog-LaRL/raw/master/data/norm-multi-woz.zip
unzip -qq -o -d data 'data/*.zip'
mv "data/MULTIWOZ2 2" "data/MultiWOZ_2.0"

# Fix broken json
cp bin/multiwoz_fix/taxi_db.json data/norm-multi-woz/taxi_db.json
cp bin/multiwoz_fix/taxi_db.json data/MultiWOZ_2.0/taxi_db.json

rm -rf data/*.zip data/__MACOSX
