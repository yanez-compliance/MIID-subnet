wallet=pub54-2
netuid=54
port=8000


for uid in 106 108 110 111 113 132 133 139 152 153 160 17 191 217 230 234 252 32 48 68 83 91; do
    pm2 start --name miner${uid}-$port python -- \
        neurons/miner_v2.py \
        --netuid ${netuid} \
        --wallet.name ${wallet} \
        --wallet.hotkey net${netuid}_uid${uid} \
        --axon.port $((10000+$uid)) \
        --logging.info \
        --neuron.nvgen_url 144.76.38.143:$port
done
