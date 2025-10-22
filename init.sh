********** Coldkeys ********************
- CV
5Ct2 20 
5DUV 20 
5Dfo7 5
- G3
5Cys 15
5Hmox 11
5Hba 10
5F52 17
- G7
5F48 17
5Fvs 16
5FqZ 15
- Whale
5Dhy 20
5FHw 20
5Hpm 20
5EWQ 20
- Shark
5DWuy 10
5Fnu 10
5EXa 1
********* Git management ***************
project_name=MIID-subnet
git clone https://github.com/williamlewis0620/$project_name
cd $project_name

# add base remote project (once project building)
git remote add upstream https://github.com/yanez-compliance/$project_name

# merging recent update (whenever subnet github changed)
git fetch upstream
git merge upsteream/main

# push update (whenever updated new approach)
git add .
git commit . -m ""
git push origin main

Caution: work and push from test server and pull it from miner's server, DON't edit directly in miner's server
test 65.109.23.184



********* Launching or restarting nvgen service ***********
python MIID/miner/nvgen_service.py
Caution: Becareful to restart service due to the dendrite requestion failure.
Restart
 o when need to change nvgen_service.py itself, not other modules, 
 o if need to chanage other modules, follow versioning up process.

********* Registering new miner and launching *************
wallet=pub54-2
netuid=54
uid=196
port=8000

btcli w new-hotkey --name ${wallet} --hotkey net${netuid}_uid${uid} && btcli s register --netuid ${netuid} --name ${wallet} --hotkey net${netuid}_uid${uid}

pm2 start --name miner${uid}-$port python -- \
    neurons/miner_v2.py \
    --netuid ${netuid} \
    --wallet.name ${wallet} \
    --wallet.hotkey net${netuid}_uid${uid} \
    --axon.port $((10000+$uid)) \
    --logging.info \
    --neuron.nvgen_url 144.76.38.143:$port
pm2 log

**************** Modifying and versioning up modules *****************
# in test server
cd MIID/miner
cp generate_name_variation_v14.py generate_name_variation_v15.py 
cp calculate_possibility_v14.py calculate_possibility_v15.py 
cp ../validator/reward_v13.py ../validator/reward_v14.py 
cp parse_query_gpt_v5.py parse_query_gpt_v6.py

# add some changed to modules

# edit versions.py
{
	"default": "v1"
	"versions":
	{
		"parser": "v6"
		"generator": "v15"
	}
}

# pull it in miner's server

************* Comparing answer with other miners *********************
# in nvgen_service
# find the poor solved last task and copy whole content of _tasks.json from MIID/miner/tasks/*/* directory

final_score, overall_score

# in test server
# paste to pyscripts/hard_tasks/<task_id>.json

# run dendrite
# specify miner id in pyscripts/request_dendrite.py and run
python pyscripts/request_dendrite.py

# find the result from pyscripts/tasks/<task_id>/<miner_id>.json and compare with mine

*************** Download wandb data and classify into wandb_download_sorted ****************

# in test server
export WANDB_API_KEY=d70d96f54657ddbde00e4cff5466f4767f9a92ee # any wandb accounts token
python unittest/wandb_download.py   --project MIID-dev-test/MIID   --outdir ./wandb_downloads   --state finished failed running   --include-running   --workers 32   --runs-per-page 10 &&  for f in wandb_downloads/*/wandb-summary.json; do jq . "$f" > "${f%.json}-formated.json"; done

mkdir -p wandb_downloads_sorted && for d in wandb_downloads/validator-*; do [[ -d "$d" ]] || continue; name=$(basename "$d"); if [[ $name =~ validator-([0-9]+)-([0-9]{4}-[0-9]{2}-[0-9]{2})_([0-9]{2})-([0-9]{2})-([0-9]{2})_([a-z0-9]+) ]]; then vnum=${BASH_REMATCH[1]}; date=${BASH_REMATCH[2]}; hour=${BASH_REMATCH[3]}; parent="wandb_downloads_sorted/${date}_${hour}/validator-${vnum}"; mkdir -p "$parent"; mv "$d" "$parent/"; fi; done

for f in wandb_downloads/*/wandb-summary-formated.json; do score=$(jq -r '.step_details.uid_metrics["188"].rule_compliance.overall_score' "$f"); if [ "$score" != "1" ]; then echo "$score $f"; fi; done

# if failed to download, contact with Bromagiza, his timezone is EST in US


















