# GeoGAD
## Dependencies:
  ## PyTorch 
  - pytorch==2.4.1
  - cudatoolkit=12.6
  ## Others
  - numpy==1.24.4
  - tqdm==4.67.1
  - pyro==1.9.1
 ## Before running the code, you need to execute the following script command
```
bash scripts/setup.sh
```
## Get Data
We have included the summary data used in our paper—sourced from SAbDab, RAbD, and SKEMPI_V2—in the summaries folder. Please download all structural data from the [download page of SAbDab](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/search/?all=true#downloads). As SAbDab is updated weekly, you may also obtain the latest summary file directly from its [official website](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab).
## Experiments
### K-fold evaluation on SAbDab

```
python -B train.py --cdr_type 3 --task 0
```

### Antigen-binding CDR-H3 Design

```
python -B train.py --cdr_type 3 --task 2 
```


### Affinity Optimization

```
python -B train.py --task 1
```
