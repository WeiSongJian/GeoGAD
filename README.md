# GeoGAD
## Dependencies:
  ## PyTorch 
  - pytorch==1.8.1
  - cudatoolkit=11.3
  ## Others
  - numpy==1.20.1
  - tqdm==4.59.0
  - torch_scatter==2.0.8
  - pyro==1.5.0
 ## Before running the code, you need to execute the following script command
```
bash scripts/setup.sh
```
## Get Data
We have included the summary data used in our paper—sourced from SAbDab, RAbD, and SKEMPI_V2—in the summaries folder. Please download all structural data from the download page of SAbDab. As SAbDab is updated weekly, you may also obtain the latest summary file directly from its official website.
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
