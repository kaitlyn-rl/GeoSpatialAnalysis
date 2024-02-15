# Geospatial analysis on public spaces and transport networks
## MAC-MIGS Industrial project

Supervised by Dr Stuart King (University of Edinburgh) and Dr Jonas Latz (Heriot-Watt University), in collaboration with Tim Hurst and Anne Braae from Brainnwave

Data provided by Tim: [Dropbox link](https://www.dropbox.com/s/bzd2lerk49kjced/wetransfer_edinburgh_2023-01-04_1350.zip?dl=0)

## How to use
Optional: Run `python src/graph_gen.py` to generate the graph, make sure the files required are within the `data` folder.

All functions needed are in `Graph.py`. See `main.py` for example usage for each metric. Uncomment the metric you want to use in `main.py` and then run `python src/main.py`. Unless otherwise specified, any plots will be saved in `results` folder, which will be created if it does not exist.