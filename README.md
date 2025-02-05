# Renewables_Scenario_Gen_GAN-TensorFlow
Unofficial implementation of Renewables Scenario Generation GAN (Chen et al., IEEE 2018) in PyTorch 2.

Y. Chen, Y. Wang, D. Kirschen and B. Zhang, "Model-Free Renewable Scenario Generation Using Generative Adversarial Networks," in IEEE Transactions on Power Systems, vol. 33, no. 3, pp. 3265-3275, May 2018, doi: 10.1109/TPWRS.2018.2794541

Original Codebase: https://github.com/chennnnnyize-zz/Renewables_Scenario_Gen_GAN

### Data Set Reference
* Solar Data: https://www.nrel.gov/grid/solar-power-data.html
* Wind Data: https://www.nrel.gov/grid/wind-integration-data.html

### Version Notes
The model was implemented and tested using `Python==3.11.9`. Further, the following modules were utilized (see [Requirements File](./requirements.txt)):
```
matplotlib==3.10.0
numpy==2.2.1
scipy==1.15.0
torch==2.5.1
```

### Usage
To conduct the experiments, the easiest way to get started is by cloning this repository and use one of the notebooks ([solar](./renewables_scenario_gen_gan_solar.ipynb)/[wind](./renewables_scenario_gen_gan_wind.ipynb)).

Alternatively, run it from the terminal.
```bash
py train.py --data=datasets/solar.csv --label=datasets/solar_label.csv --epochs=5000 --batch_size=32 --learning_rate=1e-4
```

### Results
The recreated experiments demonstrate the GAN's capability to resemble the data distribution and leverage the provided labels for conditioned generation of certain scenarios, even after short training trials. Obviously, the training can be extended and parameters adapted to improve the GAN's fidelity.

##### Solar Generation
The plots present synthesized [solar](./renewables_scenario_gen_gan_solar.ipynb) power generation scenarios arranged by ordinal label values.

<img src="../assets/generated_data_solar.png" alt="Genrated data sorted by labels" width="800"/>

##### Wind Generation
The plots present synthesized [wind](./renewables_scenario_gen_gan_wind.ipynb) power generation scenarios arranged by ordinal label values.

<img src="../assets/generated_data_wind.png" alt="Genrated data sorted by labels" width="800"/>
