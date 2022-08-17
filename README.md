# T1T2 training and testing

We suggest creatining a new virtual environment for this project,  using `requirements.txt`.

---

The input T1 and T2 maps can be labelled using the published T1T2 labelling software, available at https://github.com/jphdotam/T1T2_labeller

---

The following scripts are present.

* `1_train.py` contains the training script, which is configurable using `experiments/001.yaml`
* `2_test_nn.py` generates a JSON file using the trained network which can then be used for assessing network performance
* `3_test_humans.py` generates the analogous JSON file for the two human experts
* `4_compare_predictions.py` processes the 2 JSON files and creates the Figures published in the manuscript, along with the descriptive statistics.
---

During training, performance can be visualised through Weights and Biases' online interface.
