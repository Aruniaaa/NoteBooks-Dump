
# 🔨 Workflow Used

Here, I will talk about the deep learning workflow I used to build almost all these models, of course, everyone has a different way to go, mine is listed below!<3

---

## Step 1: Data Preparation

**Goal:** Get your data ready for training.

### Generate or Load the Data

Examples:

```python
from sklearn.datasets import make_circles
X_numpy, y_numpy = make_circles()

# OR custom data
X_numpy = np.array([[i] for i in range(1000)], dtype=np.float32)
Y_numpy = np.array([[2 * i + 1 + np.random.randn() * 5] for i in range(1000)], dtype=np.float32)
```

### Convert Numpy Arrays → Tensors

```python
X = torch.from_numpy(X_numpy)
y = torch.from_numpy(Y_numpy)
```

### Split the Data

* Use `train_test_split()` or split manually into training and testing sets.

### Visualize

> “Visualize, visualize, visualize!”
> Plot your data to understand patterns, noise, and separability.

### Scale (if necessary)

* Apply normalization or standardization depending on your model.

---

## Step 2: Model Selection & Building

**Goal:** Choose and build a model suitable for your data type.

### Pick a Model Type

Look at your data plots — is it:

* **Linear?** → Try Linear Regression, Logistic Regression
* **Nonlinear?** → Try Neural Nets, SVM, or Random Forest
* **Clustered?** → Try K-Means, DBSCAN
* **Noisy?** → Try regularization or ensemble methods

### Build the Model

* Define your architecture or initialize your chosen model.
* If building custom PyTorch models, remember to define a `forward()` method.

---

## Step 3: Optimizer, Loss Function & Metrics

**Goal:** Define how your model learns and how you’ll measure success.

### Choose a Loss Function

| Problem Type               | Common Loss                              |
| -------------------------- | ---------------------------------------- |
| Regression                 | `nn.MSELoss()`, `nn.L1Loss()`            |
| Binary Classification      | `nn.BCELoss()`, `nn.BCEWithLogitsLoss()` |
| Multi-Class Classification | `nn.CrossEntropyLoss()`                  |

### Choose an Optimizer

Common choices:

```python
torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
torch.optim.Adam(params=model.parameters(), lr=0.001)
```

### Pick Evaluation Metrics

| Problem Type   | Useful Metrics                        |
| -------------- | ------------------------------------- |
| Classification | Accuracy, F1-score, Precision, Recall |
| Regression     | R² Score, MAE, MSE                    |

*Tip:* Check out [Scikit-Learn’s model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html).

---

## Step 4: Training & Testing Loops

**Goal:** Train your model, monitor performance, and adjust weights.

### Training Loop

* Feed data into the model.
* Calculate loss.
* Perform backpropagation using the optimizer.
* Record loss values per epoch.

### Testing Loop

* Evaluate the model on unseen data.
* Compare performance metrics.

*Remember:*

* Loss & optimizer are **mandatory** (they drive learning).
* Evaluation metrics are **optional but recommended** to track real progress.

---

## Step 5: Analyze, Experiment, Rebuild

**Goal:** Reflect, tweak, and re-run.

Ask yourself:

* Did the loss decrease steadily?
* Did accuracy improve?
* Could the learning rate, optimizer, or number of epochs be off?
* Did you forget to scale or shuffle the data?

Then:

* Change **only one hyperparameter at a time** so you know what made the difference.
* Re-run training and testing.

---

## Step 6: Repeat Until Satisfied

**Goal:** Iterate and refine.
Keep repeating Step 5 until performance feels right.

---

## Step 7: Save & Celebrate 

**Goal:** Save your trained model and take a well-earned break.

Options:

* Save with **Pickle** or **Joblib**
* Example:

  ```python
  import joblib
  joblib.dump(model, 'model.pkl')
  ```

---

**TL;DR**

> Data → Model → Loss → Optimizer → Train → Evaluate → Tune → Save → Chill 

