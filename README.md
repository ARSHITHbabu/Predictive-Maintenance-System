# ProactiveCare AI

## Project Overview
The **ProactiveCare AI** project uses advanced machine learning and deep learning techniques to predict the Remaining Useful Life (RUL) of machinery and equipment. By analyzing sensor data, it helps organizations optimize maintenance schedules, reduce downtime, and improve operational efficiency. The project integrates **Deep Learning (DL)**, **Reinforcement Learning (RL)**, and **SHAP** for model explainability.

---

## How It Is Used
This system is applied in industries such as manufacturing, aerospace, and automotive. By analyzing sensor data, it can predict when a machine or equipment is likely to fail, allowing companies to perform maintenance before failures occur. The model can be deployed as a real-time monitoring system or used as a batch process for periodic analysis of equipment health.

---

## Techniques Used

### 1. **Deep Learning (DL)** 
- **What is it?**: Deep learning is a subset of machine learning that uses neural networks with many layers to model complex relationships within data.
- **Why is it used?**: Deep learning models can learn intricate patterns in large datasets, making them ideal for predictive tasks like estimating the RUL of machinery based on sensor readings.
- **How is it used?**: In this project, a deep neural network (DNN) model is trained to predict the RUL based on historical sensor data.

### 2. **Reinforcement Learning (RL)**
- **What is it?**: RL involves an agent learning to make decisions by interacting with an environment to maximize cumulative rewards.
- **Why is it used?**: RL is utilized to optimize maintenance strategies by learning the best actions (i.e., when to perform maintenance) based on the predicted RUL.
- **How is it used?**: Q-learning, an RL technique, is implemented to choose optimal actions based on a Q-table that maps states (RUL predictions) to actions (maintenance decisions).

### 3. **SHAP (SHapley Additive exPlanations)**
- **What is it?**: SHAP is a model-agnostic method for interpreting machine learning models by assigning importance values (SHAP values) to each feature in a prediction.
- **Why is it used?**: SHAP is used to enhance the interpretability of the deep learning model by explaining the contribution of each sensor to the predicted RUL.
- **How is it used?**: SHAP values are calculated for each sensor reading, allowing users to understand which sensors have the most influence on the model's predictions.

---

## Why It Is Used
Predictive maintenance helps organizations:
- **Reduce costs**: Prevent expensive unplanned downtimes by performing maintenance before failures occur.
- **Enhance safety**: Identify potential equipment failures early to mitigate risks and prevent accidents.
- **Optimize operations**: Efficiently allocate maintenance resources, increasing the availability and reliability of machinery.

---

## Project Components and Cell Descriptions

### Parameters
- **Sensor Data**: Data collected from multiple sensors on the equipment, providing real-time monitoring for predictive analysis.
- **RUL (Remaining Useful Life)**: The number of operational cycles or time remaining before equipment failure.
- **Learning Rate**: A hyperparameter controlling the model’s step size during optimization.
- **Discount Factor**: A parameter used in reinforcement learning to weigh future rewards.

---

### Code Structure and Outputs

#### Cell 1: Import Necessary Libraries
**Output**: No direct output.
- This cell imports libraries like NumPy, Pandas, Matplotlib, Gym, TensorFlow, and SHAP for various tasks, including data manipulation, model building, and visualization.

---

#### Cell 2: Function to Create an Enhanced Synthetic Dataset
**Output**: No direct output.
- Defines a function that generates synthetic data, simulating sensor readings and the corresponding RUL. The function enhances the dataset with additional features for better model performance.

---

#### Cell 3: Generate Enhanced Synthetic Data and Save to CSV
**Output**: Displays a preview of the first few rows of the synthetic dataset.
- Calls the function to generate and save the dataset as a CSV file. The preview allows verification of the data format and feature structure.

---

#### Cell 4: Define the Maintenance Environment
**Output**: No direct output.
- Implements a custom environment using OpenAI Gym, where states represent RUL, and actions include performing maintenance or letting the machine run. This environment is the basis for training the reinforcement learning agent.

---

#### Cell 5: Initialize the Environment
**Output**: No direct output.
- Initializes an instance of the maintenance environment, making it ready for interaction with the RL agent.

---

#### Cell 6: Create a Deep Learning Model for RUL Prediction
**Output**: No direct output.
- Constructs a deep neural network using Keras. The model consists of input layers (sensor data), hidden layers (to capture complex relationships), and output layers (predicted RUL).

---

#### Cell 7: Train the Deep Learning Model on the Synthetic Data
**Output**: Training logs showing the loss per epoch.
- Trains the model on the synthetic data and outputs the training progress. The model’s performance improves as the loss decreases across epochs.

---

#### Cell 8: Plot Training Loss Over Epochs
**Output**: A plot showing the training loss over epochs.
- Visualizes the model’s learning curve. A steadily decreasing loss indicates that the model is effectively learning to predict RUL.

---

#### Cell 9: Initialize Q-table and Hyperparameters
**Output**: No direct output.
- Initializes the Q-table, which the RL agent uses to store knowledge about the environment, and sets hyperparameters such as learning rate and discount factor.

---

#### Cell 10: Function to Discretize the State
**Output**: No direct output.
- Converts continuous RUL predictions into discrete bins, making it easier for the Q-learning agent to work with the data.

---

#### Cell 11: Training the Q-learning Agent
**Output**: Logs showing the agent’s progress over episodes.
- Trains the Q-learning agent by interacting with the environment. The agent learns to take optimal maintenance actions, improving over time as it receives rewards for good decisions.

---

#### Cell 12: Plot Total Rewards Over Episodes
**Output**: A plot showing the total rewards earned by the RL agent.
- Visualizes the total rewards accumulated over episodes, helping to track the agent’s learning progress.

---

#### Cell 13: Evaluate the Q-learning Agent
**Output**: The total reward earned by the agent in the evaluation phase.
- Evaluates the trained agent by running it through the environment without further learning, showing how well it has learned to optimize maintenance decisions.

---

#### Cell 14: Function to Create Time-Series Features
**Output**: No direct output.
- Defines a function to generate time-series features like rolling means and standard deviations, which are crucial for capturing trends in sensor data over time.

---

#### Cell 15: Enhance Synthetic Data with Time-Series Features
**Output**: Displays the enhanced dataset with new time-series features.
- Applies the time-series feature creation function and shows the last 15 rows of the dataset to verify the added features.

---

#### Cell 16: Visualize the Original Sensor Data and the New Features
**Output**: Plots of original sensor data alongside new features (rolling means, standard deviations).
- Visualizes the newly created features, which help in better understanding the temporal patterns in sensor data.

---

#### Cell 17: Create Lag Features and Additional Features
**Output**: No direct output.
- Generates lag features, exponential moving averages, and Z-scores for anomaly detection. These features help the model capture equipment behaviors that lead to failure.

---

#### Cell 18: SHAP Analysis for Feature Importance
**Output**: SHAP values plot showing feature importance.
- Uses SHAP to explain the deep learning model's predictions, showing the contribution of each feature (sensor) to the RUL prediction.

---

#### Cell 19: Visualize the Pairwise Relationships in the Dataset
**Output**: A pairplot visualizing relationships between features.
- Provides a visual representation of the relationships between different features, helping to identify patterns and correlations.

---

#### Cell 20: Create a Correlation Heatmap
**Output**: A heatmap of feature correlations.
- Displays the correlation matrix for the dataset, aiding in feature selection by identifying which features are highly correlated.

---

#### Cell 21: Plot Predicted vs. Actual RUL
**Output**: A plot comparing predicted RUL with actual RUL.
- Visualizes the accuracy of the model by plotting the predicted RUL against the actual values. The closer the points are to the line of equality, the better the model’s predictions.

---

#### Cell 22: Impact of Sensors on Predicted RUL
**Output**: A plot showing the impact of individual sensors on the predicted RUL.
- Displays how sensor readings affect the predicted RUL, providing actionable insights into which sensors are critical for monitoring.

---

## Usage of This Project
This project can be used by organizations to implement real-time predictive maintenance systems. It provides actionable insights into equipment health, allowing for:
- **Proactive maintenance planning**.
- **Reduction in unexpected equipment failures**.
- **Improved resource allocation**.

---

## Future Updates
Potential enhancements include:
1. **Real-time data integration**: Incorporating real-time sensor data to provide continuous monitoring of equipment.
2. **Advanced RL algorithms**: Exploring more sophisticated RL techniques like deep Q-networks (DQNs) for better maintenance strategy optimization.
3. **Anomaly detection**: Implementing unsupervised learning methods to detect anomalies in sensor data that could indicate unusual equipment behavior.
4. **Transfer learning**: Allowing the model to generalize across different types of machinery or equipment.
5. **Scalability**: Optimizing the system to handle larger datasets from multiple machines across various industries.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## .gitignore
