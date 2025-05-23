import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Step 1: Create the dataset
data = pd.read_csv('ID3.csv')
print(data)

# Step 2: Separate input and output
X = data.drop('Goes', axis=1)  # input features
y = data['Goes']               # target column

# Step 3: Convert categorical data to numbers
X_encoded = pd.get_dummies(X)
print(X_encoded)

# Step 4: Train the decision tree using entropy (ID3)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_encoded, y)

# Step 5: Show the decision tree
plt.figure(figsize=(10,6))
tree.plot_tree(model, feature_names=X_encoded.columns, class_names=model.classes_, filled=True)
plt.show()

# Step 6: Predict for a new example
new_data = pd.DataFrame({
    'Time': ['Morning'],
    'Weather': ['Sunny'],
    'Temperature': ['Cold'],
    'Company': ['Yes'],
    'Humidity': ['Normal'],
    'Wind': ['Strong']
})


# Convert new data like before
new_encoded = pd.get_dummies(new_data)
new_encoded = new_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Predict
prediction = model.predict(new_encoded)
print("Will the person go out?", prediction[0])



