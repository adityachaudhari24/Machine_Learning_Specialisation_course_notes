# Machine Learning Specialization course - Notes
This repository contains the notes and key concepts from below resources 
1) Machine Learning Specialization course by Andrew Ng on Coursera.
2) Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow by Aurélien Géron.
3) my own general research on the web and different books.

# Module 1: Supervised Machine Learning: Regression and Classification

<details>
<summary>🎯Q. What is machine learning?</summary>

- Machine Learning is the `science (and art) of programming computers` so they can learn from data.
- Machine learning is the field that gives abilities to computers to learn without being explicitly programmed. - Arthur Samuel (1959)
- ⭐⭐Machine learning⭐ and ⭐deep learning⭐ are fields aimed at `implementing algorithms` that enable computers to learn from data and perform tasks that typically require human intelligence.⭐
- There are below main types of machine learning:
  - Supervised Learning
  - Unsupervised Learning
  - Reinforcement Learning
- Each type has its own unique characteristics and applications.
</details>


<details>
<summary>🎯Q. What is supervised learning in ML? Different types (3 key types)</summary>

- `supervised learning` is an algorithm which maps input X to output Y, where the learning algorithm learns from the "right answers" (labeled data).
- `Supervised learning` is a type of machine learning where the model is trained on a labeled dataset, meaning that each training example is paired with an output label.

- `Two main types of Supervised Learning` - 
  - `Regression` - Predicting continuous values (e.g., house prices, stock prices).
  - `Classification` - Predicting discrete labels or categories (e.g., spam detection, image recognition).

⭐⭐ Regression supervised learnning ⭐⭐
- `Regression meaning` - Its finding the relationship which explains how one element depends on another. 
  - `Regression` also means trying to `predict a number` from infinitely `possible outputs`.
  - `Example` - Predicting house prices based on features like size, location, number of rooms, etc.
  - in ecommerce , given input features of user, product, context, predict purchase amount.


⭐⭐ Classification supervised learning ⭐⭐
- `Classification meaning` - Its finding the relationship which explains how one element belongs to a particular category or class.
  - `Classification` also means trying to `predict a label` from a `finite set of possible outputs`.
  - `Example` - Email spam detection (spam or not spam), image recognition (cat, dog, car, etc.)
    - in ecommerce , given input features of user, product, context, predict whether user will buy the product (yes/no)



- IMP One liners - 
- 💡 Its not always finding the straight line in supervised learning regression; this can be polynomial, logarithmic, exponential, etc.
- Classification algorithms predicts categories / classes / labels unlike regression which predicts continuous values( e.g., real numbers).
</details>


<details>
<summary>🎯Q. What is unsupervised learning in ML?</summary>

- In `unsupervised learning` we ask algorithms to find something intresting in unlabeled data. (interesting patterns) Where in `supervised learning` the algorithm learns from the labeled data.
- Examples 
  - `Anomaly detection` - identifying unusual data points that do not conform to expected behavior. Example - fraud detection in financial transactions.
  - `Clustering` - grouping similar data points together based on their features. Example - customer segmentation in marketing.
  - `Dimensionality reduction` - compress data using fewer numbers, reducing the number of features in a dataset while preserving important information. Example - Principal Component Analysis (PCA) for image compression.

</details>

<details>
<summary>🎯Q. what is meaning of `regression` in machine learning?</summary>

- In machine learning "regression" mean predicting continuous values based on input features.
- ![alt text](image-23.png)

</details>

<details>
<summary>🎯Q. What is linear regression model?</summary>

- Linear regression model is a type of regression model that assumes a linear relationship between the input features (independent variables) and the output variable (dependent variable). Meaning it tries to fit a straight line (or hyperplane in higher dimensions) to the data points.
- The goal of linear regression is to find the best-fitting line that minimizes the difference between the predicted values and the actual values in the dataset. 
- ![alt text](image.png)
- For linear regression, the model is represented by:

  **f<sub>w,b</sub>(x)<sup>(i)</sup> = wx<sup>(i)</sup> + b**

  Where:
  - **f<sub>w,b</sub>(x)** is the prediction function
  - `w` is the weight/slope parameter
  - `b` is the bias/y-intercept parameter
  - `x` is the input feature
- The formula above is how you can represent straight lines - different values of  𝑤 and  𝑏 give you different straight lines on the plot.
- The formula can be used to predict outcomes based on input features, making it a fundamental concept in machine learning.
- ![alt text](image-1.png)
- ![alt text](image-2.png)
- ![alt text](image-3.png)
</details>

<details>
<summary>🎯Q. what is cost function ?</summary>

- The `cost function` (or `loss function`) is used to define the best parameters for the model, in our liniear regression case the best values for w and b.
- The goal of linear regression is to find the parameters `w` or `w` and `b` that results in smallest possible value for the cost J.
- ![alt text](image-4.png)
- ![alt text](image-5.png)
- ![alt text](image-6.png)
- ![alt text](image-7.png)
- ![alt text](image-8.png)
- ![alt text](image-9.png)
- ![alt text](image-10.png)
- Cost and MOdel examples below
- ![alt text](image-11.png)
- ![alt text](image-12.png)
- ![alt text](image-13.png)
</details>


<details>
<summary>🎯Q. what is gradient descent ?</summary>

- `Gradient descent` is an optimization algorithm used to minimize the cost function by iteratively adjusting the model parameters (like `w` and `b` in linear regression) in the direction of the steepest descent of the cost function.
- ![alt text](image-14.png)
- ![alt text](image-15.png)
  - In Above diagram, "alpha" controls `how big a step` we take on each iteration towards the minimum.
  - If `"alpha"` is too small, the algorithm will take a long time to converge.
  - If `"alpha"` is too large, the algorithm may overshoot the minimum and fail
  - If `"alpha"` is just right, the algorithm will converge quickly to the minimum.
  - `derivative` tell us in which direction to adjust the parameters to reduce the cost.
  - `"derivative"` is the `slope of the cost function at the current point`.
  - `alpha * derivative` gives the amount to adjust the parameters (step size) per iteration.
- ![alt text](image-16.png)
- ![alt text](image-17.png)
- ![alt text](image-20.png)
- ![alt text](image-21.png)
- In below diagram, we can see until to find the global minimum point of cost function, we need to keep updating w and b values using gradient descent algorithm.
- ![alt text](image-22.png)
- In `batch gradient descent`, we use the entire training dataset to compute the gradient of the cost function for each iteration. This means that for every update of the model parameters, we consider all training examples. Meaning in gradient descent we have algorithms to calculate the W and B here it use entire batch of data to compute the gradient of the cost function for each iteration.
</details>

<details>
<summary>🎯Q. what is learning rate? </summary>

- The `learning rate` (denoted as `α` or "alpha") is a hyperparameter that controls the step size at each iteration while moving toward a minimum of a loss function during the training of a machine learning model.
- ![alt text](image-18.png)
- ![alt text](image-19.png)
</details>

<details>
<summary>🎯Q. Linear regression with multiple features? how its denoted ?</summary>

- ![alt text](image-24.png)
- 

</details>


<br>
<br>
<br>
********************************* Ignore below *********************************
<details>
<summary>Emojis used</summary>
⭐ - For important points
🔥 - super important
💡 - For key concepts/tips
⚠️ - For warnings/common mistake
🎯 - For exam targets/focus areas/ question 
🚀 - For advanced topics .
🚫 - For indicating something that cannot be used or a concerning point
</summary>
</details>

<details>
<summary>🎯Q. fsdfsdf</summary>

- 

</details>