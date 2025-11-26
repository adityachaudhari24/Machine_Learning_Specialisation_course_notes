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

- Excellent visualization in this video : https://www.youtube.com/watch?v=3dhcmeOTZ_Q&t=35s
- Given b and w , cost function helps us to understand how good or bad our model is performing. This is by measuring the difference between predicted 
  values and actual values.
- The goal of linear regression is to find the parameters `w` and `b` that results in smallest possible value for the cost J.
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

- Excellent video here - https://www.youtube.com/watch?v=sDv4f4s2SB8
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

- `connecting with the intuition of the dot product` : As dot product measures the similarity between two vectors, multiplying all input X features with their corresponding weights W and summing them up gives us a weighted sum that represents the model's prediction.
- ![alt text](image-24.png)
- ![alt text](image-25.png)
- ![alt text](image-26.png)
- ![alt text](image-27.png)
- Vectorization benifits - make code shorter, also makes code run faster by leveraging optimized linear algebra libraries which use parallel hardware capabilities.

</details>

<details>
<summary>🎯Q. Why vectorization is fast compared to normal computation?</summary>

- ![alt text](image-28.png)
- ![alt text](image-29.png)
</details>

<details>
<summary>🎯Q. Gradient descent for multiple linear regression with vectorization ?</summary>

- ![alt text](image-30.png)
- ![alt text](image-31.png)
- ![alt text](image-32.png)
- 
</details>

<details>
<summary>🎯Q. what is feature scaling and why its important ?</summary>

- Feature scaling is a technique used to standardize the range of independent variables or features of data. In machine learning, it is important because many algorithms are sensitive to the scale of the input features. If the features are on different scales, it can lead to suboptimal performance and convergence issues during training.
- ![alt text](image-33.png)
- ![alt text](image-34.png)
- If we use training data as it is, features with larger scales can dominate the learning process, making it difficult for the model to learn from features with smaller scales.
- check below example, on why we need feature scaling? without feature scaling, gradient descent may take longer to converge or may not converge at all. (converge : means in layman terms - reach to the optimal solution, in our case optimal values of W and B)
- In below example snap observe how contours are elongated ellipses, indicating that the cost function changes more rapidly in one direction than the other. This can lead to slow convergence because gradient descent will take small steps in the direction of the steepest slope, which may not be aligned with the direction of the minimum.
- ![alt text](image-35.png)
- ![alt text](image-36.png)
- ![alt text](image-37.png)
- ![alt text](image-38.png)
- ![alt text](image-39.png)
- Different methods of feature scaling - 
  - Min-Max Scaling (Normalization)
  - Standardization (Z-score normalization)
  - Robust Scaling
  - Max Abs Scaling
</details>

<details>
<summary>🎯Q. How we would know scaling actually worked ? meaning how can we tell our gradient descent is converging?</summary>

- Job of the gradient descent is to define the parameters w and b such that hopefully minimize the the cost function J(w,b).
- ![alt text](image-40.png)
- ![alt text](image-41.png)
- ![alt text](image-42.png)
</details>

<details>
<summary>🎯Q. what is feature engineering ?</summary>

- Feature engineering is the process of using domain knowledge to select, modify, or create new features from raw data to improve the performance of machine learning models.
- It involves techniques such as feature selection, feature extraction, and feature transformation to enhance the quality and relevance of the input data used for training.
- Effective feature engineering can lead to better model accuracy, reduced overfitting, and improved interpretability of the results.
- ![alt text](image-43.png)

</details>

<details>
<summary>🎯Q. what is polynomial regression ?</summary>

- Polynomial regression is an extension of linear regression that allows us to model non-linear relationships between the input features and the output variable by introducing polynomial terms. Essentially, it fits a curve to the data instead of a straight line.
- ![alt text](image-44.png)
- ![alt text](image-45.png)

</details>

<details>
<summary>🎯Q. what is logistic regression algorithm why its used ?</summary>

- Logistic regression is a statistical method used for binary classification problems, where the goal is to predict one of two possible outcomes (e.g., yes/no, true/false, 0/1) based on input features.
- ![alt text](image-46.png)
- why linear regression does not work for classification problem ? because linear regression can produce values outside the range of 0 to 1, which are not valid probabilities. Logistic regression addresses this issue by using the logistic (sigmoid) function to map predicted values to probabilities between 0 and 1.
- ![alt text](image-47.png)
- check below snap for how logistic regression formula is derived
- ![alt text](image-48.png)
- ![alt text](image-49.png)
- ![alt text](image-50.png)
- ![alt text](image-51.png)
- ![alt text](image-52.png)
- ![alt text](image-53.png)
- ![alt text](image-54.png)
- This is excellent video to understand what is sigmoid function - https://www.youtube.com/watch?v=yIYKR4sgzI8
- In essence: sigmoid turns linear regression’s output → into a probability → which makes logistic regression suitable for classification.

</details>

<details>
<summary>🎯Q. what is decision bounday and how its calculated ?</summary>

- ![alt text](image-55.png)
- ![alt text](image-56.png)
- ![alt text](image-57.png)

</details>

<details>
<summary>🎯Q. Cost function for the logistic regression ?</summary>

- why seuqred error is not a good choice ? check below snap
- ![alt text](image-59.png)
- Remember the loss function measures how well you are doing on a one training example and is by `summing up` the losses of all training examples we get the cost function. Meaning it sums up the loss over all training examples keeping one example at a time to calculate the total loss.
- ![alt text](image-60.png)
- ![alt text](image-61.png)
- ![alt text](image-62.png)
- ![alt text](image-63.png)
- ![alt text](image-64.png)


</details>

<details>
<summary>🎯Q. Gradient descent for logistic regression ?</summary>

- ![alt text](image-65.png)
- 

</details>

<details>
<summary>🎯Q. what is overfitting and how to manage it ?</summary>

- ![alt text](image-66.png)
- overfit = high variance similarly underfit = high bias
- ![alt text](image-67.png)
- addressing overfitting techniques 
- ![alt text](image-68.png)
- ![alt text](image-69.png)
- ![alt text](image-70.png)
- ![alt text](image-71.png)


</details>

<details>
<summary>🎯Q. what regularization technique and its use ?</summary>

- Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the loss function. This penalty discourages the model from fitting the training data too closely, which can lead to poor generalization on unseen data.
- ![alt text](image-72.png)
- ![alt text](image-73.png)
</details>

<br>
<br>

# Module 2: Advanced Learning Algorithms

<details>
<summary>🎯Q. what is neuron and neural network?</summary>

- A `neuron` is a fundamental unit of a neural network that processes and transmits information. It receives input signals, applies a mathematical function (activation function) to these inputs, and produces an output signal that is passed to other neurons in the network.
- ![alt text](image-74.png)
- ![alt text](image-75.png)
- ![alt text](image-76.png)
- ![alt text](image-77.png)
- ![alt text](image-78.png)
- ![alt text](image-79.png)
- ![alt text](image-80.png)
- ![alt text](image-81.png)

</details>

<details>
<summary>🎯Q. what is forward prapogation ?</summary>

- Its an `algorithm to calculate the output of neural network given input features and weights and biases of each neuron`.
- We can download the weights and biases after training the model and use those to make predictions on new data using forward propagation.
- `Forward propagation` is the process by which `input data is passed through a neural network to generate an output`. It involves calculating the weighted sum of inputs, applying activation functions, and propagating the results through each layer of the network until the final output is produced. As its from left to right direction its called `forward propagation`.
- Remember forward prapogration is different then backpropogation.
  `Forward prapogation` - use for `making predictions`.
  `Backpropogation` - use for `training the model`.

- Example below : 
- ![alt text](image-82.png)
- ![alt text](image-83.png)
- ![alt text](image-84.png)
</details>

<details>
<summary>🎯Q. How model is pragrammed in tensorFlow? </summary>

- ![alt text](image-85.png)
- ![alt text](image-86.png)
- ![alt text](image-87.png)
- ![alt text](image-88.png)
- ![alt text](image-89.png)
- ![alt text](image-90.png)
- ![alt text](image-91.png)

</details>

<details>
<summary>🎯Q. Building a neural network using tensorFlow techniques? </summary>

- ![alt text](image-92.png)
- ![alt text](image-93.png)
- ![alt text](image-94.png)
- ![alt text](image-95.png)
</details>

<details>
<summary>🎯Q. Forward prapogation math behind the scene imple just by using python and numpy to build the intuition</summary>

- ![alt text](image-97.png)
- ![alt text](image-98.png)
- 💡 Think of `dense as a calculator that uses W and b to compute outputs`. The process of learning these values is handled by the training algorithm outside the dense function. This step adjusts W and b to minimize the loss.
- Inside tensorflow library, dense layer does forward prapogation by calculating the weighted sum of inputs and adding bias, then applying activation function to produce output.
- ![alt text](image-104.png)

</details>

<details>
<summary>🎯Q. what does activation function mean actually in forward propagation </summary>

- ![alt text](image-99.png)
- ![alt text](image-100.png)

</details>

<details>
<summary>🎯Q. Does forward prop consider defauly W and B values initially ? why?</summary>

- ![alt text](image-101.png)
- ![alt text](image-102.png)
- ![alt text](image-103.png)

</details>

<details>
<summary>🎯Q. Matrics operations revision</summary>

- ![alt text](image-105.png)
- ![alt text](image-106.png)
- ![alt text](image-107.png)
- ![alt text](image-108.png)
- ![alt text](image-109.png)
- ![alt text](image-110.png)
</details>


<details>
<summary>🎯Q. Training a neural network in tensorflow 3 key steps understanding</summary>

- ![alt text](image-111.png)
- `step 1` : specifiy the model which tells tensorflow how to compute for inference (forward prapogation)
- `step 2` : compiles the model using specific loss function
- `step 3` : train the model using training data, which internally uses backpropogation to update W and B values to minimize the loss function.
- ![alt text](image-112.png)
- ![alt text](image-113.png)
- ![alt text](image-114.png)
- ![alt text](image-115.png)
</details>


<details>
<summary>🎯Q. Alternative to sigmoid activation in neural network ?  Different activation functions </summary>

- ![alt text](image-116.png)
- `ReLU activation function` - A popular activation function that introduces non-linearity by outputting the input directly if it is positive; otherwise, it outputs zero.
- Most commonly used activation functions
- ![alt text](image-117.png)
- ![alt text](image-119.png)
- ![alt text](image-120.png)
- ![alt text](image-121.png)
- why we cannot use just linear activation in neural networks ? because using linear activation function in all layers of a neural network would make the entire network equivalent to a single-layer linear model, regardless of the number of layers. `This is because the composition of linear functions is still a linear function`. As a result, the network would not be able to learn complex patterns or representations from the data, limiting its ability to solve non-linear problems effectively. [ check below snap]
- ![alt text](image-122.png)
- ![alt text](image-123.png)
- ![alt text](image-124.png)

</details>

<details>
<summary>🎯Q. Multiclass classification </summary>

- ![alt text](image-125.png)
- ![alt text](image-126.png)

</details>

<details>
<summary>🎯Q. what is softmax </summary>

- `softmax regression algorithm` is a generalization of logistic regression that is used for `multi-class classification problems`, where the goal is to predict one of several possible classes or categories.
- ![alt text](image-127.png)
- ![alt text](image-128.png)
- ![alt text](image-129.png)
- ![alt text](image-130.png)
- ![alt text](image-131.png)
- IMP step below - Its hard to read / interprete as its softmax as before however its more accurate numerically stable version of softmax function.
- ![alt text](image-132.png)
- ![alt text](image-133.png)
- ![alt text](image-134.png)

</details>


<details>
<summary>🎯Q. Multiclass classification vs multi-label classification problems </summary>

- Multiclass classification - In multiclass classification, each instance is assigned to one and only one class from a set of multiple classes. For example, classifying images of animals into categories like cat, dog, or bird.
- Multi-label classification - In multi-label classification, each instance can be assigned to multiple classes simultaneously. For example, tagging a news article with multiple topics like politics, economy, and health.

- ![alt text](image-135.png)
- ![alt text](image-136.png)

</details>

<details>
<summary>🎯Q. Optimization algorithms apart from Gradient Descent ? </summary>

- ADAM is most popular optimization algorithm used in deep learning, difference between gradient descent and ADAM is that gradient descent uses a fixed learning rate for all parameters, while ADAM adapts the learning rate for each parameter based on the first and second moments of the gradients.
- ![alt text](image-137.png)
- ![alt text](image-138.png)
- ![alt text](image-139.png)

</details>

<details>
<summary>🎯Q. Additional layer types ? </summary>

- ![alt text](image-140.png)
- ![alt text](image-141.png)
- ![alt text](image-142.png)

</details>

<details>
<summary>🎯Q. What is derivative, generate intuition behind it ? </summary>

- ![alt text](image-202.png)
- ![alt text](image-203.png)
- ![alt text](image-143.png)
- ![alt text](image-144.png)
- ![alt text](image-145.png)
- ![alt text](image-146.png)
- ![alt text](image-147.png)
- ![alt text](image-148.png)

</details>

<details>
<summary>🎯Q. how to calculate derivative in neural network ? what is computational graph ? </summary>

- Its a key concept to understand backpropogation algorithm. Its a key idea in the deep learning.
- ![alt text](image-149.png)
- ![alt text](image-150.png)
- ![alt text](image-151.png)
- ![alt text](image-152.png)
- ![alt text](image-153.png)
- ![alt text](image-154.png)
- 
</details>

<details>
<summary>🎯Q. Debugging a learning algorithm and optimization tips ? </summary>

- ![alt text](image-155.png)
- ![alt text](image-156.png)
- ![alt text](image-174.png)
</details>

<details>
<summary>🎯Q. Techniques to evaluate the model ? </summary>

- ![alt text](image-157.png)
- ![alt text](image-158.png)
- ![alt text](image-159.png)
</details>

<details>
<summary>🎯Q. cross validation vs test set ? </summary>

- ![alt text](image-160.png)
- ![alt text](image-161.png)
- ![alt text](image-162.png)
- ![alt text](image-163.png)
</details>

<details>
<summary>🎯Q. Diagnosting Bias and Variance and how to regularize it and establish base line performance? </summary>

- ![alt text](image-164.png)
- ![alt text](image-165.png)
- key takeaway - High bias does not even doing well on training set, while high variance means it does much worse on corss validation and test set compared to training set.
- ![alt text](image-166.png)
- ![alt text](image-167.png)
- ![alt text](image-168.png)
- ![alt text](image-169.png)
- ![alt text](image-170.png)
</details>


<details>
<summary>🎯Q. what is Learning curve</summary>

- Learning curve is a graphical representation that shows how a machine learning model's performance improves over time as it is trained on more data. It typically plots the model's accuracy or error rate against the number of training examples or iterations.
- ![alt text](image-171.png)
- ![alt text](image-172.png)
- When we have high bias , increasing the training data will not help much as the model is too simple to capture the underlying patterns in the data. Hoewever, when we have high variance, increasing the training data can help the model generalize better and reduce overfitting.
- ![alt text](image-173.png)
</details>

<details>
<summary>🎯Q. Bias and Variance in the neural network</summary>

- Neural networks with big data gives us the new ways to handle bias and variance.
- ![alt text](image-176.png)
- ![alt text](image-177.png)
- ![alt text](image-178.png)
- ![alt text](image-179.png)

</details>

<details>
<summary>🎯Q. Iterative loop of ML development</summary>

- ![alt text](image-184.png)
- ![alt text](image-185.png)
- ![alt text](image-186.png)
- ![alt text](image-187.png)
- ![alt text](image-188.png)

</details>

<details>
<summary>🎯Q. Techniques to adding data to reduce overfitting</summary>

- If error analysis indicated that a subset of specific set of data is causing overfitting, then we can collect more data for that specific subset to reduce overfitting.
- ![alt text](image-189.png)
- ![alt text](image-190.png)
- ![alt text](image-191.png)
- ![alt text](image-192.png)
- ![alt text](image-193.png)
- ![alt text](image-194.png)

</details>

<details>
<summary>🎯⭐⭐⭐Q. Transfer learning : using data from differet task⭐⭐⭐</summary>

- Transfer learning is a machine learning technique where a model trained on one task is reused as the starting point for a model on a second task. It leverages the knowledge gained from the first task to improve performance on the second task, `especially when the second task has limited data`.
- why transfer learning works ? because `the early layers of a neural network learn general features` that are applicable to many tasks, while the later layers learn task-specific features. By reusing the early layers from a pre-trained model, we can take advantage of the general features learned from a large dataset and adapt them to the new task with less data.
- ![alt text](image-195.png)
- ![alt text](image-196.png)

</details>
<details>
<summary>🎯Q. Full cycle of machine learning project</summary>

- ![alt text](image-197.png)
- ![alt text](image-198.png)

</details>

<details>
<summary>🎯Q. Full cycle of machine learning project</summary>

- ![alt text](image-197.png)
- ![alt text](image-198.png)

</details>

<details>
<summary>🎯Q. Fairness Bias and Ethics</summary>

- ![alt text](image-199.png)
- ![alt text](image-200.png)
- ![alt text](image-201.png)

</details>
<br>
<br>

### Decision Trees

<details>
<summary>🎯Q. Decision Trees model</summary>

- Decision tree is a supervised learning algorithm used for classification and regression tasks. It splits the data into subsets based on the value of input features, creating a tree-like model of decisions. Each internal node represents a feature, each branch represents a decision rule, and each leaf node represents an outcome or class label.
- ![alt text](image-204.png)
- ![alt text](image-205.png)
- ![alt text](image-206.png)
- ![alt text](image-207.png)
- ![alt text](image-208.png)
- ![alt text](image-209.png)

</details>

<details>
<summary>🎯Q. Measuring the purity (enthropy)</summary>

- Enthropy is a measure of impruty in the data, it starts from 0 (pure) to 1 (impure) and come back to 0 (pure) again.
- ![alt text](image-210.png)
- ![alt text](image-211.png)

</details>

<details>
<summary>🎯Q. how to reduce enthropy </summary>

- Information gain is a measure of the reduction in entropy achieved by splitting the data based on a particular feature. It quantifies how much uncertainty is reduced after the split.
- ![alt text](image-212.png)
- ![alt text](image-213.png)
- ![alt text](image-214.png)
- ![alt text](image-215.png)
- ![alt text](image-216.png)
- ![alt text](image-217.png)

</details>

<details>
<summary>🎯Q. what is one-hot encoding ?</summary>

- One-hot encoding is a technique used to convert categorical variables into a binary matrix representation. Each category is represented as a binary vector, where only one element is "hot" (1) and all other elements are "cold" (0).
- ![alt text](image-218.png)
- ![alt text](image-219.png)
- ![alt text](image-220.png)

</details>

<details>
<summary>🎯Q. Decision trees on continuous valueed features</summary>

- ![alt text](image-221.png)
- ![alt text](image-222.png)
- 

</details>

<details>
<summary>🎯Q. Using multiple decision trees, also what is random forest ?</summary>

- Random forests are an ensemble learning method that combines multiple decision trees to improve accuracy and control overfitting. 
- Each tree is trained on a random subset of the data, and the final prediction is made by averaging the predictions of all the trees.
- ![alt text](image-223.png)
- ![alt text](image-224.png)
- ![alt text](image-225.png)
- ![alt text](image-226.png)
- ![alt text](image-227.png)
- ![alt text](image-228.png)
- ![alt text](image-229.png)

</details>

<details>
<summary>🎯Q. XGBoost decision tree</summary>

- ![alt text](image-230.png)
- ![alt text](image-231.png)
- ![alt text](image-232.png)

</details>

<details>
<summary>🎯Q. When to use the decision trees?</summary>

- Decision trees and Neural network both are powerful learning algorithms.
- ![alt text](image-233.png)

</details>


<details>
<summary>🎯Q. fsdfsdf</summary>

- dsd

</details>

<br>
<br>
<br>
<br>
<br>
<br>

# General Notes on Machine Learning

- Applying ML techniques to dig into large amounts of data can help discover patterns that were not immediately apparent. This is called `data mining`.
- ⭐Remember the `loss function measures` how well you are doing on `one training example` and by summing up the losses of all training examples we get the `cost function`.⭐
<details>
<summary>🎯Q. Difference Linear Regression vs Neural Network </summary>


</details>

- ⭐Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns and relationships in data.⭐
- check M2_week2_code/home/jovyan/work/C2_W2_Relu.ipynb for more details on why non-linear activation functions are important.
- The sigmoid is best for on/off or binary situations. The ReLU provides a continuous linear relationship. Additionally it has an 'off' range where the output is zero. The "off" feature makes the ReLU a Non-Linear activation.

<details>
<summary>🎯Q. Bias and Variance key tradeoff points</summary>

- ![alt text](image-175.png)
- ![alt text](image-118.png)
- ![alt text](image-180.png)
- ![alt text](image-181.png)
- ![alt text](image-182.png)
- ![alt text](image-183.png)

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

- dsd

</details>