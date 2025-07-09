# Final year project
DETECTION OF CYBER ATTACKS IN A NETWORK USING MACHINE LEARNING ALGORITHMS 


### **Abstract**
*   The increasing time spent online has led to a rise in cyberattacks and cybercrimes.
*   Traditional protection methods are insufficient against zero-day and sophisticated attacks.
*   Machine learning techniques are being developed to detect and combat these threats.
*   The project investigates three primary machine learning techniques: Deep Belief Network, Decision Tree, and Support Vector Machine.

### **Introduction**
*   The rapid evolution of connected technologies (smart grids, IoT, 5G) has led to a massive increase in IP-associated devices and traffic.
*   This growth raises significant security concerns due to the exchange of sensitive data over untrusted networks.
*   Intrusion Detection Systems (IDS) are crucial for identifying internal and external intrusions and suspicious activities.
*   ID systems can be signature-based, anomaly-based, or hybrid. Anomaly-based IDSs, in particular, utilize statistical, information-based, and AI procedures, with recent research focusing on deep learning.
*   The document highlights the increasing sophistication of cybercrimes beyond simple login credential evaluation, emphasizing the importance of data security (availability, secrecy, genuineness).
*   Initial attack stages often involve reconnaissance, such as port scanning, which provides critical information to attackers. Machine learning algorithms like SVM are being applied to IDS models to detect such attempts.

### **Literature Survey**
*   The literature review focuses on work utilizing the NSL-KDD dataset for performance benchmarking.
*   Early work used Artificial Neural Networks (ANN) with improved backpropagation for IDS, showing reduced performance when tested with unlabeled data.
*   Later studies employed J48 decision tree classifiers and Random Tree models, demonstrating improved accuracy and lower false alert rates with reduced feature sets.
*   Two-level classification approaches, such as Discriminative Multinomial Naive Bayes (DMNB) and Ensembles of Balanced Nested Dichotomies (END) with Random Forest, have shown improved detection rates and lower false positive rates.
*   Principal Component Analysis (PCA) combined with SVM (using Radial Basis Function) achieved high detection accuracy, further improved by feature selection using information gain.
*   Other methods, including fuzzy classification with genetic algorithms and k-point algorithms, were explored for their detection accuracy and false positive rates.
*   The Optimal Path Forest (OPF) method was found to offer high identification accuracy in less time compared to SVM RBF.

### **System Analysis**
*   **Existing System Disadvantages:**
    *   Malicious cyber-attacks pose serious security issues due to their continuous evolution and high volume.
    *   Data theft is a significant problem.
    *   Lack of detailed performance analysis of various machine learning algorithms on publicly available datasets.
    *   Need for systematic updates and benchmarking of malware datasets due to the dynamic nature of attacking methods.
*   **Proposed System Advantages:**
    *   Protection against malicious network attacks.
    *   Deletion and/or quarantining of malicious elements within a network.
    *   Prevention of unauthorized user access.
    *   Denial of access to infected resources for programs.
    *   Securing confidential information.
*   **Software Requirements:**
    *   Operating System: Windows 7 Ultimate or above.
    *   Coding Language: Python.
    *   Front-End: Python.
    *   Back-End: Flask.

### **Software Environment**
*   **Django:** A Python-based web application framework (MVT design pattern) known for rapid development, security (protection against SQL injection, XSS, CSRF), scalability, and versatility. It includes modules for user authentication, content administration, and more.
    *   **Model Layer:** Provides an abstraction layer for structuring and manipulating data (models, QuerySets, migrations).
    *   **View Layer:** Encapsulates logic for processing user requests and returning responses (URLconfs, view functions, file uploads, class-based views, middleware).
    *   **Template Layer:** Offers a designer-friendly syntax for rendering information to the user.
    *   **Forms:** A rich framework for creating and manipulating form data.
    *   **Development Process:** Tools and components for developing and testing Django applications (settings, applications, exceptions, django-admin, deployment).
    *   **Admin:** Automated admin interface.
    *   **Security:** Built-in tools for security (clickjacking, CSRF protection, cryptographic signing).
    *   **Views:** Python functions that take a web request and return a web response, forming part of the user interface.
    *   **User Authentication:** Handles authentication and authorization, including users, permissions, groups, password hashing, and login tools.
*   **Machine Learning (ML):**
    *   Defined as an application of artificial intelligence where algorithms process statistical data to learn from past data and improve automatically.
    *   ML is data-driven and similar to data mining.
    *   **How ML Works:** Systems learn from historical data to build prediction models for new data. Accuracy depends on the amount of data.
    *   **Types of Learning:**
        *   **Supervised Learning:** Builds models from labeled data (inputs and desired outputs). Includes classification (limited set of values, e.g., spam/not spam) and regression (continuous outputs, e.g., temperature).
        *   **Unsupervised Learning:** Builds models from unlabeled data to find structure (grouping/clustering, density estimation, dimensionality reduction).
        *   **Active Learning:** Optimizes choice of inputs for acquiring training labels.
        *   **Reinforcement Learning:** Agents learn to maximize cumulative reward in a dynamic environment (e.g., autonomous vehicles, game playing).
    *   **Relation to Data Mining:** ML focuses on prediction, while data mining focuses on discovering unknown properties.
    *   **Relation to Statistics:** ML finds generalizable predictive patterns, while statistics draws population inferences from samples.
    *   **Prerequisites for ML:** Fundamental knowledge of probability, linear algebra, calculus, and Python coding.
    *   **Classification Algorithms:** Logistic Regression, Support Vector Machines, K-Nearest Neighbors (KNN), Naïve Bayes, Decision Tree, Random Forest.
*   **Python Packages:**
    *   **NumPy:** Numerical Python, for multidimensional array objects and mathematical operations.
    *   **Pandas:** High-performance data manipulation and analysis tool, built on NumPy, with DataFrame objects.
    *   **Keras:** High-level neural networks API for deep learning, running on TensorFlow or Theano.
    *   **Scikit-learn (Sklearn):** Free ML library for Python, featuring various algorithms (SVM, random forests, k-neighbors) and supporting NumPy and SciPy.
    *   **SciPy:** Open-source Python library for scientific and mathematical problems, built on NumPy.
    *   **TensorFlow:** Python library for fast numerical computing, used for deep learning models.
    *   **Django:** High-level Python web framework for rapid development.
    *   **Pyodbc:** Python module for accessing ODBC databases.
    *   **Matplotlib:** Visualization library for 2D plots.
    *   **OpenCV:** Python bindings for computer vision problems.
    *   **NLTK:** Natural Language Toolkit for processing human language data.
    *   **SQLAlchemy:** Library for communication between Python programs and databases (ORM tool).
    *   **Urllib:** Python module for opening URLs and accessing internet data.
*   **Installation of Packages:** Instructions provided for installing packages via `pip` in the command terminal.

### **System Design**
*   **Architecture Diagram:** Illustrates the flow from Input Traffic through Pre-processing, Feature Engineering, Training, and Testing to the Detection Model, classifying output as Normal or Anomaly.
*   **UML Diagrams:**
    *   **Use Case Diagram:** Summarizes system users (actors) and their interactions with the system.
    *   **Sequence Diagram:** Describes the order and interaction of objects (e.g., Uploading Dataset, Pre-processing Data, Feature Engineering, Data Sets Train & Test, Data Analysis, Modelling, Data Pre-processing Results).
    *   **Class Diagram:** Shows the static structure of the system, including classes like System, Pre-Processing, Data Set, and Feature Engineering, with their methods and members.

### **Source Code**
*   Includes Python code using Flask, NumPy, and joblib for loading a pre-trained machine learning model.
*   The Flask application handles web requests, processes input features, makes predictions (Normal, DOS, PROBE, R2L, U2R), and renders HTML output.
*   The HTML form allows users to input various network parameters (attack type, count, rates, flags, logged-in status, service) for prediction.

### **Screenshots**
*   Provides visual examples of the system's output interface.

### **System Study (Feasibility Study)**
*   **Economical Feasibility:** The project is economically viable as most technologies used are freely available, keeping development costs within budget.
*   **Technical Feasibility:** The system has modest technical requirements, requiring minimal changes for implementation.
*   **Social Feasibility:** The system is designed for user acceptance, with training processes to ensure efficient use and build user confidence.

### **Implementation Modules**
1.  **Data Collection:** Gathering sufficient data samples (legitimate and malicious).
2.  **Data Pre-processing:** Using data augmentation techniques for better performance.
3.  **Train and Test Modelling:** Splitting data for model training and performance testing.
4.  **Attack Detection Model:** The trained algorithm detects anomalous transactions.

### **System Testing**
*   **Purpose:** To discover errors and ensure the software meets requirements and user expectations.
*   **Types of Tests:**
    *   **Unit Testing:** Validates internal program logic and individual software units.
    *   **Integration Testing:** Tests integrated software components to ensure they run as one program and expose problems arising from component combinations.
    *   **Functional Testing:** Systematically demonstrates that functions meet specified business and technical requirements.
    *   **White Box Testing:** Tests with knowledge of the software's internal workings.
    *   **Black Box Testing:** Tests the software without knowledge of its internal workings, focusing on inputs and outputs.
    *   **Acceptance Testing:** Critical phase involving end-user participation to ensure functional requirements are met.
*   **Testing Methodologies:** Unit Testing, Integration Testing (Top-Down and Bottom-Up), User Acceptance Testing, Output Testing, Validation Testing.
*   **Test Data:** Utilizes both live test data (extracted from organizational files) and artificial test data (created for comprehensive testing of all combinations and formats).

### **Conclusion**
*   Deep learning algorithms generally outperform SVM, ANN, RF, and CNN in cyberattack detection based on the CICIDS2017 dataset.
*   The project aims to identify the most accurate algorithm for predicting cyberattacks by analyzing historical datasets.
*   Future work will involve using AI and deep learning with Apache Hadoop and Shimmer technologies for port scope attempts and other attack types.

### **Future Scope**
*   The model can be extended to compare various machine learning algorithms to select the one with the highest accuracy.
*   Future work can involve working with multiple datasets and attributes simultaneously to enhance the model's capabilities.
