# document-classifier
A document classifier that uses a network of CNNs, almost 100% vibecoded with Gemini

### How It Works:

1. **Initial CNN Generation**:  
   - A Convolutional Neural Network (CNN) is trained for each **parent folder** to recognize the broad topics or categories associated with that folder.

2. **Hierarchy of Decision Networks**:  
   - Inspired by **Geoff Hinton’s classic XOR and parity-calculation networks**, a **multi-level network** is constructed for each parent folder.  
   - This network evaluates whether an input file should be:  
     - **Classified at the current level** (if it broadly matches the parent folder’s topic or spans multiple subtopics).  
     - **Passed down to a lower-level subfolder** (if it fits a more specific subtopic).  

3. **Dynamic Routing**:  
   - The **CNN outputs** are used as inputs to the decision networks guide the decision-making process, determining the most appropriate level for classification.  
   - This ensures files are placed at the **right granularity**—either in high-level folders or deeper subfolders—based on their content.

### Usage:

```
$ python3 dc.py 
usage: dc.py [-h] {train-hierarchical,predict-hierarchical} ...

Command-line utility for hierarchical document categorization using TensorFlow.

positional arguments:
  {train-hierarchical,predict-hierarchical}
                        Available commands
    train-hierarchical  Train hierarchical document categorization models.
    predict-hierarchical
                        Predict categories for new documents using hierarchical models.

options:
  -h, --help            show this help message and exit
```
