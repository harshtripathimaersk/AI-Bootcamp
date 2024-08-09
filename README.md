# Business Objective: 

At our Maersk APMT Terminal there are several engineers that are deployed on site for the maintenance of the site and ships which sometimes require them to get the spare parts form the inventory. Now we are providing a solution where the user will be providing an image of the needed spare part and the Machine Learning model running at the backend would be providing the user with the images of the spare part that are similar to the required one for the identification and inventory information.

Now to tackle above challenges we have implemented three approaches as follows: 

_Approach 1:_  **Using Redis and vision model like Resnet50**

_Approach 2:_  **Finetuning of Pretrained MOdels**

_Approach 3:_  **Custom Vision Transformer (ViT)**

## Architecture: 

### Approach 1: 
Below is the Architecture of this approach: 
![Untitled diagram-2024-08-08-170159](https://github.com/user-attachments/assets/3bc00cf7-b5b2-4b34-a19f-a106b5fd82c5)

### Approach 2: 
### ResNet-101

ResNet-101 is a deep residual network known for its strong performance on various image classification tasks. We fine-tuned this model on our dataset by freezing the initial layers and retraining the last few layers to adapt to our specific classification problem. It was trained using GPU P100 for 100 epochs.
### MobileNetV2

MobileNetV2 is a lightweight and efficient model designed for mobile and embedded vision applications. We fine-tuned this model to achieve high accuracy while maintaining efficiency.This Models was trained for 20 epochs using GPU T4.

The combined Architecture of this approach is as follows: 
![Untitled diagram-2024-08-08-172748](https://github.com/user-attachments/assets/ab3b2330-00c2-432f-80a7-cd7e850e175a)

### Approach 3:
### Custom Vision Transformer

The Vision Transformer (ViT) model represents a shift from CNN-based architectures to transformer-based architectures for vision tasks. We built a custom Vision Transformer from scratch and trained it on our dataset for 100 epoches using GPU T4.The architure of our Vision Transformer is as follows: 
![Untitled diagram-2024-08-08-172815](https://github.com/user-attachments/assets/71c3cdee-5f8d-43f4-ac6a-5c239e4b3a54)


## Dataset
The dataset used in this project consists of 14 different classes of images regarding PC parts and has around 3000+ images . The data was divided into training and validation sets using an 80/20 split.



