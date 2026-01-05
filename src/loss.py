import numpy as np

class Loss:
    def calculate(self,output,y):
        #calculates the data and regularization losses given model output and ground truth values.
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        #y_pred:The output from the softmax layer(probabilities)
        #y_true:The actual class labels(can be one-hot or simple integers)
        samples=len(y_pred)

        #1.clip data to prevent division by 0 (log(0)=-infinity)
        #we clip from 1e-7 to (1-1e-7)
        y_pred_clipped=np.clip(y_pred,1e-7,1 - 1e-7)

        #2.handle different target format
        if len(y_true.shape)==1:
            #sparse targets([0,1,1])
            #we use "fancy indexing" to grab the probability of the correct class only
            correct_confidences=y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape)==2:
            #ONE-HOT TARGETS([[1,0,0],[0,1,0]])
            #we multiply pred by true and sum the rows
            correct_confidences=np.sum(y_pred_clipped*y_true,axis=1)

        #calculate negative log likelihood
        negative_log_likelihoods=-np.log(correct_confidences)
        return negative_log_likelihoods