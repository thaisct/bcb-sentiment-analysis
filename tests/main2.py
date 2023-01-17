########### Train a simple finbert pre-trained model - OOP

import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

class financialsentiment:

    """This code defines a function train_on_batch that takes in a batch of financial articles, 
    represented as a tensor of tokenized input IDs, a tensor of attention masks, and a tensor of labels. 
    
    It then trains the model on this batch by calculating the loss and backpropagating the gradients to update 
    the model's weights. The code then defines a training loop that loads the training data, splits it into batches,
    and trains the model on each batch using the `train_on_batch`"""

    def __init__(self):


        # Load the FinBERT model and tokenizer:
        self.model = TFAutoModel.from_pretrained("finbert")
        self.tokenizer = AutoTokenizer.from_pretrained("finbert")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    
    def preprocess_text(self, text, max_len):

        """
        Preprocess the financial articles by tokenizing them and padding/truncating them to a fixed length
        """

        input_ids = self.tokenizer.encode(text, max_length=max_len, pad_to_max_length=True)
        input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
        attention_mask = tf.cast(input_ids != 0, dtype=tf.int32)
        return input_ids, attention_mask

    
    def train_on_batch(self, input_ids, attention_mask, labels):

        """
        Function to train the model on a batch of financial articles
        """
        with tf.GradientTape() as tape:
            logits = self.model(input_ids, attention_mask, labels=labels)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

#class readhtml


DEBUG = True

if __name__ == "__main__":

    if DEBUG:

        fs = financialsentiment()

        # Define the training loop
        batch_size = 1 # How many sample articles your sample has
        max_len = 20 # The number of words/tokens in a sample text
        num_epochs = 3 # The number of epochs is a hyperparameter that you can adjust to control the length of training

        # Read a financial article and process it:
        article_texts = ["Fed Officials Fretted That Markets Would Misread Rate Slowdown",
                        "Coinbase Reaches $100 Million Settlement With New York Regulators", 
                        "Why Japan's Sudden Shift on Bond Purchases Dealt a Global Jolt"]
        labels = [-1, 1, 0] # The model is presented with a batch of training data and the corresponding labels, and it adjusts its weights to minimize the loss between its predictions and the true labels. The goal is for the model to learn to make accurate predictions for new, unseen data.

        # Loop over the number of epochs
        for epoch in range(num_epochs):

            # Loop over the training data in batches
            for i in range(0, len(article_texts), batch_size):

                # Get the current batch of data
                batch_texts = article_texts[i:i + batch_size]
                batch_labels = labels[i:i + batch_size] 

                # Preprocess the text
                input_ids, attention_mask = fs.preprocess_text(batch_texts[0], max_len) #batch_texts, max_len

                # Train on the batch
                fs.train_on_batch(input_ids, attention_mask, batch_labels[0]) #input_ids, attention_mask, batch_labels


        # Use the model to make a prediction on the processed article:
        # output = self.model(input_ids, attention_mask)
        # prediction = tf.sigmoid(output)
        # print(prediction)