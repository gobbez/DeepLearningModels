# Import the necessary layers from TensorFlow Keras
from tensorflow.keras.layers import Conv2D, Flatten

# Instantiate a Sequential model
model = Sequential()

# Add a convolutional layer of 32 filters of size 3x3
model.add(Conv2D(32, kernel_size=3, input_shape=(28, 28, 1), activation='relu'))

# Add another convolutional layer of 16 filters of size 3x3
model.add(Conv2D(16, kernel_size=3, activation='relu'))

# Flatten the output of the previous layers
model.add(Flatten())

# Add an output layer with 10 units (for digit classification) and softmax activation
model.add(Dense(10, activation='softmax'))
```

Next section:


```python
# Import the necessary modules
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.models import Model

# Get a reference to the first layer's output
first_layer_output = model.layers[0].output

# Create a new model that takes the input and uses only the output of the first layer
first_layer_model = Model(inputs=model.input, outputs=first_layer_output)

# Use this model to make predictions on X_test
activations = first_layer_model.predict(X_test)

# Plot the activations for the 15th filter (for the first digit in X_test)
axs[0].matshow(activations[0, :, :, 14], cmap='viridis')

# Do the same but for the 18th filter (now)
axs[1].matshow(activations[0, :, :, 18], cmap='viridis')
plt.show()
```

Next section:


```python
# Import the necessary modules
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the image with the correct target size for your model
img = image.load_img(img_path, target_size=(224, 224))

# Turn it into a numpy array
img_array = image.img_to_array(img)

# Expand the dimensions of the image (for batch processing)
img_expanded = np.expand_dims(img_array, axis=0)

# Pre-process the image in the same way as your original images
img_ready = preprocess_input(img_expanded)

# Create an instance of ResNet50 with 'imagenet' weights
model = ResNet50(weights='imagenet')

# Use ResNet50 to make predictions on the pre-processed image
preds = model.predict(img_ready)

# Decode the top 3 predictions (for better readability)
print('Predicted:', decode_predictions(preds, top=3)[0])
```

Next section:


```python
# Split the text into an array of words
words = text.split()

# Create a list to store sentences of length 4
sentences = []

# Iterate through the words and create sentences of length 4 (moving one word at a time)
for i in range(4, len(words)):
    sentences.append(' '.join(words[i-4:i]))

# Instantiate a Tokenizer instance
tokenizer = Tokenizer()

# Fit the Tokenizer to your data (sentences)
tokenizer.fit_on_texts(sentences)

# Convert sentences into sequences of numbers
sequences = tokenizer.texts_to_sequences(sentences)
print("Sentences: \n {} \n Sequences: \n {}".format(sentences[:5], sequences[:5]))
```

Next section:


```python
# Import the necessary layers from TensorFlow Keras
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Instantiate a Sequential model (for text generation)
model = Sequential()

# Add an Embedding layer with the correct parameters (input_dim, input_length, output_dim)
model.add(Embedding(input_dim=vocab_size, input_length=3, output_dim=8))

# Add a 32 unit LSTM layer
model.add(LSTM(32))

# Add a hidden Dense layer of 32 units and an output layer with vocab_size and softmax activation
model.add(Dense(32, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# Print the summary of your model
model.summary()
```

Finally:


```python
def predict_text(test_text, model=model):
    # Check if the input text is 3 words (required for prediction)
    if len(test_text.split()) != 3:
        print('Text input should be 3 words!')
        return False

    # Turn the test_text into a sequence of numbers using your tokenizer
    test_seq = tokenizer.texts_to_sequences([test_text])
    test_seq = np.array(test_seq)

    # Use the model to make predictions on the pre-processed text sequence
    pred = model.predict(test_seq).argmax(axis=1)[0]

    # Return the word that maps to the prediction (for better readability)
    return tokenizer.index_word[pred]
