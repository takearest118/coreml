# -*- coding: utf-8 -*-

import tensorflow as tf

def main():
    print('Hello Core ML')
    keras_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        input_shape=(224, 224, 3,),
        classes=1000,
    )

    # Download class labels (from a separate file)
    import urllib
    label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    class_labels = urllib.request.urlopen(label_url).read().splitlines()
    class_labels = class_labels[1:] # remove the first class which is background
    assert len(class_labels) == 1000

    # make sure entries of class_labels are strings
    for i, label in enumerate(class_labels):
        if isinstance(label, bytes):
            class_labels[i] = label.decode("utf8")

    import coremltools as ct

    # Define the input type as image, 
    # set pre-processing parameters to normalize the image 
    # to have its values in the interval [-1,1] 
    # as expected by the mobilenet model
    image_input = ct.ImageType(shape=(1, 224, 224, 3,),
                            bias=[-1,-1,-1], scale=1/127)

    # set class labels
    classifier_config = ct.ClassifierConfig(class_labels)

    # Convert the model using the Unified Conversion API
    model = ct.convert(
        keras_model, inputs=[image_input], classifier_config=classifier_config,
    )

    # Set feature descriptions (these show up as comments in XCode)
    model.input_description["input_1"] = "Input image to be classified"
    model.output_description["classLabel"] = "Most likely image category"

    # Set model author name
    model.author = '"Original Paper: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen'

    # Set the license of the model
    model.license = "Please see https://github.com/tensorflow/tensorflow for license information, and https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet for the original source of the model."

    # Set a short description for the Xcode UI
    model.short_description = "Detects the dominant objects present in an image from a set of 1001 categories such as trees, animals, food, vehicles, person etc. The top-1 accuracy from the original publication is 74.7%."

    # Set a version for the model
    model.version = "2.0"

    # Use PIL to load and resize the image to expected size
    from PIL import Image
    example_image = Image.open("./images/daisy.jpg").resize((224, 224))

    # Make a prediction using Core ML
    out_dict = model.predict({"input_1": example_image})

    # Print out top-1 prediction
    print(out_dict["classLabel"])

    # Save model
    model.save("MobileNetV2.mlmodel")

    # Load a saved model
    loaded_model = ct.models.MLModel("MobileNetV2.mlmodel")


if __name__ == "__main__":
    main()
