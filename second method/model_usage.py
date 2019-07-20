import cv2
import tensorflow as tf

output_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']


def prepare(filepath):
    IMG_SIZE = 28
    img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    return new_arr.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("weights.model")
prediction = model.predict([prepare('image.jpg')])
print(prediction)
for i in range(25):
    print(output_labels[i], " : ", prediction[0][i], "\n")

print("GPU Available: ", tf.test.is_gpu_available())
