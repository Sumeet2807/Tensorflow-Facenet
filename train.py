import util
import argparse

print("modules imported")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="people",
    help="path to input image")
ap.add_argument("-o", "--output", default="models/knn/trained_knn_model.clf",
    help="path to output model")
ap.add_argument("-n", "--neighbors", default=2,
    help="nearest neighbors")
args = vars(ap.parse_args())

train_directory = args["image"]
model_path = args["output"]
neighbors = args["neighbors"]

print("Training KNN classifier...")
classifier = util.train(train_directory, model_save_path=model_path, n_neighbors=neighbors)
print("Training complete!")


