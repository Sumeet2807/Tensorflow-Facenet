import util
import argparse

print("modules imported")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default="people",
    help="path to input image")
ap.add_argument("-o", "--output", default="models/classifier",
    help="path to output model")
ap.add_argument("-n", "--neighbors", default=2,
    help="nearest neighbors")
ap.add_argument("-clf", "--classifier", default='KNN',
    help="classifier type - KNN or SVC")
args = vars(ap.parse_args())

train_directory = args["image"]
model_path = args["output"] + '_' + args['classifier'] + '.clf'
neighbors = int(args["neighbors"])
classifier = args['classifier']

print("Training classifier...")
classifier = util.train(train_directory, model_save_path=model_path, classifier=classifier, n_neighbors=neighbors)
print("Training complete!")


