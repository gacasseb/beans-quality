import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from build_model import build_model
from make_prediction import make_prediction
from build_features import build_features

def main():
    while True:
        print("\nMenu:")
        print("1 - Build model")
        print("2 - Make prediction")
        print("3 - Build features")
        print("4 - Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            build_model()
        elif choice == "2":
            make_prediction()
        elif choice == "3":
            bean_parameter = input("Choose a bean parameter (L, a or b): ")
            operation = input("Choose the operation (min, max or mean): ")
            dataset = input("Choose the dataset (1, 2 or 3): ")
            build_features(bean_parameter, operation, dataset)
        elif choice == "4":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please choose again.")

if __name__ == "__main__":
    main()
