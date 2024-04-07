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
            build_features()
        elif choice == "4":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please choose again.")

if __name__ == "__main__":
    main()