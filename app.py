from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model

def main():
    data = load_data('data/housing.csv')
    processed_data = preprocess_data(data)

    model, X_test, y_test = train_model(processed_data)
    mse, r2 = evaluate_model(model, X_test, y_test)

    print("Model Training Completed Successfully!")
    print(f"Mean Squared Error: {mse}")
    print(f"R² Score: {r2}")

if __name__ == "__main__":
    main()
