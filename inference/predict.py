"""Production inference: score new weeks with the trained mmm_demo model."""

from pathlib import Path

from utils.func import engineer_features, load_data, load_model, predict, save_predictions

INPUT_PATH = Path("data/inputs/mmm_demo_scoring_input.csv")
MODEL_PATH = Path("data/model/mmm_demo_model.pkl")
OUTPUT_PATH = Path("data/outputs/mmm_demo_predictions.csv")


def main() -> None:
    df = load_data(INPUT_PATH)
    df_features = engineer_features(df)
    model = load_model(MODEL_PATH)
    df["predicted_sales"] = predict(df_features, model)
    save_predictions(df[["date", "predicted_sales"]], OUTPUT_PATH)


if __name__ == "__main__":
    main()
