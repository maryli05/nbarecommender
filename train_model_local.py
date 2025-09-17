import joblib
from app6 import load_data, train_model  # 👈 import your existing functions

# Load data
print("📂 Loading data...")
df_train, df_validation, playoff_schedule = load_data()

# Train model with desired hyperparameters
print("🚀 Training LightFM model...")
model, mapping, item_feats, user_feats = train_model(
    df_train,
    loss="warp",   # or "bpr"
    comps=64,      # latent dimensions
    epochs=30      # training epochs
)

# Save the trained model + mappings
bundle = {
    "model": model,
    "mapping": mapping,
    "item_feats": item_feats,
    "user_feats": user_feats
}

joblib.dump(bundle, "lightfm_model.pkl")
print("✅ Model trained and saved to lightfm_model.pkl")
