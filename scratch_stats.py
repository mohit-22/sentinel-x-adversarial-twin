from backend.app.main import _APP_STATE

print("Payment Twin keys:", _APP_STATE.keys())
df = _APP_STATE.get("clean_history")
if df is not None:
    print("Clean History shape:", df.shape)
    print("Amounts:", df["amount"].describe().to_dict())
    print("Timestamps:", df["timestamp"].min(), "to", df["timestamp"].max())
    print("Unique Customers:", df["customer_id"].nunique())
    print("Unique Merchants:", df["merchant"].nunique())
